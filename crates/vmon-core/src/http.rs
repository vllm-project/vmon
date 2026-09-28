// SPDX-License-Identifier: Apache-2.0

//! Bounded reads for endpoints outside the process trust boundary.

/// Maximum body size accepted from one metrics or configuration endpoint.
pub const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ResponseError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("response exceeds {0} bytes")]
    TooLarge(usize),
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
}

async fn read(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>, ResponseError> {
    if response.content_length().is_some_and(|len| len > limit as u64) {
        return Err(ResponseError::TooLarge(limit));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if chunk.len() > limit.saturating_sub(bytes.len()) {
            return Err(ResponseError::TooLarge(limit));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// Read UTF-8 text with an enforced body limit, even for chunked responses.
pub async fn text(response: reqwest::Response) -> Result<String, ResponseError> {
    Ok(String::from_utf8_lossy(&read(response, MAX_RESPONSE_BYTES).await?).into_owned())
}

/// Deserialize JSON only after enforcing the body limit.
pub async fn json<T: serde::de::DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, ResponseError> {
    Ok(serde_json::from_slice(
        &read(response, MAX_RESPONSE_BYTES).await?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn response(wire: &'static [u8]) -> reqwest::Response {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0; 4096];
            let _ = socket.read(&mut buf).await;
            socket.write_all(wire).await.unwrap();
        });
        reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap()
            .get(format!("http://{addr}"))
            .send()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn rejects_declared_and_chunked_oversized_bodies() {
        for wire in [
            &b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n123456789"[..],
            &b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n12345\r\n4\r\n6789\r\n0\r\n\r\n"[..],
        ] {
            assert!(matches!(read(response(wire).await, 8).await, Err(ResponseError::TooLarge(8))));
        }
        assert_eq!(
            read(
                response(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\n12345678").await,
                8
            )
            .await
            .unwrap(),
            b"12345678"
        );
    }
}
