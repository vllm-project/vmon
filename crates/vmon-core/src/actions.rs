// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

/// Client for vLLM dev-mode endpoints (`VLLM_SERVER_DEV_MODE=1`).
pub struct NodeActions {
    client: reqwest::Client,
}

#[derive(Debug, thiserror::Error)]
pub enum ActionError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Node returned error: {status} — {body}")]
    Server { status: u16, body: String },
}

impl Default for NodeActions {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeActions {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");
        Self { client }
    }

    /// Check if the engine is sleeping. Returns None if endpoint unavailable.
    pub async fn is_sleeping(&self, addr: &str) -> Option<bool> {
        let resp = self
            .client
            .get(format!("http://{addr}/is_sleeping"))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        // vLLM returns JSON: true or false
        let text = crate::http::text(resp).await.ok()?;
        text.trim().parse::<bool>().ok()
    }

    /// Put the engine to sleep.
    pub async fn sleep(&self, addr: &str) -> Result<(), ActionError> {
        let resp = self
            .client
            .post(format!("http://{addr}/sleep"))
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status().as_u16();
            let body = crate::http::text(resp).await.unwrap_or_default();
            Err(ActionError::Server { status, body })
        }
    }

    /// Wake the engine from sleep.
    pub async fn wake_up(&self, addr: &str) -> Result<(), ActionError> {
        let resp = self
            .client
            .post(format!("http://{addr}/wake_up"))
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status().as_u16();
            let body = crate::http::text(resp).await.unwrap_or_default();
            Err(ActionError::Server { status, body })
        }
    }

    /// Reset the prefix cache.
    pub async fn reset_prefix_cache(&self, addr: &str) -> Result<(), ActionError> {
        let resp = self
            .client
            .post(format!("http://{addr}/reset_prefix_cache"))
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status().as_u16();
            let body = crate::http::text(resp).await.unwrap_or_default();
            Err(ActionError::Server { status, body })
        }
    }
}
