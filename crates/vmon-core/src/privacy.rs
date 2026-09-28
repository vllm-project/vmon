// SPDX-License-Identifier: Apache-2.0

//! Best-effort redaction before server configuration enters the UI state.

fn sensitive_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase().replace(['-', '.', ' '], "_");
    [
        "password",
        "passwd",
        "secret",
        "credential",
        "private_key",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
    ]
    .iter()
    .any(|part| key.contains(part))
        || key == "token"
        || key.ends_with("_token")
        || matches!(
            key.as_str(),
            "access_tokens" | "refresh_tokens" | "api_tokens"
        )
}

/// Mask common credential keys recursively, including embedded JSON strings.
pub fn redact(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            for (key, value) in object {
                if sensitive_key(key) {
                    *value = "[REDACTED]".into();
                } else {
                    redact(value);
                }
            }
        }
        serde_json::Value::Array(array) => array.iter_mut().for_each(redact),
        serde_json::Value::String(text) => {
            if let Ok(mut nested) = serde_json::from_str::<serde_json::Value>(text) {
                if nested.is_object() || nested.is_array() {
                    redact(&mut nested);
                    *text = nested.to_string();
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hides_nested_and_embedded_credentials_preserving_token_counts() {
        let mut value = serde_json::json!({
            "vllm_config": {"api_key": "example-credential", "max_tokens": 100},
            "vllm_env": {"HF_TOKEN": "example-credential"},
            "extra": "{\"authorization\":\"example-credential\"}",
            "list": [{"client_secret": "example-credential"}]
        });
        redact(&mut value);
        assert!(!value.to_string().contains("example-credential"));
        assert_eq!(value["vllm_config"]["max_tokens"], 100);
    }
}
