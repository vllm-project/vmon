// SPDX-License-Identifier: Apache-2.0

//! Atomic, private report output.
use std::io::Write;
use std::path::Path;

/// Stage reports in an exclusively created private file, then atomically replace
/// the destination directory entry (never follow an existing output symlink).
pub(crate) fn write_report_atomic(path: &Path, content: &str) -> std::io::Result<()> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    tmp.write_all(content.as_bytes())?;
    tmp.as_file().sync_all()?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};

    #[test]
    fn replaces_links_without_overwriting_their_targets() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        let output = dir.path().join("report.html");
        std::fs::write(&victim, "keep me").unwrap();
        symlink(&victim, dir.path().join("report.html.tmp")).unwrap();
        symlink(&victim, &output).unwrap();
        write_report_atomic(&output, "new report").unwrap();
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep me");
        assert_eq!(std::fs::read_to_string(&output).unwrap(), "new report");
        assert!(!std::fs::symlink_metadata(&output).unwrap().file_type().is_symlink());
        assert_eq!(
            std::fs::metadata(&output).unwrap().permissions().mode() & 0o777,
            0o600
        );
        write_report_atomic(&output, "checkpoint").unwrap();
        assert_eq!(std::fs::read_to_string(&output).unwrap(), "checkpoint");
    }
}
