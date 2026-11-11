//! Warns when build output lands inside the repository.
//!
//! The host convention is an off-repo `CARGO_TARGET_DIR`. The path is not hard-coded in
//! repository configuration, so this check only warns and never fails the build.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_DIR");

    let (Some(manifest_dir), Some(out_dir)) = (
        env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from),
        env::var_os("OUT_DIR").map(PathBuf::from),
    ) else {
        return;
    };
    // crates/l2d-core -> workspace root
    let Some(workspace_root) = manifest_dir.parent().and_then(Path::parent) else {
        return;
    };
    if out_dir.starts_with(workspace_root) {
        println!(
            "cargo:warning=build output is inside the repository ({}); set CARGO_TARGET_DIR to an off-repo directory",
            out_dir.display()
        );
    }
}
