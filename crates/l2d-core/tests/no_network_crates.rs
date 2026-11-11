//! Guard for requirement 10.2: the tool sends nothing over the network, so no network
//! client may be linked. A feature or version change of a dependency (Polars above all)
//! can pull one in unnoticed; this test fails as soon as `Cargo.lock` names one.

use std::path::Path;

/// Crates that open network connections or exist only to serve ones that do.
const NETWORK_CRATES: &[&str] = &[
    "tokio",
    "reqwest",
    "hyper",
    "rustls",
    "object_store",
    "mio",
    "socket2",
    "h2",
    "native-tls",
    "openssl",
    "curl",
    "ureq",
    "quinn",
];

/// The package names in the text of a `Cargo.lock`.
fn package_names(lock: &str) -> Vec<&str> {
    lock.lines()
        .filter_map(|line| line.strip_prefix("name = \""))
        .filter_map(|rest| rest.strip_suffix('"'))
        .collect()
}

fn network_crates_in(lock: &str) -> Vec<&str> {
    package_names(lock)
        .into_iter()
        .filter(|name| NETWORK_CRATES.contains(name))
        .collect()
}

#[test]
fn cargo_lock_names_no_network_client_crate() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("Cargo.lock");
    let lock = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let names = package_names(&lock);
    assert!(
        names.contains(&"polars") && names.contains(&"l2d-core"),
        "{} does not look like the workspace lock file ({} packages)",
        path.display(),
        names.len()
    );
    let found = network_crates_in(&lock);
    assert!(
        found.is_empty(),
        "Cargo.lock names network crates {found:?}: a dependency or feature change linked \
         them (requirement 10.2). Check `cargo tree -i <crate>` and the Polars entry in the \
         workspace manifest."
    );
}

#[test]
fn the_guard_finds_a_network_crate_in_a_lock_file_that_has_one() {
    let lock = "[[package]]\nname = \"polars\"\nversion = \"0.53.0\"\n\n\
                [[package]]\nname = \"tokio\"\nversion = \"1.0.0\"\n\n\
                [[package]]\nname = \"tokio-util\"\nversion = \"0.7.0\"\n\n\
                [[package]]\nname = \"socket2\"\nversion = \"0.5.0\"\n";
    assert_eq!(network_crates_in(lock), ["tokio", "socket2"]);
    assert!(network_crates_in("[[package]]\nname = \"polars\"\n").is_empty());
}
