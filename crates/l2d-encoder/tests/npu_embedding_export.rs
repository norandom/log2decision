//! Writes the vectors the NPU gives for CSIC's distinct texts and for a set of probe
//! texts to files, for the embedding qualification of task 2.15
//! (`docs/2026-11-14-embedding-qualification.md`, scripts in
//! `tools/embedding-qualification/`). Through `EncoderSession` only; nothing here
//! touches `ort`, and no CPU provider runs.
//!
//! Files, in the directory `L2D_EXPORT_DIR` names (outside the repository):
//!
//! - `csic.f32`: the vectors, little-endian `f32`, `dimension` values per text, in the
//!   order of `csic.jsonl`;
//! - `csic.jsonl`: per text one JSON object: `i`, `label` (`good`/`bad`), `group` (the
//!   URL path), `truncated`, `text_longer_than_input`, `empty`, `part_bytes`,
//!   `read_bytes`, `text`;
//! - `probes.f32`, `probes.jsonl`: the same for the probe texts of [`probes`], with
//!   `name`, `kind`, `base` (the probe this one is a variant of) and `relation` instead
//!   of label and group;
//! - `export.json`: the encoder identity, the input mode, the runtime release, the
//!   driver, the placement and the counts.
//!
//! Run on request, in release (see `npu_support` for the variables):
//!
//! ```text
//! $env:L2D_RUNTIME_DIR = "D:\Source\log2decision\runtime\ryzenai"
//! $env:L2D_SCIENCE_DIR = "D:\Source\log2decision\models\science-2.1"
//! $env:L2D_CSIC_CSV = "D:\Source\log2decision\data\csic-2010\csic_database.csv"
//! $env:L2D_NPU_WORK_DIR = "D:\BuildArtifacts\log2decision\task-2.15"
//! $env:L2D_EXPORT_DIR = "D:\BuildArtifacts\log2decision\task-2.15\vectors"
//! cargo test --release -p l2d-encoder --test npu_embedding_export -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(windows)]

mod npu_support;

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use l2d_encoder::{EncoderSession, TextInput};
use npu_support::*;
use serde_json::json;

const TAG: &str = "npu_embedding_export";
/// The checksum tasks 2.12 and 2.6 printed for the vectors of CSIC's 25,608 distinct
/// texts from the published model: the vectors written here are those vectors.
const PUBLISHED_CHECKSUM: &str = "1570811.2676";

/// One probe text. `base` names the probe it is a variant of and `relation` says how
/// it differs; a probe without a base stands for itself.
struct Probe {
    name: String,
    kind: &'static str,
    base: Option<&'static str>,
    relation: &'static str,
    text: String,
}

/// A small generator with a fixed seed (xorshift), so the random probes are the same
/// in every run.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

const ANADIR: &str = "GET http://localhost:8080/tienda1/publico/anadir.jsp?";
const AUTENTICAR: &str = "GET http://localhost:8080/tienda1/publico/autenticar.jsp?";
const PAGAR: &str = "GET http://localhost:8080/tienda1/publico/pagar.jsp?";
const REGISTRO: &str = "POST http://localhost:8080/tienda1/publico/registro.jsp\n";
const CART: &str = "id=2&nombre=Jam%F3n+Ib%E9rico&precio=85&cantidad=49&B1=A%F1adir+al+carrito";
const LOGIN: &str = "modo=entrar&login=bob&pwd=s3creto&remember=off&B1=Entrar";
const PAY: &str = "modo=insertar&precio=1764&B1=Pasar+por+caja";
const REGISTER: &str = "modo=registro&login=alice&password=tr3boles&nombre=Alicia&\
                        apellidos=Mora+Vega&email=alicia%40correo.es&dni=41372566K&\
                        direccion=Calle+Mayor+12&ciudad=Lugo&cp=27001&provincia=Lugo&\
                        ntc=4539047372198823&B1=Registrar";

/// The attack values, by class, in the form a query string carries them.
const ATTACKS: [(&str, &str, &str); 8] = [
    ("sqli", "or", "49%27+OR+%271%27%3D%271"),
    (
        "sqli",
        "drop",
        "%27%3B+DROP+TABLE+usuarios%3B+SELECT+*+FROM+datos+WHERE+nombre+LIKE+%27%25",
    ),
    (
        "xss",
        "script",
        "%3Cscript%3Ealert%28document.cookie%29%3C%2Fscript%3E",
    ),
    ("xss", "img", "%3Cimg+src%3Dx+onerror%3Dalert%281%29%3E"),
    ("traversal", "passwd", "..%2F..%2F..%2F..%2Fetc%2Fpasswd"),
    ("traversal", "winini", "..%5C..%5C..%5Cwindows%5Cwin.ini%00"),
    ("cmdi", "cat", "49%3Bcat+%2Fetc%2Fpasswd"),
    ("cmdi", "ping", "49%7Cping+-c+3+127.0.0.1"),
];

/// The probe texts, in a fixed order. Names are unique.
fn probes() -> Vec<Probe> {
    let mut all = Vec::new();
    let mut add = |name: String,
                   kind: &'static str,
                   base: Option<&'static str>,
                   relation: &'static str,
                   text: String| {
        all.push(Probe {
            name,
            kind,
            base,
            relation,
            text,
        });
    };

    // Benign requests in CSIC's layout. `cart` is on an endpoint the published model
    // was not fitted on (`anadir.jsp`); `search` is not a CSIC endpoint at all.
    add(
        "cart".into(),
        "benign",
        None,
        "base",
        format!("{ANADIR}{CART}"),
    );
    add(
        "login".into(),
        "benign",
        None,
        "base",
        format!("{AUTENTICAR}{LOGIN}"),
    );
    add(
        "pay".into(),
        "benign",
        None,
        "base",
        format!("{PAGAR}{PAY}"),
    );
    add(
        "register".into(),
        "benign",
        None,
        "base",
        format!("{REGISTRO}{REGISTER}"),
    );
    add(
        "search".into(),
        "benign",
        None,
        "base",
        "GET http://localhost:8080/search?q=red+shoes&page=2&sort=price".into(),
    );

    // Exact duplicates.
    add(
        "cart-again".into(),
        "duplicate",
        Some("cart"),
        "duplicate",
        format!("{ANADIR}{CART}"),
    );
    add(
        "login-again".into(),
        "duplicate",
        Some("login"),
        "duplicate",
        format!("{AUTENTICAR}{LOGIN}"),
    );
    add(
        "register-again".into(),
        "duplicate",
        Some("register"),
        "duplicate",
        format!("{REGISTRO}{REGISTER}"),
    );

    // Near duplicates: one character, one parameter value, the case, the order.
    add(
        "cart-one-char".into(),
        "near",
        Some("cart"),
        "one character",
        format!("{ANADIR}{}", CART.replace("cantidad=49", "cantidad=48")),
    );
    add(
        "cart-one-char-id".into(),
        "near",
        Some("cart"),
        "one character",
        format!("{ANADIR}{}", CART.replace("id=2", "id=3")),
    );
    add(
        "login-one-char".into(),
        "near",
        Some("login"),
        "one character",
        format!("{AUTENTICAR}{}", LOGIN.replace("login=bob", "login=rob")),
    );
    add(
        "cart-one-value".into(),
        "near",
        Some("cart"),
        "one value",
        format!(
            "{ANADIR}{}",
            CART.replace("Jam%F3n+Ib%E9rico", "Queso+Manchego")
        ),
    );
    add(
        "login-one-value".into(),
        "near",
        Some("login"),
        "one value",
        format!(
            "{AUTENTICAR}{}",
            LOGIN.replace("pwd=s3creto", "pwd=caballo77")
        ),
    );
    add(
        "register-one-value".into(),
        "near",
        Some("register"),
        "one value",
        format!("{REGISTRO}{}", REGISTER.replace("Lugo&cp", "Vigo&cp")),
    );
    add(
        "cart-upper".into(),
        "near",
        Some("cart"),
        "case",
        format!("{ANADIR}{}", CART.to_uppercase()),
    );
    add(
        "login-upper".into(),
        "near",
        Some("login"),
        "case",
        format!("{AUTENTICAR}{}", LOGIN.to_uppercase()),
    );
    add(
        "cart-reordered".into(),
        "near",
        Some("cart"),
        "parameter order",
        format!(
            "{ANADIR}cantidad=49&precio=85&nombre=Jam%F3n+Ib%E9rico&id=2&B1=A%F1adir+al+carrito"
        ),
    );

    // The same payload on another endpoint and with another method. In `payload` mode
    // the encoder does not read method or path, so these have the input of their base.
    add(
        "cart-other-endpoint".into(),
        "endpoint",
        Some("cart"),
        "same payload, other endpoint",
        format!("GET http://localhost:8080/tienda1/miembros/editar.jsp?{CART}"),
    );
    add(
        "cart-other-site".into(),
        "endpoint",
        Some("cart"),
        "same payload, other endpoint",
        format!("GET https://shop.example.org/api/v2/basket.php?{CART}"),
    );
    add(
        "cart-as-post".into(),
        "endpoint",
        Some("cart"),
        "same payload, other endpoint",
        format!("POST http://localhost:8080/tienda1/publico/anadir.jsp\n{CART}"),
    );

    // Attacks: each value once in the cart request (parameter `cantidad`), once in the
    // login request (`pwd`) and once in the payment request (`precio`).
    for (class, name, value) in ATTACKS {
        let class_kind = match class {
            "sqli" => "attack sqli",
            "xss" => "attack xss",
            "traversal" => "attack traversal",
            _ => "attack cmdi",
        };
        add(
            format!("{class}-{name}-cart"),
            class_kind,
            Some("cart"),
            "attack in a benign request",
            format!(
                "{ANADIR}{}",
                CART.replace("cantidad=49", &format!("cantidad={value}"))
            ),
        );
        add(
            format!("{class}-{name}-login"),
            class_kind,
            Some("login"),
            "attack in a benign request",
            format!(
                "{AUTENTICAR}{}",
                LOGIN.replace("pwd=s3creto", &format!("pwd={value}"))
            ),
        );
        add(
            format!("{class}-{name}-pay"),
            class_kind,
            Some("pay"),
            "attack in a benign request",
            format!(
                "{PAGAR}{}",
                PAY.replace("precio=1764", &format!("precio={value}"))
            ),
        );
    }

    // Random texts as the payload: printable ASCII, percent-encoded bytes, and
    // characters outside ASCII.
    let mut random = Random(0x2015_2026_1114_0001);
    for n in 0..6 {
        let text: String = (0..120)
            .map(|_| char::from(b'!' + random.below(94) as u8))
            .filter(|c| *c != '?')
            .collect();
        add(
            format!("random-printable-{n}"),
            "random",
            None,
            "base",
            format!("GET http://localhost:8080/r?{text}"),
        );
    }
    for n in 0..4 {
        let text: String = (0..60)
            .map(|_| format!("%{:02X}", random.below(256)))
            .collect();
        add(
            format!("random-percent-bytes-{n}"),
            "random",
            None,
            "base",
            format!("GET http://localhost:8080/r?{text}"),
        );
    }
    for n in 0..4 {
        let text: String = (0..80)
            .map(|_| char::from_u32(0xA1 + random.below(0x500) as u32).unwrap_or('x'))
            .collect();
        add(
            format!("random-non-ascii-{n}"),
            "random",
            None,
            "base",
            format!("GET http://localhost:8080/r?{text}"),
        );
    }

    // Empty and very short texts. Without `?` and without a line feed a text has no
    // payload, so its input is empty.
    for (name, text) in [
        ("empty", ""),
        ("one-letter", "a"),
        ("question-mark", "?"),
        ("short-payload-a", "?a"),
        ("short-payload-a1", "?a=1"),
        ("short-payload-b1", "?b=1"),
        (
            "no-payload-index",
            "GET http://localhost:8080/tienda1/index.jsp",
        ),
        (
            "no-payload-syslog",
            "Nov 14 08:13:02 host sshd[4121]: Failed password for root from 10.0.0.7 port 51122 ssh2",
        ),
        (
            "no-payload-attack-in-path",
            "GET http://localhost:8080/tienda1/../../../../etc/passwd",
        ),
    ] {
        add(name.into(), "short", None, "base", text.into());
    }

    // Long texts (the input is the first 256 bytes of the payload).
    let filler = |seed: u64, bytes: usize| -> String {
        let mut random = Random(seed);
        let mut text = String::new();
        let mut n = 0;
        while text.len() < bytes {
            text.push_str(&format!("&campo{n}=valor{}", random.below(100_000)));
            n += 1;
        }
        text
    };
    let head = format!("{CART}{}", filler(7, 256));
    let head = &head[..256];
    add(
        "long-a".into(),
        "long",
        None,
        "base",
        format!("{ANADIR}{head}{}", filler(11, 1800)),
    );
    add(
        "long-same-first-256".into(),
        "long",
        Some("long-a"),
        "differs after byte 256",
        format!("{ANADIR}{head}{}", filler(13, 1800)),
    );
    add(
        "long-attack-after-256".into(),
        "long",
        Some("long-a"),
        "differs after byte 256",
        format!(
            "{ANADIR}{head}&x=%27%3B+DROP+TABLE+usuarios%3B--{}",
            filler(11, 1800)
        ),
    );
    add(
        "long-attack-before-256".into(),
        "long",
        Some("long-a"),
        "attack in a benign request",
        format!(
            "{ANADIR}{}{}",
            head.replace("cantidad=49", "cantidad=%27%3B+DROP+TABLE+usuarios%3B--"),
            filler(11, 1800)
        ),
    );
    add(
        "long-other".into(),
        "long",
        None,
        "base",
        format!("{ANADIR}{}", filler(17, 2000).trim_start_matches('&')),
    );
    all
}

#[test]
fn the_probe_texts_are_fixed_and_their_names_unique() {
    let first = probes();
    let second = probes();
    assert_eq!(first.len(), second.len());
    assert!(
        first
            .iter()
            .zip(&second)
            .all(|(a, b)| a.name == b.name && a.text == b.text)
    );
    let names: BTreeSet<&str> = first.iter().map(|probe| probe.name.as_str()).collect();
    assert_eq!(names.len(), first.len(), "probe names are unique");
    for probe in &first {
        if let Some(base) = probe.base {
            assert!(names.contains(base), "{}: unknown base {base}", probe.name);
        }
        assert!(!probe.text.contains('\0'), "{}: a NUL byte", probe.name);
    }
    let duplicate = |a: &str, b: &str| {
        let text = |name: &str| &first.iter().find(|p| p.name == name).unwrap().text;
        text(a) == text(b)
    };
    assert!(duplicate("cart", "cart-again"));
    assert!(!duplicate("cart", "cart-one-char"));
}

fn export_dir() -> Option<PathBuf> {
    match std::env::var_os("L2D_EXPORT_DIR").filter(|value| !value.is_empty()) {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            assert!(
                dir.parent().is_some() && dir.components().count() > 2,
                "L2D_EXPORT_DIR must not be a drive root: {}",
                dir.display()
            );
            std::fs::create_dir_all(&dir)
                .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));
            Some(dir)
        }
        None => {
            skip(
                TAG,
                "L2D_EXPORT_DIR is not set (the directory the vector files go to)",
            );
            None
        }
    }
}

fn write_vectors(path: &Path, vectors: &[f32]) {
    let mut bytes = Vec::with_capacity(vectors.len() * 4);
    for value in vectors {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
}

fn write_lines(path: &Path, lines: &[serde_json::Value]) {
    let file =
        std::fs::File::create(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut out = std::io::BufWriter::new(file);
    for line in lines {
        writeln!(out, "{line}").unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    }
    out.flush()
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
}

fn encode(session: &mut EncoderSession, texts: &[&str]) -> (Vec<f32>, Vec<TextInput>) {
    let dimension = session.info().identity.dimension as usize;
    let mut vectors = Vec::with_capacity(texts.len() * dimension);
    let mut flags = Vec::with_capacity(texts.len());
    session
        .encode_into(texts, &mut vectors, &mut flags)
        .unwrap_or_else(|error| panic!("{error}"));
    (vectors, flags)
}

fn checksum(vectors: &[f32]) -> f64 {
    vectors
        .iter()
        .enumerate()
        .map(|(at, value)| f64::from(*value) * (1.0 + (at % 97) as f64))
        .sum()
}

fn flag_fields(input: &TextInput) -> serde_json::Value {
    json!({
        "truncated": input.truncated,
        "text_longer_than_input": input.text_longer_than_input,
        "empty": input.empty,
        "part_bytes": input.part_bytes,
        "read_bytes": input.read_bytes,
    })
}

fn with(mut fields: serde_json::Value, more: serde_json::Value) -> serde_json::Value {
    let (Some(fields_map), Some(more)) = (fields.as_object_mut(), more.as_object()) else {
        panic!("two JSON objects");
    };
    for (key, value) in more {
        fields_map.insert(key.clone(), value.clone());
    }
    fields
}

#[test]
#[ignore = "hardware run: needs the AMD runtime files, the NPU and the trained model; run on request"]
fn the_vectors_of_csic_and_of_the_probes_are_written() {
    let Some(paths) = session_project(TAG) else {
        return;
    };
    let Some(science) = directory_from(TAG, "L2D_SCIENCE_DIR", "the published model of task 2.1")
    else {
        return;
    };
    let Some(out) = export_dir() else {
        return;
    };
    let package = published_package_in(&paths, &science);
    let logs = csic_logs();
    let probes = probes();
    let mut session =
        EncoderSession::open(&package, &paths).unwrap_or_else(|error| panic!("{error}"));
    let info = session.info().clone();
    let dimension = usize::from(package.identity().dimension);
    println!(
        "{TAG}: encoder {} ({} mode) on the NPU; AMD Ryzen AI {}, driver {}; placement {} of {} \
         nodes on the NPU; first run through the Vitis AI provider {:?}, elsewhere {:?}",
        info.identity,
        info.input_mode.as_str(),
        info.runtime_release,
        info.driver_version.as_deref().unwrap_or("unknown"),
        info.placement.npu_nodes,
        info.placement.all_nodes,
        info.executed.on_provider,
        info.executed.elsewhere
    );

    let texts: Vec<&str> = logs.iter().map(|log| log.text.as_str()).collect();
    let (vectors, flags) = encode(&mut session, &texts);
    assert_eq!(vectors.len(), logs.len() * dimension);
    let sum = checksum(&vectors);
    println!(
        "{TAG}: {} distinct CSIC texts, {dimension} values each; checksum {sum:.4} (tasks 2.12 \
         and 2.6: {PUBLISHED_CHECKSUM}); truncated {}, whole text longer than the input {}, \
         empty input {}",
        logs.len(),
        flags.iter().filter(|input| input.truncated).count(),
        flags
            .iter()
            .filter(|input| input.text_longer_than_input)
            .count(),
        flags.iter().filter(|input| input.empty).count()
    );
    assert_eq!(
        format!("{sum:.4}"),
        PUBLISHED_CHECKSUM,
        "these are not the vectors tasks 2.12 and 2.6 measured for the published model"
    );
    // A second pass gives the same bits: what is written does not depend on the pass.
    let (again, _) = encode(&mut session, &texts);
    let differing = vectors
        .chunks_exact(dimension)
        .zip(again.chunks_exact(dimension))
        .filter(|(a, b)| a.iter().zip(*b).any(|(x, y)| x.to_bits() != y.to_bits()))
        .count();
    println!("{TAG}: second pass over CSIC: {differing} vectors differ in any bit");
    assert_eq!(differing, 0);

    let lines: Vec<serde_json::Value> = logs
        .iter()
        .zip(&flags)
        .enumerate()
        .map(|(i, (log, input))| {
            with(
                json!({
                    "i": i,
                    "label": if log.bad { "bad" } else { "good" },
                    "group": log.group,
                }),
                with(flag_fields(input), json!({ "text": log.text })),
            )
        })
        .collect();
    write_vectors(&out.join("csic.f32"), &vectors);
    write_lines(&out.join("csic.jsonl"), &lines);

    let probe_texts: Vec<&str> = probes.iter().map(|probe| probe.text.as_str()).collect();
    let (probe_vectors, probe_flags) = encode(&mut session, &probe_texts);
    // The probes one at a time give the same bits as the probes together.
    let mut alone = Vec::new();
    for text in &probe_texts {
        alone.extend(encode(&mut session, &[text]).0);
    }
    let differing = probe_vectors
        .iter()
        .zip(&alone)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "{TAG}: {} probe texts; encoded one at a time: {differing} values differ in any bit",
        probes.len()
    );
    assert_eq!(differing, 0);
    let lines: Vec<serde_json::Value> = probes
        .iter()
        .zip(&probe_flags)
        .enumerate()
        .map(|(i, (probe, input))| {
            with(
                json!({
                    "i": i,
                    "name": probe.name,
                    "kind": probe.kind,
                    "base": probe.base,
                    "relation": probe.relation,
                }),
                with(flag_fields(input), json!({ "text": probe.text })),
            )
        })
        .collect();
    write_vectors(&out.join("probes.f32"), &probe_vectors);
    write_lines(&out.join("probes.jsonl"), &lines);

    let record = json!({
        "task": "npu-log-decision 2.15",
        "encoder": info.identity.to_string(),
        "input_mode": info.input_mode.as_str(),
        "dimension": dimension,
        "device": "npu",
        "runtime_release": info.runtime_release.to_string(),
        "driver_version": info.driver_version,
        "placement": {
            "npu_nodes": info.placement.npu_nodes,
            "cpu_nodes": info.placement.cpu_nodes,
            "all_nodes": info.placement.all_nodes,
            "first_run_through_provider": format!("{:?}", info.executed.on_provider),
            "first_run_elsewhere": format!("{:?}", info.executed.elsewhere),
        },
        "csic": { "texts": logs.len(), "checksum": format!("{sum:.4}") },
        "probes": { "texts": probes.len() },
        "layout": "little-endian f32, `dimension` values per text, in the order of the .jsonl file",
    });
    let text = serde_json::to_string_pretty(&record).unwrap_or_else(|error| panic!("{error}"));
    std::fs::write(out.join("export.json"), text).unwrap_or_else(|error| panic!("{error}"));
    println!(
        "{TAG}: wrote csic.f32, csic.jsonl, probes.f32, probes.jsonl and export.json to {}",
        out.display()
    );
}
