"""Expected values for the Rust `ByteCodec` (task 2.5), made with the science gate's own code.

Development only. Writes the JSON file that the tests of `l2d-encoder` read:

- `byte-codec-expected.json`: for each text and each input mode the byte ids
  (`l2dsci.byte_ids`), the truncation flag, and the SHA-256 of that text's part of the
  graph input (`l2dsci.table_input`, float32 little-endian, `[64, 1, 256]` in C order);
  and the SHA-256 of every batch `[32, 64, 1, 256]` the pipeline feeds the model, the
  last one padded with all-padding rows as `l2dsci.onnx_vectors` pads it.
Run from `tools\\science` (the table is copied next to the fixtures by hand, see the
README):

    uv run python codec_fixture.py D:\\Source\\log2decision\\models\\science-2.1 ..\\..\\crates\\l2d-encoder\\tests\\fixtures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

import l2dsci

RUN = "split17-f0-s0-strip-bce-norm-unit"
# The Rust input modes and the routing mode of `l2dsci.byte_ids` each one must equal.
MODES = {"whole": "keep", "payload": "strip"}

def texts() -> list[tuple[str, str]]:
    """(name, text). Lengths are in UTF-8 bytes; 256 is the input length."""
    a = "abcdefghijklmnopqrstuvwxyz0123456789"

    def ascii_of(n: int) -> str:
        return (a * (n // len(a) + 1))[:n]

    high = "".join(chr(c) for c in range(0x80, 0x100))          # 2-byte characters, 256 bytes
    out = [
        ("empty", ""),
        ("ascii-no-delimiter", "GET http://localhost:8080/tienda1/index.jsp"),
        ("csic-get-query",
         "GET http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=Jam%F3n+Ib%E9rico"
         "&precio=85&cantidad=%27%3B+DROP+TABLE+usuarios%3B+SELECT+*+FROM+datos+WHERE+nombre"
         "+LIKE+%27%25&B1=A%F1adir+al+carrito"),
        ("csic-post-body",
         "POST http://localhost:8080/tienda1/publico/autenticar.jsp\n"
         "modo=entrar&login=bob%40%3CSCRipt%3Ealert%28Paros%29%3C%2FscrIPT%3E.parosproxy.org"
         "&pwd=84m3ri156&remember=off&B1=Entrar"),
        ("query-and-body", "POST /a/b.jsp?x=1\ny=2&z=3"),
        ("body-before-query", "POST /a/b.jsp\ny=2?z=3\nw=4"),
        ("only-question-mark", "?"),
        ("only-line-feed", "\n"),
        ("delimiter-last", "GET /a/b.jsp?"),
        ("delimiter-first", "?a=1&b=2"),
        ("crlf", "GET /a HTTP/1.1\r\nHost: x\r\n\r\nbody"),
        ("ordinary-log-line", "Nov 13 10:02:11 host sshd[411]: Failed password for root from 10.0.0.7 port 22"),
        ("whole-255", ascii_of(255)),
        ("whole-256", ascii_of(256)),
        ("whole-257", ascii_of(257)),
        ("whole-300", ascii_of(300)),
        ("whole-5000", ascii_of(5000)),
        ("payload-255", "GET /p?" + ascii_of(255)),
        ("payload-256", "GET /p?" + ascii_of(256)),
        ("payload-257", "GET /p?" + ascii_of(257)),
        ("payload-300", "POST /p\n" + ascii_of(300)),
        # Multi-byte characters at the cut. The model was trained on raw bytes cut at 256.
        ("whole-2byte-ends-at-256", ascii_of(254) + "\u00e9"),
        ("whole-2byte-cut-at-256", ascii_of(255) + "\u00e9"),
        ("whole-3byte-cut-after-1", ascii_of(255) + "\u20ac"),
        ("whole-3byte-cut-after-2", ascii_of(254) + "\u20ac"),
        ("whole-4byte-cut-after-1", ascii_of(255) + "\U0001f600"),
        ("whole-4byte-cut-after-3", ascii_of(253) + "\U0001f600"),
        ("whole-4byte-ends-at-256", ascii_of(252) + "\U0001f600"),
        ("payload-2byte-cut-at-256", "GET /p?" + ascii_of(255) + "\u00e9"),
        ("payload-3byte-cut-after-2", "POST /p\n" + ascii_of(254) + "\u20ac tail"),
        ("payload-4byte-cut-after-3", "GET /p?" + ascii_of(253) + "\U0001f600\U0001f600"),
        ("payload-2byte-ends-at-256", "GET /p?" + ascii_of(254) + "\u00e9"),
        ("multibyte-routing", "GET /caf\u00e9/\u20ac?n=\u00f1&e=\U0001f600"),
        ("high-characters", high),
        ("high-characters-payload", "GET /h?" + high[:100] + "\u0800\uffff\U00010000\U0010ffff"),
        # A NUL byte has the padding id.
        ("nul-in-payload", "GET /n?x=\x00y"),
        ("nul-in-routing", "GET /n\x00/a?x=1"),
        ("nul-only", "\x00\x00\x00"),
        ("nul-at-255", ascii_of(255) + "\x00"),
        ("nul-payload-only", "GET /n?\x00\x00"),
        ("control-bytes", "\x01\x02\t\x7f \x1b[0m"),
    ]
    names = [name for name, _ in out]
    assert len(set(names)) == len(names)
    assert len(out) > l2dsci.BATCH and len(out) % l2dsci.BATCH != 0
    return out


def sha256(array: np.ndarray) -> str:
    assert array.dtype == np.dtype("<f4") and array.flags["C_CONTIGUOUS"]
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", help="directory of the published science-2.1 artifacts")
    parser.add_argument("out", help="directory of the fixtures (crates/l2d-encoder/tests/fixtures)")
    args = parser.parse_args()

    stem = os.path.join(args.models, RUN)
    table_path = stem + ".table.bin"
    table = np.fromfile(table_path, dtype="<f4").reshape(256, l2dsci.TABLE_WIDTH)
    named = texts()
    plain = [text for _, text in named]

    expected = {
        "generator": "tools/science/codec_fixture.py",
        "source": "l2dsci.byte_ids, l2dsci.table_input, batch padding as in l2dsci.onnx_vectors",
        "table_sha256": l2dsci.file_sha256(table_path),
        "input_bytes": l2dsci.INPUT_BYTES,
        "channels": l2dsci.TABLE_WIDTH,
        "batch": l2dsci.BATCH,
        "pad_byte": l2dsci.PAD_BYTE,
        "texts": [{"name": name, "text": text} for name, text in named],
        "modes": {},
    }
    for mode, routing in MODES.items():
        ids, truncated = l2dsci.byte_ids(plain, routing)
        rows = l2dsci.table_input(table, ids)                    # [n, 64, 1, 256]
        batches = []
        for start in range(0, len(ids), l2dsci.BATCH):
            chunk = ids[start:start + l2dsci.BATCH]
            n = len(chunk)
            if n < l2dsci.BATCH:                                 # as in l2dsci.onnx_vectors
                chunk = np.concatenate(
                    [chunk, np.full((l2dsci.BATCH - n, l2dsci.INPUT_BYTES), l2dsci.PAD_BYTE, np.uint8)])
            batches.append({"rows": n, "sha256": sha256(l2dsci.table_input(table, chunk))})
        expected["modes"][mode] = {
            "l2dsci_routing": routing,
            "rows": [
                {"ids": ids[row].tobytes().hex(), "truncated": bool(truncated[row]),
                 "input_sha256": sha256(np.ascontiguousarray(rows[row]))}
                for row in range(len(plain))
            ],
            "batches": batches,
        }
    for name, content in (("byte-codec-expected.json", expected),):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(content, handle, ensure_ascii=True, indent=1)
            handle.write("\n")
        print(f"wrote {os.path.join(args.out, name)}")
    print(f"{len(plain)} texts, modes {list(MODES)}, table {expected['table_sha256']}")


if __name__ == "__main__":
    main()
