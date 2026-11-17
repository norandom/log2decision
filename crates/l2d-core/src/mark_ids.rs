//! Sorted IDs in memory or in fixed-width files. External sorting uses binary merges:
//! at most 64 run names, two input records and one bounded output buffer coexist.
use crate::L2dError;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub(crate) const WORK_BYTES: u64 = 256 * 1024;
pub(crate) const ID_BYTES: u64 = 136;
const WIDTH: usize = 64;

fn failure(error: impl std::fmt::Display) -> L2dError {
    L2dError::Refused {
        reason: format!("mark-set IDs on disk could not be read or written: {error}"),
    }
}

#[derive(Debug)]
struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug)]
pub(crate) struct Disk {
    file: File,
    _directory: Arc<Directory>,
    count: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum Ids {
    Memory(Vec<String>),
    Disk(Arc<Disk>),
}

impl Default for Ids {
    fn default() -> Self {
        Self::Memory(Vec::new())
    }
}

impl Ids {
    pub(crate) fn equals(&self, other: &Self) -> Result<bool, L2dError> {
        if self.len() != other.len() {
            return Ok(false);
        }
        for index in 0..self.len() {
            if self.get(index)? != other.get(index)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Memory(ids) => ids.len(),
            Self::Disk(disk) => disk.count,
        }
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(crate) fn spilled(&self) -> bool {
        matches!(self, Self::Disk(_))
    }
    pub(crate) fn held(&self) -> u64 {
        if self.spilled() {
            WORK_BYTES
        } else {
            self.len() as u64 * ID_BYTES
        }
    }
    pub(crate) fn get(&self, index: usize) -> Result<String, L2dError> {
        match self {
            Self::Memory(ids) => ids
                .get(index)
                .cloned()
                .ok_or_else(|| failure("ID index out of range")),
            Self::Disk(disk) => {
                if index >= disk.count {
                    return Err(failure("ID index out of range"));
                }
                let mut bytes = [0; WIDTH];
                let mut done = 0;
                while done < WIDTH {
                    #[cfg(windows)]
                    let count = std::os::windows::fs::FileExt::seek_read(
                        &disk.file,
                        &mut bytes[done..],
                        (index * WIDTH + done) as u64,
                    );
                    #[cfg(unix)]
                    let count = std::os::unix::fs::FileExt::read_at(
                        &disk.file,
                        &mut bytes[done..],
                        (index * WIDTH + done) as u64,
                    );
                    let count = count.map_err(failure)?;
                    if count == 0 {
                        return Err(failure("ID file ended early"));
                    }
                    done += count;
                }
                String::from_utf8(bytes.to_vec()).map_err(failure)
            }
        }
    }
    pub(crate) fn contains(&self, id: &str) -> Result<bool, L2dError> {
        let (mut low, mut high) = (0, self.len());
        while low < high {
            let middle = low + (high - low) / 2;
            match self.get(middle)?.as_str().cmp(id) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => return Ok(true),
            }
        }
        Ok(false)
    }
    pub(crate) fn each(
        &self,
        mut each: impl FnMut(&str) -> Result<(), L2dError>,
    ) -> Result<(), L2dError> {
        for index in 0..self.len() {
            each(&self.get(index)?)?;
        }
        Ok(())
    }
}

impl Serialize for Ids {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for index in 0..self.len() {
            let id = self.get(index).map_err(serde::ser::Error::custom)?;
            sequence.serialize_element(&id)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for Ids {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Ids;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("strictly ascending log IDs")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Ids, A::Error> {
                let mut builder = Builder::new(64 * 1024);
                let mut previous: Option<String> = None;
                while let Some(id) = sequence.next_element::<String>()? {
                    if previous.as_ref().is_some_and(|previous| previous >= &id) {
                        return Err(serde::de::Error::custom("IDs are not strictly ascending"));
                    }
                    builder.push(&id).map_err(serde::de::Error::custom)?;
                    previous = Some(id);
                }
                builder.finish().map_err(serde::de::Error::custom)
            }
        }
        deserializer.deserialize_seq(Visitor)
    }
}

pub(crate) struct Builder {
    cap: usize,
    ids: Vec<String>,
    directory: Option<Arc<Directory>>,
    runs: Vec<Option<PathBuf>>,
    sequence: u64,
}

impl Builder {
    pub(crate) fn new(bytes: u64) -> Self {
        Self {
            cap: (bytes / ID_BYTES).max(1) as usize,
            ids: Vec::new(),
            directory: None,
            runs: Vec::new(),
            sequence: 0,
        }
    }
    pub(crate) fn push(&mut self, id: &str) -> Result<(), L2dError> {
        if id.len() != WIDTH
            || !id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(failure(
                "expected a 64-character lower-case hexadecimal log ID",
            ));
        }
        if self.ids.len() == self.cap {
            self.flush()?;
        }
        self.ids.push(id.into());
        Ok(())
    }
    fn path(&mut self) -> Result<PathBuf, L2dError> {
        if self.directory.is_none() {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(failure)?
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "l2d-mark-ids-{}-{stamp}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).map_err(failure)?;
            self.directory = Some(Arc::new(Directory(path)));
        }
        self.sequence += 1;
        Ok(self
            .directory
            .as_ref()
            .unwrap()
            .0
            .join(format!("{}.ids", self.sequence)))
    }
    fn flush(&mut self) -> Result<(), L2dError> {
        if self.ids.is_empty() {
            return Ok(());
        }
        self.ids.sort_unstable();
        self.ids.dedup();
        let mut path = self.path()?;
        let mut writer = BufWriter::with_capacity(8192, File::create(&path).map_err(failure)?);
        for id in &self.ids {
            writer.write_all(id.as_bytes()).map_err(failure)?;
        }
        writer.flush().map_err(failure)?;
        drop(writer);
        self.ids.clear();
        let mut level = 0;
        loop {
            if level == self.runs.len() {
                self.runs.push(None);
            }
            match self.runs[level].take() {
                None => {
                    self.runs[level] = Some(path);
                    break;
                }
                Some(previous) => {
                    let output = self.path()?;
                    merge(&previous, &path, &output)?;
                    fs::remove_file(previous).map_err(failure)?;
                    fs::remove_file(path).map_err(failure)?;
                    path = output;
                    level += 1;
                }
            }
        }
        Ok(())
    }
    pub(crate) fn finish(mut self) -> Result<Ids, L2dError> {
        if self.directory.is_none() {
            self.ids.sort_unstable();
            self.ids.dedup();
            self.ids.shrink_to_fit();
            return Ok(Ids::Memory(self.ids));
        }
        self.flush()?;
        let mut result: Option<PathBuf> = None;
        for index in 0..self.runs.len() {
            if let Some(run) = self.runs[index].take() {
                result = Some(match result {
                    None => run,
                    Some(previous) => {
                        let output = self.path()?;
                        merge(&previous, &run, &output)?;
                        fs::remove_file(previous).map_err(failure)?;
                        fs::remove_file(run).map_err(failure)?;
                        output
                    }
                });
            }
        }
        let file = File::open(result.unwrap()).map_err(failure)?;
        let count = (file.metadata().map_err(failure)?.len() / WIDTH as u64) as usize;
        Ok(Ids::Disk(Arc::new(Disk {
            file,
            count,
            _directory: self.directory.unwrap(),
        })))
    }
}

/// Bounds all raw JSON bytes except the two canonical root ID arrays BEFORE serde
/// sees them. Thus a huge query/name/source/redaction or positions list cannot evade
/// the old LOAD_FACTOR bound just because IDs are now streamed. ID strings are exactly
/// 64 ASCII hex bytes; escaping/nesting in those arrays is refused, never buffered.
pub(crate) struct JsonReader<R> {
    reader: R,
    remaining: u64,
    depth: usize,
    string: bool,
    escaped: bool,
    root_word: Vec<u8>,
    pending_ids: bool,
    ids: bool,
    id_length: usize,
}

impl<R: Read> JsonReader<R> {
    pub(crate) fn new(reader: R, metadata_bytes: u64) -> Self {
        Self {
            reader,
            remaining: metadata_bytes,
            depth: 0,
            string: false,
            escaped: false,
            root_word: Vec::with_capacity(20),
            pending_ids: false,
            ids: false,
            id_length: 0,
        }
    }
    fn byte(&mut self, byte: u8) -> std::io::Result<()> {
        let invalid = |message| std::io::Error::other(message);
        if !self.ids {
            self.remaining = self
                .remaining
                .checked_sub(1)
                .ok_or_else(|| invalid("mark-set metadata exceeds the memory allowance"))?;
        }
        if self.string {
            if self.ids {
                if byte == b'"' {
                    if self.id_length != 64 {
                        return Err(invalid("ID is not 64 hex bytes"));
                    }
                    self.string = false;
                } else {
                    self.id_length += 1;
                    if self.id_length > 64
                        || !(byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                    {
                        return Err(invalid("ID is not canonical hexadecimal"));
                    }
                }
                return Ok(());
            }
            if self.escaped {
                self.escaped = false;
                self.root_word.clear();
            } else if byte == b'\\' {
                self.escaped = true;
                self.root_word.clear();
            } else if byte == b'"' {
                self.string = false;
            } else if self.depth == 1 && self.root_word.len() < 20 {
                self.root_word.push(byte);
            }
            return Ok(());
        }
        if self.ids && !byte.is_ascii_whitespace() && !matches!(byte, b'"' | b',' | b']') {
            return Err(invalid("ID array contains a non-string value"));
        }
        match byte {
            b'"' => {
                self.string = true;
                self.id_length = 0;
                self.root_word.clear();
            }
            b':' if self.depth == 1 => {
                self.pending_ids = self.root_word == b"ids" || self.root_word == b"deselected_ids";
            }
            b'[' => {
                self.ids = self.depth == 1 && self.pending_ids;
                self.pending_ids = false;
                self.depth += 1;
            }
            b'{' => {
                self.depth += 1;
                self.pending_ids = false;
            }
            b']' | b'}' => {
                self.depth = self.depth.saturating_sub(1);
                self.ids = false;
                self.pending_ids = false;
            }
            b',' => {
                self.pending_ids = false;
                self.root_word.clear();
            }
            _ => {}
        }
        Ok(())
    }
}

impl<R: Read> Read for JsonReader<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let max = out.len().min(8192);
        let count = self.reader.read(&mut out[..max])?;
        for &byte in &out[..count] {
            self.byte(byte)?;
        }
        Ok(count)
    }
}

fn next(reader: &mut impl Read) -> Result<Option<[u8; WIDTH]>, L2dError> {
    let mut bytes = [0; WIDTH];
    if reader.read(&mut bytes[..1]).map_err(failure)? == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut bytes[1..]).map_err(failure)?;
    Ok(Some(bytes))
}

fn merge(a: &Path, b: &Path, out: &Path) -> Result<(), L2dError> {
    let mut a = BufReader::with_capacity(8192, File::open(a).map_err(failure)?);
    let mut b = BufReader::with_capacity(8192, File::open(b).map_err(failure)?);
    let mut out = BufWriter::with_capacity(8192, File::create(out).map_err(failure)?);
    let (mut left, mut right) = (next(&mut a)?, next(&mut b)?);
    let mut last = None;
    while left.is_some() || right.is_some() {
        let value = if right.is_none() || left.is_some_and(|left| left <= right.unwrap()) {
            let value = left.take().unwrap();
            left = next(&mut a)?;
            value
        } else {
            let value = right.take().unwrap();
            right = next(&mut b)?;
            value
        };
        if last != Some(value) {
            out.write_all(&value).map_err(failure)?;
            last = Some(value);
        }
    }
    out.flush().map_err(failure)
}
