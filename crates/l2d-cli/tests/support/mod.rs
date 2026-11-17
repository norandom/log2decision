use std::process::Output;

/// Concurrent command tests may wait for another command's admission. Preserve every
/// other byte, so their existing strict error/output assertions still catch regressions.
pub fn diagnostics(output: &Output) -> String {
    String::from_utf8(output.stderr.clone())
        .expect("stderr is UTF-8")
        .split_inclusive('\n')
        .filter(|line| !admission_wait(line.trim_end_matches(['\r', '\n'])))
        .collect()
}

fn admission_wait(line: &str) -> bool {
    let Some(line) = line.strip_prefix("waiting: ") else {
        return false;
    };
    let Some((holder, line)) = line.split_once(" (process ") else {
        return false;
    };
    let Some((pid, line)) = line.split_once(") holds ") else {
        return false;
    };
    let Some((gate, requested)) = line.split_once("; requested ") else {
        return false;
    };
    !holder.is_empty()
        && pid.parse::<u32>().is_ok_and(|pid| pid > 0)
        && matches!(gate, "host" | "npu")
        && !requested.is_empty()
}
