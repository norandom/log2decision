//! The peak memory of the process as the operating system observed it, beside the
//! peak the budget counted as admitted (performance.md: "Record counted allocations
//! and observed process peaks separately").

/// The peak working set of this process; `None` when the system does not give it.
#[cfg(windows)]
pub(crate) fn peak_bytes() -> Option<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: the counters are plain numbers for which all zeros is a valid value.
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = size;
    // SAFETY: the handle is the pseudo handle of this process, which needs no
    // closing, and `counters` is a writable structure of the size that is passed.
    let read = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) };
    (read != 0).then_some(counters.PeakWorkingSetSize as u64)
}

/// The peak resident set of this process; `None` when the system does not give it.
#[cfg(not(windows))]
pub(crate) fn peak_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?;
    let kib: u64 = line.trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kib * 1024)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_peak_of_this_process_is_above_what_it_just_touched() {
        let before = super::peak_bytes();
        // Windows gives the peak working set; another system may give none.
        assert!(before.is_some() || !cfg!(windows));
        let Some(before) = before else {
            return;
        };
        // Sixty-four megabytes, every page written.
        let touched = vec![1u8; 64 << 20];
        assert_eq!(
            touched
                .iter()
                .step_by(4096)
                .map(|byte| u64::from(*byte))
                .sum::<u64>(),
            16_384
        );
        let after = super::peak_bytes().unwrap();
        assert!(after >= before);
        assert!(after >= 64 << 20, "{after}");
    }
}
