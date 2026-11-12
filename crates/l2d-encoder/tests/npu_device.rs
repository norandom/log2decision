//! The NPU device query of task 2.3: what the Windows configuration manager returns is
//! parsed and judged by pure functions, tested here without the system call.

use l2d_encoder::npu::{
    ComputeAccelerator, NpuMatch, NpuStatus, instance_ids_from_list, pci_ids, string_from_property,
};

fn utf16(text: &str) -> Vec<u16> {
    text.encode_utf16().collect()
}

fn device(instance_id: &str, description: &str, driver: &str) -> ComputeAccelerator {
    ComputeAccelerator {
        instance_id: instance_id.into(),
        description: description.into(),
        driver_version: driver.into(),
    }
}

const STRIX: &str = "PCI\\VEN_1022&DEV_17F0&SUBSYS_17F01022&REV_10\\4&2F1A8B3&0&0142";

#[test]
fn the_device_id_list_is_split_at_its_terminators() {
    // The configuration manager returns the ids one after another, each ended by a NUL,
    // and the list by one more.
    let list = utf16("PCI\\VEN_1022&DEV_17F0\\1\0ROOT\\OTHER\\0000\0\0");
    assert_eq!(
        instance_ids_from_list(&list),
        vec!["PCI\\VEN_1022&DEV_17F0\\1", "ROOT\\OTHER\\0000"]
    );
    assert!(instance_ids_from_list(&utf16("\0")).is_empty());
    assert!(instance_ids_from_list(&[]).is_empty());
}

#[test]
fn a_string_property_ends_at_its_terminator_or_at_the_reported_length() {
    let mut buffer = utf16("32.0.20102.4480\0leftover");
    // The size comes back in bytes and includes the terminator.
    assert_eq!(string_from_property(&buffer, 32), "32.0.20102.4480");
    // A length without terminator is honoured.
    assert_eq!(string_from_property(&buffer, 8), "32.0");
    // A length beyond the buffer does not read past it.
    buffer.truncate(4);
    assert_eq!(string_from_property(&buffer, 400), "32.0");
    assert_eq!(string_from_property(&[], 0), "");
}

#[test]
fn vendor_and_device_id_are_read_from_a_pci_instance_id() {
    assert_eq!(pci_ids(STRIX), Some(("1022".into(), "17F0".into())));
    // Case does not matter; the ids come back in upper case.
    assert_eq!(
        pci_ids("pci\\ven_1022&dev_17f0&subsys_17f01022"),
        Some(("1022".into(), "17F0".into()))
    );
    assert_eq!(pci_ids("ROOT\\COMPUTEACCELERATOR\\0000"), None);
    assert_eq!(pci_ids("PCI\\VEN_1022"), None);
    assert_eq!(pci_ids(""), None);
}

#[test]
fn a_host_without_such_a_device_reports_absence() {
    let status = NpuStatus::from_devices(Vec::new());
    assert!(!status.is_present());
    assert_eq!(status.driver_version(), None);
    let NpuStatus::Absent { reason, seen } = &status else {
        panic!("{status:?}");
    };
    assert!(seen.is_empty());
    assert!(reason.contains("ComputeAccelerator"), "{reason}");
}

#[test]
fn a_device_of_another_vendor_is_reported_and_passed_over() {
    let other = device(
        "PCI\\VEN_8086&DEV_7D1D&SUBSYS_0\\3&1",
        "Other NPU",
        "1.2.3.4",
    );
    let status = NpuStatus::from_devices(vec![other.clone()]);
    let NpuStatus::Absent { reason, seen } = &status else {
        panic!("{status:?}");
    };
    assert_eq!(seen, &vec![other]);
    assert!(reason.contains("AMD"), "{reason}");
    let lines = status.log_lines().join("\n");
    assert!(lines.contains("Other NPU"), "{lines}");
    assert!(lines.contains("VEN_8086"), "{lines}");
}

#[test]
fn the_known_npu_is_taken_and_its_driver_version_reported() {
    let npu = device(STRIX, "NPU Compute Accelerator Device", "32.0.20102.4480");
    let status = NpuStatus::from_devices(vec![npu.clone()]);
    assert!(status.is_present());
    assert_eq!(status.driver_version(), Some("32.0.20102.4480"));
    let NpuStatus::Present {
        taken,
        matched,
        seen,
    } = &status
    else {
        panic!("{status:?}");
    };
    assert_eq!(taken, &npu);
    assert_eq!(*matched, NpuMatch::KnownDevice);
    assert_eq!(seen.len(), 1);
}

#[test]
fn a_known_device_id_wins_over_an_earlier_amd_device_and_every_match_is_logged() {
    let unknown = device(
        "PCI\\VEN_1022&DEV_ABCD&SUBSYS_0\\1",
        "Some AMD accelerator",
        "1.0.0.0",
    );
    let other = device("PCI\\VEN_8086&DEV_7D1D\\2", "Other NPU", "9.9.9.9");
    let npu = device(STRIX, "NPU Compute Accelerator Device", "32.0.20102.4480");
    let status = NpuStatus::from_devices(vec![unknown.clone(), other, npu.clone()]);
    let NpuStatus::Present { taken, matched, .. } = &status else {
        panic!("{status:?}");
    };
    assert_eq!(taken, &npu);
    assert_eq!(*matched, NpuMatch::KnownDevice);

    let lines = status.log_lines();
    let text = lines.join("\n");
    // All three devices, which one was taken and why, and the AMD one that was not.
    assert!(text.contains("Some AMD accelerator"), "{text}");
    assert!(text.contains("Other NPU"), "{text}");
    assert_eq!(
        lines.iter().filter(|line| line.contains("taken")).count(),
        1,
        "{text}"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.contains("taken") && line.contains("DEV_17F0")),
        "{text}"
    );
    assert!(text.contains("32.0.20102.4480"), "{text}");
}

#[test]
fn an_amd_device_with_an_unknown_id_is_taken_and_marked_as_such() {
    let unknown = device(
        "PCI\\VEN_1022&DEV_ABCD&SUBSYS_0\\1",
        "Some AMD accelerator",
        "1.0.0.0",
    );
    let second = device(
        "PCI\\VEN_1022&DEV_EF01&SUBSYS_0\\1",
        "Another AMD accelerator",
        "2.0.0.0",
    );
    let status = NpuStatus::from_devices(vec![unknown.clone(), second]);
    let NpuStatus::Present { taken, matched, .. } = &status else {
        panic!("{status:?}");
    };
    // The first of the vendor's devices, in the configuration manager's order.
    assert_eq!(taken, &unknown);
    assert_eq!(*matched, NpuMatch::AmdVendorOnly);
    assert_eq!(status.driver_version(), Some("1.0.0.0"));
    let text = status.log_lines().join("\n");
    assert!(text.contains("not a known NPU device id"), "{text}");
    assert!(text.contains("Another AMD accelerator"), "{text}");
}

/// A run record built from the accessors must be able to tell the two matches apart, and
/// the summary line of the log must not call an unknown AMD device's driver the NPU's.
#[test]
fn how_the_device_was_matched_is_in_the_accessors_and_in_the_summary_line() {
    let known = NpuStatus::from_devices(vec![device(STRIX, "NPU", "32.0.20102.4480")]);
    assert_eq!(known.matched(), Some(NpuMatch::KnownDevice));
    assert_eq!(
        known.taken().map(|taken| taken.instance_id.as_str()),
        Some(STRIX)
    );
    assert_eq!(
        known.log_lines().last().map(String::as_str),
        Some("NPU driver version: 32.0.20102.4480")
    );

    let vendor_only = NpuStatus::from_devices(vec![device(
        "PCI\\VEN_1022&DEV_ABCD&SUBSYS_0\\1",
        "Some AMD accelerator",
        "1.0.0.0",
    )]);
    assert_eq!(vendor_only.matched(), Some(NpuMatch::AmdVendorOnly));
    let lines = vendor_only.log_lines();
    let summary = lines.last().unwrap();
    assert!(summary.ends_with(": 1.0.0.0"), "{summary}");
    assert!(summary.contains("not a known NPU device id"), "{summary}");
    assert!(!summary.starts_with("NPU driver version"), "{summary}");

    let absent = NpuStatus::from_devices(Vec::new());
    assert_eq!(absent.matched(), None);
    assert_eq!(absent.taken(), None);
}

/// One device's missing property does not end the query: the device is listed with
/// "unknown" and can still be taken.
#[test]
fn a_device_without_a_readable_property_is_listed_as_unknown() {
    let no_driver = ComputeAccelerator::from_properties(STRIX.into(), Some("NPU".into()), None);
    assert_eq!(no_driver.description, "NPU");
    assert_eq!(no_driver.driver_version, "unknown");
    let nothing = ComputeAccelerator::from_properties("ROOT\\X\\0".into(), None, Some("  ".into()));
    assert_eq!(nothing.description, "unknown");
    assert_eq!(nothing.driver_version, "unknown");
    let whole = ComputeAccelerator::from_properties(
        STRIX.into(),
        Some("NPU".into()),
        Some("32.0.20102.4480".into()),
    );
    assert_eq!(whole, device(STRIX, "NPU", "32.0.20102.4480"));

    let status = NpuStatus::from_devices(vec![nothing, no_driver]);
    assert!(status.is_present());
    assert_eq!(status.matched(), Some(NpuMatch::KnownDevice));
    // Present, but no driver version to put into a run record.
    assert_eq!(status.driver_version(), None);
    let text = status.log_lines().join("\n");
    assert!(
        text.contains("unknown (ROOT\\X\\0), driver unknown"),
        "{text}"
    );
    assert!(text.contains("NPU driver version: unknown"), "{text}");
}

/// A description longer than the buffer comes back cut at the buffer, not as a failure.
#[test]
fn a_property_longer_than_the_buffer_is_cut_at_the_buffer() {
    let long: Vec<u16> = "d".repeat(600).encode_utf16().collect();
    let buffer = &long[..511];
    // The call reports the size of the whole value; only the buffer is read.
    assert_eq!(string_from_property(buffer, 1202), "d".repeat(511));
}

/// The system call itself: on any host it answers, with a device or with absence.
#[test]
fn the_query_answers_on_this_host() {
    let status = NpuStatus::query().unwrap_or_else(|error| panic!("{error}"));
    for line in status.log_lines() {
        println!("npu_device: {line}");
    }
    if let Some(version) = status.driver_version() {
        assert!(!version.is_empty());
    }
}
