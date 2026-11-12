//! Presence of the AMD NPU and its driver version, read from the Windows configuration
//! manager. No process is started and nothing of the runtime is loaded.
//!
//! The system call ([`NpuStatus::query`]) only collects the present devices of the setup
//! class `ComputeAccelerator`; which of them is the NPU is decided by
//! [`NpuStatus::from_devices`], which needs no device.
//!
//! # Which device is taken
//!
//! A PCI device of AMD whose device id is in [`KNOWN_NPU_DEVICE_IDS`] is taken first.
//! Without one, the first other PCI device of AMD in the class is taken and marked
//! [`NpuMatch::AmdVendorOnly`]: the list holds the one id seen so far (the reference
//! host), other processor generations have other ids, and whether such a device runs the
//! encoder is decided by the provider, which is registered so that it fails with an
//! error. Requiring a listed id would turn every host the list does not know into "no
//! NPU". [`NpuStatus::log_lines`] names every device seen, the one taken and why, so a
//! wrong pick is visible in the output.

use l2d_core::L2dError;

/// Device setup class `ComputeAccelerator`, the class the AMD NPU driver installs under.
pub const COMPUTE_ACCELERATOR_CLASS: &str = "{F01A9D53-3FF6-48D2-9F97-C8A7004BE10C}";

/// PCI vendor id of AMD.
pub const AMD_PCI_VENDOR: &str = "1022";

/// PCI device ids known to be an AMD NPU. `17F0` is the NPU of the Ryzen AI 9 HX 370, the
/// reference host (task 2.2).
pub const KNOWN_NPU_DEVICE_IDS: &[&str] = &["17F0"];

/// What stands for a device property the configuration manager did not give.
pub const UNKNOWN: &str = "unknown";

/// A present device of the class `ComputeAccelerator`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeAccelerator {
    /// Device instance id, for example `PCI\VEN_1022&DEV_17F0&SUBSYS_17F01022&REV_10\...`.
    pub instance_id: String,
    /// The device description, or [`UNKNOWN`].
    pub description: String,
    /// The driver version, or [`UNKNOWN`] (a device without a driver has none).
    pub driver_version: String,
}

impl ComputeAccelerator {
    /// A device from what the configuration manager gave for it. A property that could
    /// not be read, or that is empty, becomes [`UNKNOWN`]: one device's missing property
    /// does not end the query, and the device is still listed.
    pub fn from_properties(
        instance_id: String,
        description: Option<String>,
        driver_version: Option<String>,
    ) -> Self {
        let known = |value: Option<String>| {
            value
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| UNKNOWN.to_string())
        };
        ComputeAccelerator {
            instance_id,
            description: known(description),
            driver_version: known(driver_version),
        }
    }
}

/// Why a device was taken as the NPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuMatch {
    /// An AMD device whose id is in [`KNOWN_NPU_DEVICE_IDS`].
    KnownDevice,
    /// An AMD device of the class with an id that is not in the list.
    AmdVendorOnly,
}

/// What the host has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpuStatus {
    Present {
        /// The device taken as the NPU.
        taken: ComputeAccelerator,
        matched: NpuMatch,
        /// Every present device of the class, in the configuration manager's order.
        seen: Vec<ComputeAccelerator>,
    },
    Absent {
        reason: String,
        /// Devices of the class that are not AMD's.
        seen: Vec<ComputeAccelerator>,
    },
}

/// Vendor and device id of a PCI device instance id, in upper case. None for a device
/// that is not on the PCI bus or whose id does not have both.
pub fn pci_ids(instance_id: &str) -> Option<(String, String)> {
    let upper = instance_id.to_ascii_uppercase();
    let hardware = upper.strip_prefix("PCI\\")?.split('\\').next()?;
    let field = |prefix: &str| {
        hardware
            .split('&')
            .find_map(|part| part.strip_prefix(prefix))
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    Some((field("VEN_")?, field("DEV_")?))
}

/// The device instance ids of a list as the configuration manager returns it: UTF-16, one
/// id after another, each ended by a NUL and the list by one more.
pub fn instance_ids_from_list(list: &[u16]) -> Vec<String> {
    list.split(|unit| *unit == 0)
        .filter(|id| !id.is_empty())
        .map(String::from_utf16_lossy)
        .collect()
}

/// A string device property: `buffer` as the call filled it, `bytes` the size it
/// reported. The string ends at its NUL, at the reported size or at the end of the
/// buffer, whichever comes first.
pub fn string_from_property(buffer: &[u16], bytes: u32) -> String {
    let units = &buffer[..(bytes as usize / 2).min(buffer.len())];
    let end = units
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(units.len());
    String::from_utf16_lossy(&units[..end])
}

fn is_amd(device: &ComputeAccelerator) -> bool {
    pci_ids(&device.instance_id).is_some_and(|(vendor, _)| vendor == AMD_PCI_VENDOR)
}

fn is_known(device: &ComputeAccelerator) -> bool {
    pci_ids(&device.instance_id).is_some_and(|(vendor, id)| {
        vendor == AMD_PCI_VENDOR && KNOWN_NPU_DEVICE_IDS.contains(&id.as_str())
    })
}

impl NpuStatus {
    /// Asks the configuration manager. A host without the device is not an error: it is
    /// [`NpuStatus::Absent`]. A device whose description or driver version cannot be read
    /// is listed with [`UNKNOWN`]. An error is a failed listing of the class's devices.
    pub fn query() -> Result<Self, L2dError> {
        #[cfg(windows)]
        {
            windows::compute_accelerators()
                .map(Self::from_devices)
                .map_err(|detail| L2dError::Runtime {
                    detail: format!("the Windows device query failed: {detail}"),
                })
        }
        #[cfg(not(windows))]
        {
            Ok(NpuStatus::Absent {
                reason: "the NPU is looked for through the Windows configuration manager; this \
                         is not a Windows host"
                    .into(),
                seen: Vec::new(),
            })
        }
    }

    /// Decides which of the present devices of the class `ComputeAccelerator` is the NPU.
    pub fn from_devices(seen: Vec<ComputeAccelerator>) -> Self {
        let known = seen.iter().find(|device| is_known(device));
        let taken = known.or_else(|| seen.iter().find(|device| is_amd(device)));
        match taken.cloned() {
            Some(taken) => NpuStatus::Present {
                matched: if known.is_some() {
                    NpuMatch::KnownDevice
                } else {
                    NpuMatch::AmdVendorOnly
                },
                taken,
                seen,
            },
            None => NpuStatus::Absent {
                reason: if seen.is_empty() {
                    "no present device of the class ComputeAccelerator: this host has no NPU, \
                     or its driver is not installed"
                        .into()
                } else {
                    format!(
                        "none of the {} present device(s) of the class ComputeAccelerator is an \
                         AMD device (PCI vendor {AMD_PCI_VENDOR})",
                        seen.len()
                    )
                },
                seen,
            },
        }
    }

    pub fn is_present(&self) -> bool {
        matches!(self, NpuStatus::Present { .. })
    }

    /// The driver version of the device taken as the NPU; none when there is no such
    /// device or its driver version is not known. Read it together with
    /// [`NpuStatus::matched`]: with [`NpuMatch::AmdVendorOnly`] this is the driver of an
    /// AMD device that is not known to be an NPU, and a run record must say so.
    pub fn driver_version(&self) -> Option<&str> {
        match self {
            NpuStatus::Present { taken, .. } if taken.driver_version != UNKNOWN => {
                Some(&taken.driver_version)
            }
            _ => None,
        }
    }

    /// How the device taken as the NPU was matched; none when there is no such device.
    pub fn matched(&self) -> Option<NpuMatch> {
        match self {
            NpuStatus::Present { matched, .. } => Some(*matched),
            NpuStatus::Absent { .. } => None,
        }
    }

    /// The device taken as the NPU.
    pub fn taken(&self) -> Option<&ComputeAccelerator> {
        match self {
            NpuStatus::Present { taken, .. } => Some(taken),
            NpuStatus::Absent { .. } => None,
        }
    }

    /// The status in lines for a log: every device seen, which was taken and why, or why
    /// none was.
    pub fn log_lines(&self) -> Vec<String> {
        let (seen, taken) = match self {
            NpuStatus::Present { seen, taken, .. } => (seen, Some(taken)),
            NpuStatus::Absent { seen, .. } => (seen, None),
        };
        let mut lines = vec![format!(
            "present devices of the class ComputeAccelerator: {}",
            seen.len()
        )];
        // The device taken is marked once, also when two entries are equal.
        let taken_at = taken.and_then(|taken| seen.iter().position(|device| device == taken));
        for (index, device) in seen.iter().enumerate() {
            let note = if Some(index) == taken_at {
                match self {
                    NpuStatus::Present {
                        matched: NpuMatch::KnownDevice,
                        ..
                    } => "taken as the NPU: a known NPU device id".to_string(),
                    _ => format!(
                        "taken as the NPU: the first AMD device of the class; its id is not a \
                         known NPU device id ({})",
                        KNOWN_NPU_DEVICE_IDS.join(", ")
                    ),
                }
            } else if is_known(device) {
                "a known NPU device id, passed over: an earlier one is used".to_string()
            } else if is_amd(device) {
                "an AMD device, passed over".to_string()
            } else {
                "not an AMD device, passed over".to_string()
            };
            lines.push(format!(
                "{}. {} ({}), driver {}: {note}",
                index + 1,
                device.description,
                device.instance_id,
                device.driver_version
            ));
        }
        match self {
            NpuStatus::Present {
                taken,
                matched: NpuMatch::KnownDevice,
                ..
            } => {
                lines.push(format!("NPU driver version: {}", taken.driver_version));
            }
            NpuStatus::Present { taken, .. } => {
                lines.push(format!(
                    "driver version of the AMD device taken as the NPU (its id is not a known \
                     NPU device id): {}",
                    taken.driver_version
                ));
            }
            NpuStatus::Absent { reason, .. } => lines.push(format!("no NPU: {reason}")),
        }
        lines
    }
}

#[cfg(windows)]
mod windows {
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        CM_GETIDLIST_FILTER_CLASS, CM_GETIDLIST_FILTER_PRESENT, CM_Get_DevNode_PropertyW,
        CM_Get_Device_ID_List_SizeW, CM_Get_Device_ID_ListW, CM_LOCATE_DEVNODE_NORMAL,
        CM_Locate_DevNodeW, CR_BUFFER_SMALL, CR_SUCCESS,
    };
    use windows_sys::Win32::Devices::Properties::{
        DEVPKEY_Device_DeviceDesc, DEVPKEY_Device_DriverVersion, DEVPROP_TYPE_STRING,
    };
    use windows_sys::Win32::Foundation::DEVPROPKEY;

    use super::{
        COMPUTE_ACCELERATOR_CLASS, ComputeAccelerator, instance_ids_from_list, string_from_property,
    };

    /// Every present device of the class `ComputeAccelerator`, in the configuration
    /// manager's order.
    pub(super) fn compute_accelerators() -> Result<Vec<ComputeAccelerator>, String> {
        let class: Vec<u16> = COMPUTE_ACCELERATOR_CLASS
            .encode_utf16()
            .chain(Some(0))
            .collect();
        let flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
        let mut length = 0u32;
        // SAFETY: `class` is a NUL-terminated UTF-16 string and `length` a valid output.
        let status = unsafe { CM_Get_Device_ID_List_SizeW(&mut length, class.as_ptr(), flags) };
        if status != CR_SUCCESS {
            return Err(format!("CM_Get_Device_ID_List_SizeW failed with {status}"));
        }
        let mut list = vec![0u16; length as usize];
        // SAFETY: `list` has the `length` elements the size call asked for.
        let status =
            unsafe { CM_Get_Device_ID_ListW(class.as_ptr(), list.as_mut_ptr(), length, flags) };
        if status != CR_SUCCESS {
            return Err(format!("CM_Get_Device_ID_ListW failed with {status}"));
        }
        let mut devices = Vec::new();
        for instance_id in instance_ids_from_list(&list) {
            let id: Vec<u16> = instance_id.encode_utf16().chain(Some(0)).collect();
            let mut node = 0u32;
            // SAFETY: `id` is a NUL-terminated device instance id.
            let status =
                unsafe { CM_Locate_DevNodeW(&mut node, id.as_ptr(), CM_LOCATE_DEVNODE_NORMAL) };
            // A device that cannot be located, or a property that cannot be read (a device
            // without a driver has no driver version), does not end the query: the device
            // is listed with what is known of it.
            let (description, driver_version) = if status == CR_SUCCESS {
                (
                    device_string(node, &DEVPKEY_Device_DeviceDesc),
                    device_string(node, &DEVPKEY_Device_DriverVersion),
                )
            } else {
                (None, None)
            };
            devices.push(ComputeAccelerator::from_properties(
                instance_id,
                description,
                driver_version,
            ));
        }
        Ok(devices)
    }

    /// A string property of a device node; none when the node has no such property or it
    /// is not a string.
    fn device_string(node: u32, key: &DEVPROPKEY) -> Option<String> {
        let mut units = 512usize;
        // A second pass with the size the first one asked for, when the value is longer.
        for _ in 0..2 {
            let mut kind = 0u32;
            let mut buffer = vec![0u16; units];
            let mut bytes = (buffer.len() * 2) as u32;
            // SAFETY: `buffer` is `bytes` bytes long and aligned for UTF-16; the other
            // pointers are valid for the call.
            let status = unsafe {
                CM_Get_DevNode_PropertyW(
                    node,
                    key,
                    &mut kind,
                    buffer.as_mut_ptr().cast(),
                    &mut bytes,
                    0,
                )
            };
            if status == CR_BUFFER_SMALL && bytes as usize > buffer.len() * 2 {
                units = (bytes as usize).div_ceil(2);
                continue;
            }
            if status != CR_SUCCESS || kind != DEVPROP_TYPE_STRING {
                return None;
            }
            return Some(string_from_property(&buffer, bytes));
        }
        None
    }
}
