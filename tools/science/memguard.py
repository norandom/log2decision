"""Requirement 15 for the development scripts of task 2.17 (Windows only).

- `limit(gib)`: puts this process into a Windows job object whose process memory limit is
  `gib` GiB of committed memory. An allocation beyond it fails inside the process
  (`MemoryError`, or a failed allocation in a native library) instead of the host's
  guard ending the process at 92 % of physical memory.
- `peaks()`: the peak committed memory (private bytes) and the peak working set of this
  process, from `GetProcessMemoryInfo`; recorded per run.
- `hungry(name)`: a named lock for the project on this machine. A second hungry process
  refuses, naming the holder (criterion 4 of the design's operating criteria).
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from ctypes import wintypes

GIB = 1024 ** 3
LOCK = os.environ.get("L2D_HUNGRY_LOCK", r"D:\BuildArtifacts\log2decision\hungry.lock")


class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimit(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_JOB = None
_LIMIT_GIB = None


def limit(gib: float) -> None:
    """Sets the process memory limit (committed memory) of this process to `gib` GiB."""
    global _JOB, _LIMIT_GIB
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
    info = _ExtendedLimit()
    info.BasicLimitInformation.LimitFlags = 0x00000100      # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = int(gib * GIB)
    if not kernel.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
    if not kernel.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(kernel.GetCurrentProcess())):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
    _JOB, _LIMIT_GIB = job, gib


def peaks() -> dict:
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    psapi.GetProcessMemoryInfo(wintypes.HANDLE(kernel.GetCurrentProcess()), ctypes.byref(counters), counters.cb)
    return {"limit_gib": _LIMIT_GIB, "peak_committed_gib": round(counters.PeakPagefileUsage / GIB, 3),
            "peak_working_set_gib": round(counters.PeakWorkingSetSize / GIB, 3),
            "note": "committed memory is what the limit counts; the working set includes pages of memory-mapped files"}


def host_memory() -> dict:
    class Status(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = Status()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {"physical_gib": round(status.ullTotalPhys / GIB, 2), "available_gib": round(status.ullAvailPhys / GIB, 2),
            "load_percent": int(status.dwMemoryLoad)}


class hungry:
    """`with hungry("training fold 0 seed 0"):` holds the project's lock for hungry
    processes; a second one refuses and names the holder."""

    def __init__(self, name: str):
        self.name = name
        self.handle = None

    def __enter__(self):
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        try:
            self.handle = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(LOCK, encoding="utf-8") as handle:
                    holder = json.load(handle)
            except (OSError, ValueError):
                holder = "unknown"
            # A lock whose process is gone is stale.
            pid = holder.get("pid") if isinstance(holder, dict) else None
            if pid and not _alive(pid):
                os.remove(LOCK)
                return self.__enter__()
            raise SystemExit(f"refused: another hungry process of the project holds the host: {holder}")
        os.write(self.handle, json.dumps({"pid": os.getpid(), "operation": self.name,
                                          "since": time.strftime("%Y-%m-%d %H:%M:%S")}).encode())
        os.close(self.handle)
        return self

    def __exit__(self, *_):
        try:
            os.remove(LOCK)
        except OSError:
            pass


def _alive(pid: int) -> bool:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = wintypes.DWORD()
    kernel.GetExitCodeProcess(wintypes.HANDLE(handle), ctypes.byref(code))
    kernel.CloseHandle(wintypes.HANDLE(handle))
    return code.value == 259                                  # STILL_ACTIVE
