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
from pathlib import Path
import msvcrt

GIB = 1024 ** 3
LOCK = str(Path(os.environ.get("ProgramData", r"C:\ProgramData")) /
           "log2decision-resource-gates" / "host.lock")


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
    """Host-wide gate shared with l2d-core. OS ownership, never PID-file deletion.

    `wait=False` refuses naming the holder; `wait=True` waits and records that fact.
    Hardware NPU utilization by software outside this protocol remains unknown.
    """

    def __init__(self, name: str, wait: bool = False):
        if not name or len(name.encode()) > 512:
            raise ValueError("operation name must contain 1 to 512 bytes")
        self.name, self.wait = name, wait
        self.handle = None
        self.waited = False
        self.observation = None
        self.witness = None
        self.witness_path = None

    @staticmethod
    def _holder(path, observer=None):
        started = time.monotonic()
        while True:
            try:
                with open(Path(path).with_suffix(".holder.json"), encoding="utf-8") as handle:
                    holder = json.loads(handle.read(4096))
                name = holder.get("lease_file", "")
                if not isinstance(holder.get("operation"), str) or not isinstance(holder.get("process_id"), int) or not name or Path(name).name != name:
                    raise ValueError("invalid holder or lease witness")
                witness = os.open(Path(path).with_name(name), os.O_RDWR | os.O_BINARY)
                try:
                    if hungry._try(witness):
                        raise ValueError("stale or incomplete holder publication")
                finally:
                    os.close(witness)
                return holder
            except (OSError, ValueError, AttributeError) as error:
                probe = os.open(path, os.O_RDWR | os.O_BINARY)
                try:
                    if hungry._try(probe):
                        return None
                finally:
                    os.close(probe)
                if time.monotonic() - started >= 5:
                    raise SystemExit(f"refused: cannot read resource holder: {error}") from error
                if observer is not None:
                    observer.waited = True
                time.sleep(0.01)

    @staticmethod
    def _try(handle):
        os.lseek(handle, 0, os.SEEK_SET)
        try:
            msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as error:
            if error.errno in (13, 36):
                return False
            raise

    @classmethod
    def _observe(cls, path, observer=None):
        handle = os.open(path, os.O_CREAT | os.O_RDWR | os.O_BINARY)
        try:
            return None if cls._try(handle) else cls._holder(path, observer)
        finally:
            os.close(handle)

    def __enter__(self):
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        self.observation = {"host_memory": host_memory(),
                            "host_holder": self._observe(LOCK, self),
                            "npu_holder": self._observe(str(Path(LOCK).with_name("npu.lock")), self),
                            "external_npu_utilization": "unknown"}
        self.handle = os.open(LOCK, os.O_CREAT | os.O_RDWR | os.O_BINARY)
        try:
            announced = False
            while not self._try(self.handle):
                holder = self._holder(LOCK, self)
                if holder is None:
                    continue
                message = f"{holder['operation']} (process {holder['process_id']}) holds host; requested {self.name}"
                if not self.wait:
                    raise SystemExit(f"refused: {message}")
                if not announced:
                    print(f"waiting: {message}", file=__import__('sys').stderr, flush=True)
                    announced = True
                self.waited = True
                time.sleep(0.025)
            name = f"lease-{os.getpid()}-{time.time_ns()}.lock"
            self.witness_path = Path(LOCK).with_name(name)
            self.witness = os.open(self.witness_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_BINARY)
            if not self._try(self.witness):
                raise OSError("new lease witness was busy")
            publication = self.witness_path.with_suffix(".publishing")
            with open(publication, "w", encoding="utf-8") as handle:
                json.dump({"process_id": os.getpid(), "operation": self.name, "lease_file": name}, handle)
            os.replace(publication, Path(LOCK).with_suffix(".holder.json"))
            return self
        except BaseException:
            if self.witness is not None:
                os.close(self.witness)
                self.witness = None
            os.close(self.handle)
            self.handle = None
            raise

    def report(self):
        return {"gate_held": True, "operation": self.name, "waited_for_host": self.waited,
                "observation": self.observation, "memory_scope": "this process only",
                "spilled": False}

    def __exit__(self, *args):
        if self.handle is not None:
            try:
                try:
                    Path(LOCK).with_suffix(".holder.json").unlink(missing_ok=True)
                finally:
                    if self.witness is not None:
                        os.close(self.witness)
                        self.witness = None
                        self.witness_path.unlink(missing_ok=True)
            except OSError as error:
                if not args or args[0] is None:
                    raise
                print(f"resource gate cleanup failed: {error}", file=__import__('sys').stderr)
            finally:
                os.close(self.handle)
                self.handle = None
