"""Windows-only suspended launch and JobObject custody for one fixed child.

This is NOT a TRAININGDAY training authorization or a host-wide exclusive lease.
It confines one process after host preflight.  Import only on Windows.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes as w
from dataclasses import dataclass
import hashlib
import msvcrt
import os
from pathlib import Path
import subprocess
import time
from typing import Callable


if os.name != "nt":
    raise RuntimeError("Windows JobObject supervisor cannot run on non-Windows hosts")

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
secur32 = ctypes.WinDLL("secur32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
SIZE_T = ctypes.c_size_t
ULONG_PTR = ctypes.c_size_t
DWORD = w.DWORD
HANDLE = w.HANDLE
LPVOID = ctypes.c_void_p
INVALID_HANDLE = ctypes.c_void_p(-1).value

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
STARTF_USESTDHANDLES = 0x00000100
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
KILL_ON_JOB_CLOSE = 0x00002000
JOB_MEMORY = 0x00000200
ACTIVE_PROCESS = 0x00000008
JOB_CPU_RATE_ENABLE = 0x00000001
JOB_CPU_RATE_HARD_CAP = 0x00000004
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
WAIT_FAILED = 0xFFFFFFFF
TH32CS_SNAPPROCESS = 0x00000002
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


class Win32Hold(RuntimeError):
    """Windows primitive failed: no eligible model workload may proceed."""


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(x, ctypes.c_ulonglong) for x in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", DWORD),
        ("MinimumWorkingSetSize", SIZE_T),
        ("MaximumWorkingSetSize", SIZE_T),
        ("ActiveProcessLimit", DWORD),
        ("Affinity", ULONG_PTR),
        ("PriorityClass", DWORD),
        ("SchedulingClass", DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", SIZE_T),
        ("JobMemoryLimit", SIZE_T),
        ("PeakProcessMemoryUsed", SIZE_T),
        ("PeakJobMemoryUsed", SIZE_T),
    ]


class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
    _fields_ = [("ControlFlags", DWORD), ("CpuRate", DWORD)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", DWORD), ("lpReserved", w.LPWSTR), ("lpDesktop", w.LPWSTR),
        ("lpTitle", w.LPWSTR), ("dwX", DWORD), ("dwY", DWORD),
        ("dwXSize", DWORD), ("dwYSize", DWORD), ("dwXCountChars", DWORD),
        ("dwYCountChars", DWORD), ("dwFillAttribute", DWORD),
        ("dwFlags", DWORD), ("wShowWindow", w.WORD), ("cbReserved2", w.WORD),
        ("lpReserved2", LPVOID), ("hStdInput", HANDLE), ("hStdOutput", HANDLE),
        ("hStdError", HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", HANDLE), ("hThread", HANDLE),
                ("dwProcessId", DWORD), ("dwThreadId", DWORD)]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", LPVOID)]


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", DWORD),
        ("TotalProcesses", DWORD),
        ("ActiveProcesses", DWORD),
        ("TotalTerminatedProcesses", DWORD),
    ]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", DWORD), ("dwMemoryLoad", DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", DWORD), ("dwHighDateTime", DWORD)]

    def as_int(self) -> int:
        return (self.dwHighDateTime << 32) | self.dwLowDateTime


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD), ("cntUsage", DWORD), ("th32ProcessID", DWORD),
        ("th32DefaultHeapID", ULONG_PTR), ("th32ModuleID", DWORD),
        ("cntThreads", DWORD), ("th32ParentProcessID", DWORD),
        ("pcPriClassBase", ctypes.c_long), ("dwFlags", DWORD),
        ("szExeFile", w.WCHAR * 260),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", DWORD), ("PageFaultCount", DWORD),
        ("PeakWorkingSetSize", SIZE_T), ("WorkingSetSize", SIZE_T),
        ("QuotaPeakPagedPoolUsage", SIZE_T), ("QuotaPagedPoolUsage", SIZE_T),
        ("QuotaPeakNonPagedPoolUsage", SIZE_T), ("QuotaNonPagedPoolUsage", SIZE_T),
        ("PagefileUsage", SIZE_T), ("PeakPagefileUsage", SIZE_T),
        ("PrivateUsage", SIZE_T),
    ]


def _bind(name, restype, *argtypes):
    fn = getattr(kernel, name)
    fn.argtypes = list(argtypes)
    fn.restype = restype
    return fn


CreateFileW = _bind("CreateFileW", HANDLE, w.LPCWSTR, DWORD, DWORD,
                    LPVOID, DWORD, DWORD, HANDLE)
GetFileAttributesW = _bind("GetFileAttributesW", DWORD, w.LPCWSTR)
CreateJobObjectW = _bind("CreateJobObjectW", HANDLE, LPVOID, w.LPCWSTR)
SetInformationJobObject = _bind("SetInformationJobObject", w.BOOL, HANDLE, ctypes.c_int,
                                LPVOID, DWORD)
QueryInformationJobObject = _bind("QueryInformationJobObject", w.BOOL, HANDLE, ctypes.c_int,
                                  LPVOID, DWORD, LPVOID)
CreateProcessW = _bind("CreateProcessW", w.BOOL, w.LPCWSTR, w.LPWSTR,
                       LPVOID, LPVOID, w.BOOL, DWORD, LPVOID,
                       w.LPCWSTR, ctypes.POINTER(STARTUPINFOW),
                       ctypes.POINTER(PROCESS_INFORMATION))
AssignProcessToJobObject = _bind("AssignProcessToJobObject", w.BOOL, HANDLE, HANDLE)
IsProcessInJob = _bind("IsProcessInJob", w.BOOL, HANDLE, HANDLE, ctypes.POINTER(w.BOOL))
ResumeThread = _bind("ResumeThread", DWORD, HANDLE)
TerminateProcess = _bind("TerminateProcess", w.BOOL, HANDLE, DWORD)
TerminateJobObject = _bind("TerminateJobObject", w.BOOL, HANDLE, DWORD)
WaitForSingleObject = _bind("WaitForSingleObject", DWORD, HANDLE, DWORD)
GetExitCodeProcess = _bind("GetExitCodeProcess", w.BOOL, HANDLE, ctypes.POINTER(DWORD))
CloseHandle = _bind("CloseHandle", w.BOOL, HANDLE)
GlobalMemoryStatusEx = _bind("GlobalMemoryStatusEx", w.BOOL, ctypes.POINTER(MEMORYSTATUSEX))
GetSystemTimes = _bind("GetSystemTimes", w.BOOL, ctypes.POINTER(FILETIME),
                       ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME))
GetComputerNameExW = _bind("GetComputerNameExW", w.BOOL, ctypes.c_int,
                           w.LPWSTR, ctypes.POINTER(DWORD))
GetUserNameExW = secur32.GetUserNameExW
GetUserNameExW.argtypes = [ctypes.c_int, w.LPWSTR, ctypes.POINTER(DWORD)]
GetUserNameExW.restype = w.BOOL
GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
GetProcessMemoryInfo.argtypes = [HANDLE, LPVOID, DWORD]
GetProcessMemoryInfo.restype = w.BOOL
OpenProcess = _bind("OpenProcess", HANDLE, DWORD, w.BOOL, DWORD)
GetProcessTimes = _bind("GetProcessTimes", w.BOOL, HANDLE,
                        ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
                        ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME))
CreateToolhelp32Snapshot = _bind("CreateToolhelp32Snapshot", HANDLE, DWORD, DWORD)
Process32FirstW = _bind("Process32FirstW", w.BOOL, HANDLE, ctypes.POINTER(PROCESSENTRY32W))
Process32NextW = _bind("Process32NextW", w.BOOL, HANDLE, ctypes.POINTER(PROCESSENTRY32W))
InitializeProcThreadAttributeList = _bind(
    "InitializeProcThreadAttributeList", w.BOOL, LPVOID, DWORD, DWORD,
    ctypes.POINTER(SIZE_T))
UpdateProcThreadAttribute = _bind(
    "UpdateProcThreadAttribute", w.BOOL, LPVOID, DWORD, ULONG_PTR,
    LPVOID, SIZE_T, LPVOID, LPVOID)
DeleteProcThreadAttributeList = _bind("DeleteProcThreadAttributeList", None, LPVOID)


def _err(name: str) -> Win32Hold:
    return Win32Hold(f"{name} failed: Win32 {ctypes.get_last_error()}")


def _close(handle) -> None:
    if handle and handle != INVALID_HANDLE:
        CloseHandle(handle)


def available_physical_bytes() -> int:
    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(s)
    if not GlobalMemoryStatusEx(ctypes.byref(s)):
        raise _err("GlobalMemoryStatusEx")
    return int(s.ullAvailPhys)


def observed_windows_identity() -> tuple[str, str]:
    """Read kernel/secur32 identity, never client-provided environment text."""
    host_buf = ctypes.create_unicode_buffer(256)
    host_len = DWORD(len(host_buf))
    if not GetComputerNameExW(5, host_buf, ctypes.byref(host_len)):
        raise _err("GetComputerNameExW")
    user_buf = ctypes.create_unicode_buffer(512)
    user_len = DWORD(len(user_buf))
    if not GetUserNameExW(2, user_buf, ctypes.byref(user_len)):
        raise _err("GetUserNameExW SamCompatible")
    return host_buf.value, user_buf.value


def process_working_set(handle) -> tuple[int, int]:
    mem = PROCESS_MEMORY_COUNTERS_EX()
    mem.cb = ctypes.sizeof(mem)
    if not GetProcessMemoryInfo(handle, ctypes.byref(mem), mem.cb):
        raise _err("GetProcessMemoryInfo")
    return int(mem.WorkingSetSize), int(mem.PeakWorkingSetSize)


def host_cpu_percent() -> float:
    def sample():
        idle, kernel_time, user_time = FILETIME(), FILETIME(), FILETIME()
        if not GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel_time), ctypes.byref(user_time)):
            raise _err("GetSystemTimes")
        return idle.as_int(), kernel_time.as_int(), user_time.as_int()

    a = sample()
    time.sleep(1.0)
    b = sample()
    idle_delta = b[0] - a[0]
    total_delta = (b[1] - a[1]) + (b[2] - a[2])
    if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
        raise Win32Hold("invalid host CPU sample")
    return 100.0 * (total_delta - idle_delta) / total_delta


def running_named_processes() -> dict[str, set[int]]:
    snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == INVALID_HANDLE:
        raise _err("CreateToolhelp32Snapshot")
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        if not Process32FirstW(snapshot, ctypes.byref(entry)):
            raise _err("Process32FirstW")
        names: dict[str, set[int]] = {}
        while True:
            names.setdefault(entry.szExeFile.lower(), set()).add(int(entry.th32ProcessID))
            if not Process32NextW(snapshot, ctypes.byref(entry)):
                if ctypes.get_last_error() == 18:  # ERROR_NO_MORE_FILES
                    return names
                raise _err("Process32NextW")
    finally:
        _close(snapshot)


def running_executable_names() -> set[str]:
    return set(running_named_processes())


def observed_process_snapshot(pid: int) -> tuple[int, int]:
    """Return OS-observed (creation-time, working-set bytes) for a fixed PID."""
    process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not process:
        raise _err("OpenProcess daemon PID")
    try:
        created, exited, kernel_time, user_time = (
            FILETIME(), FILETIME(), FILETIME(), FILETIME()
        )
        if not GetProcessTimes(process, ctypes.byref(created), ctypes.byref(exited),
                               ctypes.byref(kernel_time), ctypes.byref(user_time)):
            raise _err("GetProcessTimes daemon")
        rss, _ = process_working_set(process)
        return created.as_int(), rss
    finally:
        _close(process)


def assert_not_reparse(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        attr = GetFileAttributesW(str(current))
        if attr == INVALID_FILE_ATTRIBUTES:
            raise _err(f"GetFileAttributesW {current.name}")
        if attr & FILE_ATTRIBUTE_REPARSE_POINT:
            raise Win32Hold(f"reparse point forbidden in staged path: {current.name}")


class PinnedReadFile:
    """Hold an OS handle denying write/delete shares until child terminates."""

    def __init__(self, path: Path):
        assert_not_reparse(path)
        handle = CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ,
                             None, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, None)
        if not handle or handle == INVALID_HANDLE:
            raise _err("CreateFileW immutable pin")
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
            self.file = os.fdopen(descriptor, "rb")
        except Exception:
            _close(handle)
            raise

    def sha256(self) -> str:
        self.file.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: self.file.read(1024 * 1024), b""):
            digest.update(chunk)
        self.file.seek(0)
        return digest.hexdigest()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class ChildResult:
    pid: int
    exit_code: int | None
    reason: str
    peak_job_commit_bytes: int
    last_free_ram_bytes: int
    job_active_processes_after: int
    process_handle_signaled: bool
    peak_working_set_bytes: int
    accounting_poll_count: int
    accounting_wait_seconds: float


def bounded_accounting_drain(read_active: Callable[[], int], *,
                             timeout_seconds: float = 2.0,
                             interval_seconds: float = 0.05,
                             clock: Callable[[], float] = time.monotonic,
                             pause: Callable[[float], None] = time.sleep) -> tuple[int, float]:
    """Post-handle-release Job active-process reconciliation.

    Windows can retain an exited process in ActiveProcesses while callers
    retain references. The OS process handle MUST be signaled, its status
    captured, and its process/thread handles closed BEFORE calling this.
    """
    if timeout_seconds <= 0 or interval_seconds <= 0:
        raise Win32Hold("invalid Job accounting wait budget")
    began = clock()
    deadline = began + timeout_seconds
    polls = 0
    while True:
        active = read_active()
        polls += 1
        if active == 0:
            return polls, clock() - began
        if type(active) is not int or active < 0:
            raise Win32Hold("invalid Job ActiveProcesses accounting value")
        remaining = deadline - clock()
        if remaining <= 0:
            raise Win32Hold(f"Job ActiveProcesses did not drain after handle close: {active}")
        pause(min(interval_seconds, remaining))


def run_job(*, executable: Path, argv: list[str], cwd: Path,
            stdout_path: Path, stderr_path: Path, memory_cap: int,
            cpu_percent: int, wall_seconds: int, runtime_free_floor: int,
            forced_test_stop: bool = False,
            occupancy_guard: Callable[[int], None] | None = None) -> ChildResult:
    """Launch suspended, assign to bounded Job, then resume; fail closed.

    Job commit limit does not cap file-backed RSS; host RAM is monitored too.
    A model process can still affect other user workloads within CPU/RAM caps;
    this is one-job process custody, not host-wide exclusive ownership.
    """
    job = CreateJobObjectW(None, None)
    if not job:
        raise _err("CreateJobObjectW")
    child = PROCESS_INFORMATION()
    out = err = stdin = None
    started = False
    attribute_buffer = None
    attribute_initialized = False
    try:
        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE | JOB_MEMORY | ACTIVE_PROCESS
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        limits.JobMemoryLimit = memory_cap
        if not SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise _err("SetInformationJobObject memory/process")
        rate = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION()
        rate.ControlFlags = JOB_CPU_RATE_ENABLE | JOB_CPU_RATE_HARD_CAP
        rate.CpuRate = cpu_percent * 100
        if not SetInformationJobObject(job, 15, ctypes.byref(rate), ctypes.sizeof(rate)):
            raise _err("SetInformationJobObject CPU hard cap")

        # Dedicated output files are never overwritten. The three inheritable
        # std handles are opened only for this one CreateProcessW call.
        out = open(stdout_path, "xb", buffering=0)
        err = open(stderr_path, "xb", buffering=0)
        stdin = open(os.devnull, "rb", buffering=0)
        streams = [stdin, out, err]
        for stream in streams:
            os.set_handle_inheritable(msvcrt.get_osfhandle(stream.fileno()), True)

        si = STARTUPINFOEXW()
        si.StartupInfo.cb = ctypes.sizeof(si)
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES
        si.StartupInfo.hStdInput = msvcrt.get_osfhandle(stdin.fileno())
        si.StartupInfo.hStdOutput = msvcrt.get_osfhandle(out.fileno())
        si.StartupInfo.hStdError = msvcrt.get_osfhandle(err.fileno())
        # Only these three pipe/file handles may enter the child, not a
        # coincidentally inheritable Executor/Python connection or file.
        attribute_size = SIZE_T(0)
        InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(attribute_size))
        if attribute_size.value == 0:
            raise _err("InitializeProcThreadAttributeList size")
        attribute_buffer = ctypes.create_string_buffer(attribute_size.value)
        si.lpAttributeList = ctypes.cast(attribute_buffer, LPVOID)
        if not InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0,
                                                 ctypes.byref(attribute_size)):
            raise _err("InitializeProcThreadAttributeList")
        attribute_initialized = True
        listed_handles = (HANDLE * 3)(
            si.StartupInfo.hStdInput, si.StartupInfo.hStdOutput,
            si.StartupInfo.hStdError,
        )
        if not UpdateProcThreadAttribute(
            si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            ctypes.cast(listed_handles, LPVOID), ctypes.sizeof(listed_handles),
            None, None,
        ):
            raise _err("UpdateProcThreadAttribute HANDLE_LIST")
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
        # Never inherit Executor credentials, tokens, proxy settings, secrets,
        # or arbitrary PATH from the orchestration process.
        environment_lines = [
            r"SystemRoot=C:\Windows", r"WINDIR=C:\Windows",
            r"SystemDrive=C:", r"PATH=C:\Windows\System32",
            "TEMP=" + str(stdout_path.parent),
            "TMP=" + str(stdout_path.parent),
            "OMP_NUM_THREADS=2",
        ]
        environment = ctypes.create_unicode_buffer(
            "\0".join(sorted(environment_lines, key=str.casefold)) + "\0\0"
        )
        flags = (CREATE_SUSPENDED | CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
                 | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT)
        if not CreateProcessW(str(executable), command, None, None, True,
                              flags, ctypes.cast(environment, LPVOID), str(cwd),
                              ctypes.cast(ctypes.byref(si), ctypes.POINTER(STARTUPINFOW)),
                              ctypes.byref(child)):
            raise _err("CreateProcessW suspended")
        started = True
        # Inheritable handles are not needed after the child was created.
        for stream in streams:
            os.set_handle_inheritable(msvcrt.get_osfhandle(stream.fileno()), False)
        if not AssignProcessToJobObject(job, child.hProcess):
            raise _err("AssignProcessToJobObject")
        in_job = w.BOOL(False)
        if not IsProcessInJob(child.hProcess, job, ctypes.byref(in_job)) or not in_job.value:
            raise Win32Hold("OS did not verify process membership before resume")
        if occupancy_guard is not None:
            # The sole Job-owned child is not an unrelated competing runner.
            # Its PID is retained on a real OS process HANDLE, not claimed by
            # the caller; every other llama-cli PID remains disallowed.
            occupancy_guard(int(child.dwProcessId))
        if available_physical_bytes() < runtime_free_floor + memory_cap:
            raise Win32Hold("host RAM fell before ResumeThread: child remains suspended")
        if ResumeThread(child.hThread) == WAIT_FAILED:
            raise _err("ResumeThread")

        started_at = time.monotonic()
        reason = "CHILD_EXITED"
        last_free = available_physical_bytes()
        peak_rss = 0
        while True:
            result = WaitForSingleObject(child.hProcess, 1000)
            if result == WAIT_OBJECT_0:
                break
            if result != WAIT_TIMEOUT:
                raise _err("WaitForSingleObject")
            last_free = available_physical_bytes()
            rss, rss_peak = process_working_set(child.hProcess)
            peak_rss = max(peak_rss, rss_peak)
            if last_free < runtime_free_floor:
                reason = "HOST_FREE_RAM_FLOOR"
                break
            if rss > memory_cap:
                reason = "PROCESS_RSS_LIMIT"
                break
            if occupancy_guard is not None:
                try:
                    occupancy_guard(int(child.dwProcessId))
                except Exception:
                    reason = "CONCURRENT_MODEL_OR_OCCUPANCY_UNVERIFIED"
                    break
            if (os.fstat(out.fileno()).st_size > 1024 * 1024 or
                    os.fstat(err.fileno()).st_size > 1024 * 1024):
                reason = "OUTPUT_FILE_LIMIT"
                break
            if forced_test_stop:
                reason = "SELFTEST_FORCED_STOP"
                break
            if time.monotonic() - started_at >= wall_seconds:
                reason = "WALL_TIME_LIMIT"
                break
        if reason != "CHILD_EXITED":
            if not TerminateJobObject(job, 90):
                raise _err("TerminateJobObject")
            if WaitForSingleObject(child.hProcess, 5000) != WAIT_OBJECT_0:
                raise Win32Hold("Job killed but child termination not confirmed")

        code = DWORD(0)
        if not GetExitCodeProcess(child.hProcess, ctypes.byref(code)):
            raise _err("GetExitCodeProcess")
        # A signaled process can still count as Job 'ActiveProcesses' until
        # all references to that process are released. Retain and examine the
        # original HANDLE through exit, then close thread/process handles
        # BEFORE waiting for kernel Job accounting to reach zero. Closing the
        # process handle here is safe: Wait was signaled and exit code latched.
        if child.hThread:
            if not CloseHandle(child.hThread):
                raise _err("CloseHandle exited child thread")
            child.hThread = None
        if child.hProcess:
            if not CloseHandle(child.hProcess):
                raise _err("CloseHandle exited child process")
            child.hProcess = None

        def read_active() -> int:
            accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
            if not QueryInformationJobObject(job, 1, ctypes.byref(accounting),
                                             ctypes.sizeof(accounting), None):
                raise _err("QueryInformationJobObject active processes")
            return int(accounting.ActiveProcesses)

        accounting_polls, accounting_wait = bounded_accounting_drain(read_active)
        final = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        if not QueryInformationJobObject(job, 9, ctypes.byref(final), ctypes.sizeof(final), None):
            raise _err("QueryInformationJobObject")
        return ChildResult(int(child.dwProcessId), int(code.value), reason,
                           int(final.PeakJobMemoryUsed), last_free,
                           0, True, peak_rss, accounting_polls, accounting_wait)
    finally:
        # Even failure *before* Job assignment cannot strand the suspended
        # child. Verify its retained process HANDLE, not a possibly reused PID.
        cleanup_error = None
        if started and child.hProcess:
            status = WaitForSingleObject(child.hProcess, 0)
            if status == WAIT_TIMEOUT:
                if job:
                    TerminateJobObject(job, 91)
                status = WaitForSingleObject(child.hProcess, 2000)
                if status == WAIT_TIMEOUT:
                    # Assignment may have failed; kill that exact held child.
                    if not TerminateProcess(child.hProcess, 91):
                        cleanup_error = "TerminateProcess failed outside Job"
                    status = WaitForSingleObject(child.hProcess, 5000)
            if status != WAIT_OBJECT_0:
                cleanup_error = "cannot verify child exit by original process handle"
        _close(job)
        _close(child.hThread)
        _close(child.hProcess)
        for stream in (stdin, out, err):
            if stream is not None:
                try:
                    os.set_handle_inheritable(msvcrt.get_osfhandle(stream.fileno()), False)
                except OSError:
                    pass
                stream.close()
        if attribute_initialized:
            DeleteProcThreadAttributeList(si.lpAttributeList)
        if cleanup_error:
            raise Win32Hold(cleanup_error)
