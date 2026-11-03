// Real Windows JobObject custody of a harmless test child only.
// No rights, GPU, NAS, broker authorization or training launch is granted.
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Vera.TrainingDay.HostBroker {
internal sealed class JobObjectChildCustody : IDisposable {
    const uint CREATE_SUSPENDED=0x00000004, CREATE_NO_WINDOW=0x08000000;
    const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE=0x00002000;
    const int JobObjectExtendedLimitInformation=9;
    const uint STILL_ACTIVE=259;
    const uint WAIT_OBJECT_0=0, WAIT_FAILED=0xFFFFFFFF;
    IntPtr job, process, thread;
    bool disposed;
    public int ProcessId { get; private set; }
    private JobObjectChildCustody() {}

    // A caller-reported PID is not a process identity. These facts are read
    // from the still-owned kernel process HANDLE, never from a name or a
    // separately reopened PID (which Windows may eventually reuse).
    internal sealed class OwnedProcessIdentity {
        internal readonly int Pid;
        internal readonly ulong CreationFileTimeUtc;
        internal OwnedProcessIdentity(int pid, ulong created) {
            Pid = pid; CreationFileTimeUtc = created;
        }
        internal bool Matches(int claimedPid, ulong claimedCreationFileTimeUtc) {
            return claimedPid == Pid && claimedCreationFileTimeUtc == CreationFileTimeUtc;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    struct FileTimeValue {
        public uint LowDateTime, HighDateTime;
        internal ulong Value {
            get { return ((ulong)HighDateTime << 32) | LowDateTime; }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    struct BasicLimits {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public IntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct IoCounters {
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct ExtendedLimits {
        public BasicLimits BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    struct StartupInfo {
        public int cb;
        public string reserved, desktop, title;
        public int x,y,xSize,ySize,xCountChars,yCountChars,fillAttribute,flags;
        public short showWindow, reserved2;
        public IntPtr reserved2Ptr, stdInput, stdOutput, stdError;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct ProcessInfo {
        public IntPtr process, thread;
        public int processId, threadId;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern IntPtr CreateJobObject(IntPtr attrs, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool SetInformationJobObject(IntPtr handle, int kind,
        ref ExtendedLimits info, int length);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern bool CreateProcessW(string application, StringBuilder cmd,
        IntPtr processAttributes, IntPtr threadAttributes, bool inheritHandles,
        uint flags, IntPtr environment, string currentDirectory,
        ref StartupInfo startup, out ProcessInfo created);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool GetExitCodeProcess(IntPtr process, out uint code);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern uint GetProcessId(IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool GetProcessTimes(IntPtr process,
        out FileTimeValue creation, out FileTimeValue exit,
        out FileTimeValue kernel, out FileTimeValue user);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool TerminateProcess(IntPtr process, uint exitCode);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    // A test-only, fixed executable and loopback target. Intentionally no
    // public arbitrary-command or training launch method exists here.
    public static JobObjectChildCustody LaunchHarmlessProbe() {
        string system=Environment.GetFolderPath(Environment.SpecialFolder.System);
        string executable=Path.Combine(system, "ping.exe");
        if (!File.Exists(executable))
            throw new InvalidOperationException("Windows ping test child absent");
        JobObjectChildCustody owner=new JobObjectChildCustody();
        try {
            owner.job=CreateJobObject(IntPtr.Zero, null);
            if (owner.job==IntPtr.Zero) throw NativeError("CreateJobObject");
            var limits=new ExtendedLimits();
            limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!SetInformationJobObject(owner.job, JobObjectExtendedLimitInformation,
                    ref limits, Marshal.SizeOf(typeof(ExtendedLimits))))
                throw NativeError("SetInformationJobObject");
            var start=new StartupInfo();
            start.cb=Marshal.SizeOf(typeof(StartupInfo));
            ProcessInfo child;
            var command=new StringBuilder("\"" + executable + "\" -n 15 127.0.0.1");
            if (!CreateProcessW(executable, command, IntPtr.Zero, IntPtr.Zero,
                    false, CREATE_SUSPENDED|CREATE_NO_WINDOW, IntPtr.Zero,
                    system, ref start, out child))
                throw NativeError("CreateProcessW suspended");
            owner.process=child.process;
            owner.thread=child.thread;
            owner.ProcessId=child.processId;
            if (!AssignProcessToJobObject(owner.job, owner.process))
                throw NativeError("AssignProcessToJobObject");
            if (ResumeThread(owner.thread)==0xFFFFFFFF)
                throw NativeError("ResumeThread");
            return owner;
        } catch {
            if (owner.process!=IntPtr.Zero) TerminateProcess(owner.process, 1);
            owner.Dispose();
            throw;
        }
    }
    public bool IsInJob {
        get {
            if (disposed) throw new ObjectDisposedException("JobObjectChildCustody");
            bool inside;
            if (!IsProcessInJob(process, job, out inside))
                throw NativeError("IsProcessInJob");
            return inside;
        }
    }
    public bool IsAlive {
        get {
            if (disposed) throw new ObjectDisposedException("JobObjectChildCustody");
            uint code;
            if (!GetExitCodeProcess(process, out code))
                throw NativeError("GetExitCodeProcess");
            return code==STILL_ACTIVE;
        }
    }
    internal OwnedProcessIdentity ReadOwnedKernelIdentityForNativeTestOnly() {
        if (disposed) throw new ObjectDisposedException("JobObjectChildCustody");
        if (process == IntPtr.Zero)
            throw new InvalidOperationException("owned process handle absent");
        uint kernelPid = GetProcessId(process);
        if (kernelPid == 0) throw NativeError("GetProcessId");
        if (kernelPid > Int32.MaxValue || kernelPid != (uint)ProcessId)
            throw new InvalidOperationException("child PID does not match retained HANDLE");
        FileTimeValue creation, exit, kernel, user;
        if (!GetProcessTimes(process, out creation, out exit, out kernel, out user))
            throw NativeError("GetProcessTimes");
        if (creation.Value == 0)
            throw new InvalidOperationException("owned child creation time missing");
        return new OwnedProcessIdentity((int)kernelPid, creation.Value);
    }
    // For the native reservation regression only: close the owning Job,
    // then verify actual termination using the still-open process HANDLE,
    // never a claimed PID or process name that could be reused.
    internal bool CloseJobAndConfirmChildExit() {
        if (disposed) throw new ObjectDisposedException("JobObjectChildCustody");
        if (job==IntPtr.Zero || process==IntPtr.Zero)
            throw new InvalidOperationException("test child custody already released");
        if (!CloseHandle(job)) throw NativeError("CloseHandle JobObject");
        job=IntPtr.Zero;
        uint wait=WaitForSingleObject(process, 5000);
        if (wait==WAIT_FAILED) throw NativeError("WaitForSingleObject");
        return wait==WAIT_OBJECT_0;
    }
    public void Dispose() {
        if (disposed) return;
        disposed=true;
        // Closing the final JobObject handle kills every assigned test child.
        if (job!=IntPtr.Zero) { CloseHandle(job); job=IntPtr.Zero; }
        if (thread!=IntPtr.Zero) { CloseHandle(thread); thread=IntPtr.Zero; }
        if (process!=IntPtr.Zero) { CloseHandle(process); process=IntPtr.Zero; }
    }
    private static Win32Exception NativeError(string call) {
        return new Win32Exception(Marshal.GetLastWin32Error(), call + " failed");
    }
}
}
