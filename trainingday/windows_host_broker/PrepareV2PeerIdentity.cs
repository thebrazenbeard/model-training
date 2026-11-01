// R185 source-only PREPARE_V2 peer observation for an independently installed
// Windows service. NOT compiled into the R161 installed deny-only broker.
// This class never grants, reserves GPUs, opens training data, or launches jobs.
using System;
using System.ComponentModel;
using System.IO;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Security.Principal;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace Vera.TrainingDay.HostBroker {
  internal sealed class PrepareV2PeerIdentity : IDisposable {
    public const string State = "HOLD";
    public bool AuthorizesTraining { get { return false; } }

    const uint PROCESS_QUERY_LIMITED_INFORMATION = 0x00001000;
    const uint SYNCHRONIZE = 0x00100000;
    const uint TOKEN_QUERY = 0x00000008;
    const uint WAIT_TIMEOUT = 0x00000102;
    const uint WAIT_OBJECT_0 = 0;

    [StructLayout(LayoutKind.Sequential)]
    struct FileTime {
      public uint Low;
      public uint High;
      public ulong Value { get { return ((ulong)High << 32) | Low; } }
    }

    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool GetNamedPipeClientProcessId(SafePipeHandle pipe, out uint pid);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool GetProcessTimes(IntPtr process, out FileTime created,
      out FileTime exited, out FileTime kernel, out FileTime user);
    [DllImport("kernel32.dll", EntryPoint="QueryFullProcessImageNameW",
      CharSet=CharSet.Unicode, ExactSpelling=true, SetLastError=true)]
    static extern bool QueryFullProcessImageNameW(IntPtr process, uint flags,
      StringBuilder image, ref uint imageChars);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("advapi32.dll", SetLastError=true)]
    static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool CloseHandle(IntPtr handle);

    public uint ProcessId { get; private set; }
    public ulong CreationFileTime { get; private set; }
    public string ObservedImagePath { get; private set; }
    public string ObservedSid { get; private set; }
    public TokenImpersonationLevel ObservedImpersonationLevel { get; private set; }
    private IntPtr processHandle;
    private bool disposed;

    private PrepareV2PeerIdentity() { }

    // Call only AFTER a completed bounded pipe read. The pipe peer, token,
    // PID, birth time and executable path come from Windows, not JSON fields.
    // All results remain observations: installed image trust policy, immutable
    // request, NAS witness, rights and GPU authority are still absent.
    internal static PrepareV2PeerIdentity Capture(NamedPipeServerStream pipe) {
      if (pipe == null || !pipe.IsConnected)
        throw new UnauthorizedAccessException("connected local pipe required");
      uint observedPid;
      if (!GetNamedPipeClientProcessId(pipe.SafePipeHandle, out observedPid)
          || observedPid == 0)
        throw NativeError("GetNamedPipeClientProcessId");

      IntPtr processHandle = OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, false, observedPid);
      if (processHandle == IntPtr.Zero) throw NativeError("OpenProcess");
      try {
        RequireLive(processHandle);
        ulong creationFileTime = ReadCreation(processHandle);
        string observedImagePath = ReadImage(processHandle);

        string observedSid = null;
        TokenImpersonationLevel impersonation = TokenImpersonationLevel.None;
        pipe.RunAsClient(() => {
          using (var identity = WindowsIdentity.GetCurrent(true)) {
            if (identity == null || identity.User == null)
              throw new UnauthorizedAccessException("client impersonation absent");
            impersonation = identity.ImpersonationLevel;
            if (impersonation != TokenImpersonationLevel.Impersonation &&
                impersonation != TokenImpersonationLevel.Delegation)
              throw new UnauthorizedAccessException("identification-only token rejected");
            if (!new WindowsPrincipal(identity).IsInRole(WindowsBuiltInRole.Administrator))
              throw new UnauthorizedAccessException("client token not administrator");
            observedSid = identity.User.Value;
          }
        });
        if (String.IsNullOrEmpty(observedSid))
          throw new UnauthorizedAccessException("client SID missing");

        // An impersonated pipe token is insufficient to bind process identity
        // unless the OS process-token owner also matches. No caller assertion
        // substitutes for either Windows observation.
        IntPtr token = IntPtr.Zero;
        try {
          if (!OpenProcessToken(processHandle, TOKEN_QUERY, out token) || token == IntPtr.Zero)
            throw NativeError("OpenProcessToken");
          using (var processIdentity = new WindowsIdentity(token)) {
            if (processIdentity.User == null ||
                !String.Equals(processIdentity.User.Value, observedSid,
                  StringComparison.Ordinal))
              throw new UnauthorizedAccessException("pipe/process token owners differ");
          }
        } finally {
          if (token != IntPtr.Zero) CloseHandle(token);
        }

        uint confirmedPid;
        if (!GetNamedPipeClientProcessId(pipe.SafePipeHandle, out confirmedPid)
            || observedPid != confirmedPid)
          throw new UnauthorizedAccessException("pipe peer PID changed");
        ulong confirmedCreationFileTime = ReadCreation(processHandle);
        if (creationFileTime != confirmedCreationFileTime ||
            !String.Equals(observedImagePath, ReadImage(processHandle),
              StringComparison.OrdinalIgnoreCase))
          throw new UnauthorizedAccessException("process identity changed");
        RequireLive(processHandle);

        var result = new PrepareV2PeerIdentity {
          ProcessId = observedPid,
          CreationFileTime = creationFileTime,
          ObservedImagePath = observedImagePath,
          ObservedSid = observedSid,
          ObservedImpersonationLevel = impersonation,
          processHandle = processHandle
        };
        processHandle = IntPtr.Zero; // Ownership moved; caller must Dispose.
        return result;
      } finally {
        if (processHandle != IntPtr.Zero) CloseHandle(processHandle);
      }
    }

    // Caller-provided strings can only cause additional denials. Even an
    // exact match cannot authorize anything without a durable anti-replay
    // witness, protected executable hash policy and independent grants.
    internal string DenyClaim(uint claimedPid, string claimedSid,
        string claimedImagePath, ulong claimedCreationFileTime) {
      if (disposed) throw new ObjectDisposedException("PrepareV2PeerIdentity");
      if (WaitForSingleObject(processHandle, 0) != WAIT_TIMEOUT)
        return "HOLD_PROCESS_EXITED_OR_UNOBSERVABLE";
      if (claimedPid != ProcessId || claimedCreationFileTime != CreationFileTime ||
          !String.Equals(claimedSid, ObservedSid, StringComparison.Ordinal) ||
          !String.Equals(claimedImagePath, ObservedImagePath,
            StringComparison.OrdinalIgnoreCase))
        return "HOLD_PEER_MISMATCH";
      return "HOLD_NO_DURABLE_REPLAY_WITNESS";
    }

    private static void RequireLive(IntPtr process) {
      if (WaitForSingleObject(process, 0) != WAIT_TIMEOUT)
        throw new InvalidOperationException("PROCESS_EXITED_OR_UNOBSERVABLE");
    }
    private static ulong ReadCreation(IntPtr process) {
      FileTime created, exited, kernel, user;
      if (!GetProcessTimes(process, out created, out exited, out kernel, out user) ||
          created.Value == 0)
        throw NativeError("GetProcessTimes");
      return created.Value;
    }
    private static string ReadImage(IntPtr process) {
      var image = new StringBuilder(32768);
      uint length = (uint)image.Capacity;
      if (!QueryFullProcessImageNameW(process, 0, image, ref length) || length == 0)
        throw NativeError("QueryFullProcessImageNameW");
      return image.ToString();
    }
    private static Win32Exception NativeError(string call) {
      return new Win32Exception(Marshal.GetLastWin32Error(), call + " failed");
    }
    public void Dispose() {
      if (disposed) return;
      disposed = true;
      if (processHandle != IntPtr.Zero) {
        CloseHandle(processHandle);
        processHandle = IntPtr.Zero;
      }
    }
  }
}
