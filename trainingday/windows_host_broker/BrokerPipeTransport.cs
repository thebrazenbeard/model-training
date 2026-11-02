using System;
using System.ComponentModel;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using Microsoft.Win32.SafeHandles;

namespace Vera.TrainingDay.HostBroker {
  // LocalSystem service boundary: remote SMB pipe clients and pipe-name squatting
  // must fail before any future authority-bearing request can be parsed.
  internal static class BrokerPipeTransport {
    const uint PipeAccessDuplex = 0x00000003;
    const uint FileFlagOverlapped = 0x40000000;
    const uint FileFlagFirstPipeInstance = 0x00080000;
    const uint PipeRejectRemoteClients = 0x00000008;
    const uint PipeMaxInstances = 1;
    const uint BufferSize = 4096;

    [StructLayout(LayoutKind.Sequential)]
    struct SecurityAttributes {
      public int Length;
      public IntPtr Descriptor;
      [MarshalAs(UnmanagedType.Bool)] public bool InheritHandle;
    }

    [DllImport("kernel32.dll", EntryPoint="CreateNamedPipeW", CharSet=CharSet.Unicode,
      SetLastError=true, ExactSpelling=true)]
    static extern SafePipeHandle CreateNamedPipe(string name, uint openMode,
      uint pipeMode, uint maxInstances, uint outBufferSize, uint inBufferSize,
      uint defaultTimeout, ref SecurityAttributes security);

    internal static PipeSecurity StrictPipeAcl() {
      var acl=new PipeSecurity();
      acl.SetAccessRuleProtection(true,false);
      acl.AddAccessRule(new PipeAccessRule(
        new SecurityIdentifier(WellKnownSidType.LocalSystemSid,null),
        PipeAccessRights.FullControl,AccessControlType.Allow));
      acl.AddAccessRule(new PipeAccessRule(
        new SecurityIdentifier(WellKnownSidType.BuiltinAdministratorsSid,null),
        PipeAccessRights.ReadWrite,AccessControlType.Allow));
      return acl;
    }

    internal static NamedPipeServerStream CreateServer(string pipeName) {
      if (String.IsNullOrEmpty(pipeName) || pipeName.Length > 240 ||
          pipeName.IndexOf('\\') >= 0 || pipeName.IndexOf('/') >= 0)
        throw new ArgumentException("invalid local pipe name", "pipeName");

      byte[] descriptor=StrictPipeAcl().GetSecurityDescriptorBinaryForm();
      IntPtr nativeDescriptor=Marshal.AllocHGlobal(descriptor.Length);
      try {
        Marshal.Copy(descriptor,0,nativeDescriptor,descriptor.Length);
        var attributes=new SecurityAttributes {
          Length=Marshal.SizeOf(typeof(SecurityAttributes)),
          Descriptor=nativeDescriptor,
          InheritHandle=false
        };
        SafePipeHandle handle=CreateNamedPipe(@"\\.\pipe\"+pipeName,
          PipeAccessDuplex|FileFlagOverlapped|FileFlagFirstPipeInstance,
          PipeRejectRemoteClients,PipeMaxInstances,BufferSize,BufferSize,0,
          ref attributes);
        if (handle.IsInvalid) {
          int error=Marshal.GetLastWin32Error();
          handle.Dispose();
          throw new Win32Exception(error,"secure pipe creation failed");
        }
        try { return new NamedPipeServerStream(PipeDirection.InOut,true,false,handle); }
        catch { handle.Dispose(); throw; }
      } finally { Marshal.FreeHGlobal(nativeDescriptor); }
    }
  }
}
