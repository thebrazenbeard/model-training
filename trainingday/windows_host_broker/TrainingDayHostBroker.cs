using System;
using System.IO;
using System.IO.Pipes;
using System.Security.Principal;
using System.ServiceProcess;
using System.Diagnostics;
using System.Threading;

namespace Vera.TrainingDay.HostBroker {
  sealed class BrokerService : ServiceBase {
    const string PipeName="VERA-TrainingDay-HostBroker-V1";
    volatile bool stopping;
    Thread worker;
    readonly ManualResetEvent stopSignal = new ManualResetEvent(false);
    readonly string bootId=Guid.NewGuid().ToString("N");
    public BrokerService() { ServiceName="VeraTrainingDayHostBroker"; CanStop=true; AutoLog=false; }
    protected override void OnStart(string[] args) {
      stopping=false;
      stopSignal.Reset();
      worker=new Thread(Loop) { IsBackground=true, Name="DenyUntilIndependentAttestation" };
      worker.Start();
    }
    protected override void OnStop() {
      stopping=true;
      stopSignal.Set();
      try { using(var c=new NamedPipeClientStream(".", PipeName, PipeDirection.Out)) c.Connect(300); }
      catch { }
      if(worker!=null) worker.Join(1200);
    }
    static bool VerifiedAdminClient(NamedPipeServerStream pipe) {
      bool admin=false;
      pipe.RunAsClient(() => {
        using(var ident=WindowsIdentity.GetCurrent(true)) {
          if(ident==null || ident.User==null) throw new UnauthorizedAccessException();
          admin=new WindowsPrincipal(ident).IsInRole(WindowsBuiltInRole.Administrator);
        }
      });
      return admin;
    }
    void Loop() {
      while(!stopping) {
        try {
          using(var pipe=BrokerPipeTransport.CreateServer(PipeName)) {
            pipe.WaitForConnection();
            if(stopping) return;
            using(var reader=new StreamReader(pipe,System.Text.Encoding.ASCII,false,1024,true))
            using(var writer=new StreamWriter(pipe,System.Text.Encoding.ASCII,1024,true) { AutoFlush=true }) {
              // Windows caches the client's impersonation identity upon a
              // completed pipe read, not merely upon WaitForConnection.
              string request=BoundedPipeRequest.Read(pipe,stopSignal);
              bool authenticated=false;
              try { authenticated=VerifiedAdminClient(pipe); } catch { }
              writer.WriteLine(BrokerHoldResponse.Format(request,authenticated,
                Process.GetCurrentProcess().Id,bootId));
            }
          }
        } catch(Exception ex) {
          try {
            string dir=@"C:\ProgramData\VERA\TrainingDayHostBroker";
            Directory.CreateDirectory(dir);
            File.AppendAllText(Path.Combine(dir,"broker-errors.log"),
              DateTime.UtcNow.ToString("o")+" "+ex.GetType().Name+Environment.NewLine);
          } catch { }
          Thread.Sleep(200);
        }
      }
    }
    static void Main() { ServiceBase.Run(new BrokerService()); }
  }
}
