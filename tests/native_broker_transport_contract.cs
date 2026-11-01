using System;
using System.ComponentModel;
using System.IO.Pipes;
using System.Threading;
using Vera.TrainingDay.HostBroker;

public static class NativeBrokerTransportContract {
  static void Check(bool value, string name) {
    if (!value) throw new Exception(name);
    Console.WriteLine("PASS " + name);
  }

  public static int Main() {
    string name="VERA-TRAININGDAY-TRANSPORT-"+Guid.NewGuid().ToString("N");
    using(var server=BrokerPipeTransport.CreateServer(name)) {
      // The service must fail closed if another server already owns the name.
      bool denied=false;
      try { using(var duplicate=BrokerPipeTransport.CreateServer(name)) { } }
      catch(Win32Exception ex) { denied=ex.NativeErrorCode==5; }
      Check(denied,"first_instance_exclusive");

      using(var client=new NamedPipeClientStream(".",name,PipeDirection.InOut)) {
        client.Connect(1000);
        server.WaitForConnection();
        Check(server.IsConnected && client.IsConnected,"local_admin_connection");
      }
    }
    bool invalid=false;
    try { BrokerPipeTransport.CreateServer(@"remote\pipe"); }
    catch(ArgumentException) { invalid=true; }
    Check(invalid,"reject_nonlocal_pipe_name");
    Console.WriteLine("PASS ALL 3 transport contracts; no service installed or grant issued");
    return 0;
  }
}
