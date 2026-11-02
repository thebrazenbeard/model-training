using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using Vera.TrainingDay.HostBroker;

public static class NativeBrokerRequestContract {
  static void Check(string name, string input, Type error, string expected = null,
                    bool stopFirst = false) {
    Console.WriteLine("BEGIN " + name);
    string id = "VERA-TRAININGDAY-CONTRACT-" + Guid.NewGuid().ToString("N");
    using (var server = new NamedPipeServerStream(id, PipeDirection.InOut, 1,
      PipeTransmissionMode.Byte, PipeOptions.Asynchronous))
    using (var client = new NamedPipeClientStream(".", id, PipeDirection.Out))
    using (var stop = new ManualResetEvent(false)) {
      client.Connect(1000);
      Console.WriteLine("CLIENT_CONNECTED " + name);
      server.WaitForConnection();
      Console.WriteLine("SERVER_CONNECTED " + name);
      Thread writerThread = null;
      if (input != null) {
        byte[] data = Encoding.ASCII.GetBytes(input);
        writerThread = new Thread(() => {
          try { client.Write(data, 0, data.Length); client.Flush(); }
          catch (IOException) { /* rejection can close pipe while client writes */ }
        }) { IsBackground = true };
        writerThread.Start();
      }
      if (stopFirst) stop.Set();
      Console.WriteLine("BEFORE_READ " + name);
      var watch = Stopwatch.StartNew();
      Exception failure = null;
      string result = null;
      try { result = BoundedPipeRequest.Read(server, stop); }
      catch (Exception ex) { failure = ex; }
      if (error == null && failure != null)
        throw new Exception(name + " unexpectedly rejected", failure);
      if (error != null && (failure == null || failure.GetType() != error))
        throw new Exception(name + " expected " + error.Name +
          " but got " + (failure == null ? "success" : failure.GetType().Name));
      if (error == null && result != expected)
        throw new Exception(name + " unexpected response: " + result);
      if (watch.ElapsedMilliseconds > 4000)
        throw new Exception(name + " exceeded hard test deadline");
      Console.WriteLine("PASS " + name + " elapsed_ms=" + watch.ElapsedMilliseconds);
    }
  }
  public static int Main(string[] args) {
    Check("valid_LF", "STATUS_V1\n", null, "STATUS_V1");
    Check("valid_CRLF", "PREPARE_V1\r\n", null, "PREPARE_V1");
    Check("max_128", new string('X', 128) + "\n", null, new string('X',128));
    Check("reject_129", new string('X', 129) + "\n", typeof(InvalidDataException));
    Check("reject_embedded_NUL", "STATUS\0V1\n", typeof(InvalidDataException));
    Check("reject_unterminated_CR", "STATUS\rX\n", typeof(InvalidDataException));
    Check("idle_timeout", null, typeof(TimeoutException));
    Check("slow_drip_timeout", "S", typeof(TimeoutException));
    Check("stop_cancels", null, typeof(OperationCanceledException), stopFirst:true);
    Check("usable_after_timeout", "STATUS_V1\n", null, "STATUS_V1");
    Console.WriteLine("PASS ALL 10 request/cancellation contracts");
    return 0;
  }
}
