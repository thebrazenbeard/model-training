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
  // Keep a byte arriving every 100 ms, well inside the per-read idle window.
  // A reader that incorrectly restarts its 1,500 ms deadline per byte must
  // hit the outer watchdog, not pass merely because the client went idle.
  static void CheckPacedSlowDrip() {
    const string name = "slow_drip_total_deadline";
    Console.WriteLine("BEGIN " + name);
    string id = "VERA-TRAININGDAY-CONTRACT-" + Guid.NewGuid().ToString("N");
    using (var server = new NamedPipeServerStream(id, PipeDirection.InOut, 1,
      PipeTransmissionMode.Byte, PipeOptions.Asynchronous))
    using (var client = new NamedPipeClientStream(".", id, PipeDirection.Out))
    using (var stop = new ManualResetEvent(false))
    using (var writerReady = new ManualResetEvent(false)) {
      client.Connect(1000);
      server.WaitForConnection();
      int sent = 0;
      var writerThread = new Thread(() => {
        writerReady.Set();
        try {
          for (int i = 0; i < 120; i++) {
            client.WriteByte((byte)'S');
            Interlocked.Increment(ref sent);
            Thread.Sleep(100);
          }
        } catch (IOException) { /* reader timeout closes the pipe */ }
        catch (ObjectDisposedException) { /* fixture teardown */ }
      }) { IsBackground = true };
      writerThread.Start();
      if (!writerReady.WaitOne(3000))
        throw new Exception(name + " writer thread did not start");

      var watch = Stopwatch.StartNew();
      Exception failure = null;
      int watchdogFired = 0;
      using (var watchdog = new Timer(_ => {
        Interlocked.Exchange(ref watchdogFired, 1);
        try { stop.Set(); } catch (ObjectDisposedException) { }
      }, null, 7000, Timeout.Infinite)) {
        try { BoundedPipeRequest.Read(server, stop); }
        catch (Exception ex) { failure = ex; }
      }
      int sentBeforeRejection = Interlocked.CompareExchange(ref sent, 0, 0);
      if (failure == null || failure.GetType() != typeof(TimeoutException) ||
          Interlocked.CompareExchange(ref watchdogFired, 0, 0) != 0)
        throw new Exception(name + " expected total-deadline TimeoutException before watchdog; got " +
          (failure == null ? "success" : failure.GetType().Name));
      if (sentBeforeRejection < 4)
        throw new Exception(name + " fixture did not actually deliver paced bytes");
      Console.WriteLine("PASS " + name + " elapsed_ms=" + watch.ElapsedMilliseconds +
        " paced_bytes=" + sentBeforeRejection);
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
    CheckPacedSlowDrip();
    Check("stop_cancels", null, typeof(OperationCanceledException), stopFirst:true);
    Check("usable_after_timeout", "STATUS_V1\n", null, "STATUS_V1");
    Console.WriteLine("PASS ALL 10 request/cancellation contracts");
    return 0;
  }
}
