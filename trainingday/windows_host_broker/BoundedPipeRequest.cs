using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;

namespace Vera.TrainingDay.HostBroker {
  // This reader cannot grant authority. It only bounds local pipe availability.
  internal static class BoundedPipeRequest {
    internal const int MaxBytes = 128;
    internal const int TotalDeadlineMilliseconds = 1500;
    internal static string Read(Stream pipe, WaitHandle stop) {
      if (pipe == null || stop == null) throw new ArgumentNullException();
      var watch = Stopwatch.StartNew();
      var value = new StringBuilder();
      var one = new byte[1];
      bool carriageReturn = false;
      while (true) {
        int remaining = TotalDeadlineMilliseconds - (int)watch.ElapsedMilliseconds;
        if (remaining <= 0) {
          pipe.Dispose();
          throw new TimeoutException("pipe request deadline");
        }
        IAsyncResult pending = pipe.BeginRead(one, 0, 1, null, null);
        WaitHandle ready = pending.AsyncWaitHandle;
        int choice = WaitHandle.WaitAny(new WaitHandle[] { ready, stop }, remaining);
        if (choice != 0) {
          // Windows may still signal the APM wait handle from its IOCP callback.
          // Closing it before cancellation finishes crashes the entire process.
          pipe.Dispose(); // cancel pending overlapped read
          if (ready.WaitOne(4000)) {
            try { pipe.EndRead(pending); }
            catch (IOException) { }
            catch (ObjectDisposedException) { }
            ready.Close();
          }
          // If cancellation did not complete, leave the wait handle alive:
          // a late native completion must never signal an already-closed handle.
          if (choice == 1) throw new OperationCanceledException("service stopping");
          throw new TimeoutException("pipe request deadline");
        }
        int received;
        try { received = pipe.EndRead(pending); }
        finally { ready.Close(); }
        if (received != 1) throw new EndOfStreamException("incomplete request");
        byte b = one[0];
        if (b == 10) return value.ToString();
        if (carriageReturn) throw new InvalidDataException("CR must precede LF");
        if (b == 13) { carriageReturn = true; continue; }
        if (b < 32 || b > 126) throw new InvalidDataException("nonprintable request");
        if (value.Length >= MaxBytes) throw new InvalidDataException("request too long");
        value.Append((char)b);
      }
    }
  }
}
