# The per-node cost of a captured hipGraph of ~330 near-empty kernels, against direct launches
# (the first milestone's graph-gap gate, specs/DREAMFERENCE_LING_ENGINE_AMD.md §6.5). Uses bw.co's wr kernel with n = 0.
param([int]$Nodes = 330, [int]$Reps = 200)
$ErrorActionPreference = "Stop"
$src = @"
using System; using System.Runtime.InteropServices;
public static class Gr {
  const string D = "amdhip64_7.dll";
  [DllImport(D)] static extern int hipInit(uint f);
  [DllImport(D)] static extern int hipMalloc(out IntPtr p, UIntPtr n);
  [DllImport(D)] static extern int hipStreamCreate(out IntPtr s);
  [DllImport(D)] static extern int hipStreamSynchronize(IntPtr s);
  [DllImport(D)] static extern int hipStreamBeginCapture(IntPtr s, int mode);
  [DllImport(D)] static extern int hipStreamEndCapture(IntPtr s, out IntPtr g);
  [DllImport(D)] static extern int hipGraphInstantiate(out IntPtr e, IntPtr g, IntPtr errNode, IntPtr log, UIntPtr n);
  [DllImport(D)] static extern int hipGraphLaunch(IntPtr e, IntPtr s);
  [DllImport(D)] static extern int hipModuleLoad(out IntPtr m, string path);
  [DllImport(D)] static extern int hipModuleGetFunction(out IntPtr f, IntPtr m, string name);
  [DllImport(D)] static extern int hipModuleLaunchKernel(IntPtr f, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint shm, IntPtr stream, IntPtr[] args, IntPtr extra);
  static void C(int r, string w) { if (r != 0) throw new Exception(w + " -> hipError " + r); }
  static IntPtr P(long v) { IntPtr p = Marshal.AllocHGlobal(8); Marshal.WriteInt64(p, v); return p; }
  public static string Run(string co, int nodes, int reps) {
    C(hipInit(0), "init"); IntPtr m, f, s, buf; C(hipModuleLoad(out m, co), "mod"); C(hipModuleGetFunction(out f, m, "wr"), "fn");
    C(hipMalloc(out buf, (UIntPtr)4096), "malloc"); C(hipStreamCreate(out s), "stream");
    IntPtr[] a = new IntPtr[] { P(buf.ToInt64()), P(0), P(256), P(buf.ToInt64()) };
    // direct launches
    for (int i = 0; i < nodes; i++) C(hipModuleLaunchKernel(f, 1, 1, 1, 256, 1, 1, 0, s, a, IntPtr.Zero), "warm");
    C(hipStreamSynchronize(s), "sync");
    var sw = System.Diagnostics.Stopwatch.StartNew();
    for (int r = 0; r < reps / 10; r++) for (int i = 0; i < nodes; i++) C(hipModuleLaunchKernel(f, 1, 1, 1, 256, 1, 1, 0, s, a, IntPtr.Zero), "launch");
    C(hipStreamSynchronize(s), "sync");
    double direct = sw.Elapsed.TotalMilliseconds * 1000 / (reps / 10 * nodes);
    // captured graph
    C(hipStreamBeginCapture(s, 0), "begin");
    for (int i = 0; i < nodes; i++) C(hipModuleLaunchKernel(f, 1, 1, 1, 256, 1, 1, 0, s, a, IntPtr.Zero), "cap");
    IntPtr g, e; C(hipStreamEndCapture(s, out g), "end"); C(hipGraphInstantiate(out e, g, IntPtr.Zero, IntPtr.Zero, UIntPtr.Zero), "inst");
    C(hipGraphLaunch(e, s), "glaunch"); C(hipStreamSynchronize(s), "sync");
    sw.Restart();
    for (int r = 0; r < reps; r++) C(hipGraphLaunch(e, s), "glaunch");
    C(hipStreamSynchronize(s), "sync");
    double perGraph = sw.Elapsed.TotalMilliseconds * 1000 / reps;
    return String.Format("direct launches: {0:F1} us per kernel; graph of {1} nodes: {2:F0} us per replay, {3:F2} us per node", direct, nodes, perGraph, perGraph / nodes);
  }
}
"@
Add-Type -TypeDefinition $src
[Gr]::Run("C:\llm\amd-port\bw.co", $Nodes, $Reps)
