# Host for wmma.hip: WMMA throughput per type (specs/DREAMFERENCE_LING_ENGINE_AMD.md §1.1).
param([int]$Iters = 20000)
$ErrorActionPreference = "Stop"
$env:Path = "C:\TheRock\build\bin;" + $env:Path
$src = @"
using System; using System.Runtime.InteropServices;
public static class W {
  const string D = "amdhip64_7.dll";
  [DllImport(D)] public static extern int hipInit(uint f);
  [DllImport(D)] public static extern int hipMalloc(out IntPtr p, UIntPtr n);
  [DllImport(D)] public static extern int hipDeviceSynchronize();
  [DllImport(D)] public static extern int hipModuleLoad(out IntPtr m, string path);
  [DllImport(D)] public static extern int hipModuleGetFunction(out IntPtr f, IntPtr m, string name);
  [DllImport(D)] public static extern int hipModuleLaunchKernel(IntPtr f, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint shm, IntPtr stream, IntPtr[] args, IntPtr extra);
  [DllImport(D)] public static extern int hipEventCreate(out IntPtr e);
  [DllImport(D)] public static extern int hipEventRecord(IntPtr e, IntPtr s);
  [DllImport(D)] public static extern int hipEventSynchronize(IntPtr e);
  [DllImport(D)] public static extern int hipEventElapsedTime(out float ms, IntPtr a, IntPtr b);
  static void C(int r, string w) { if (r != 0) throw new Exception(w + " -> hipError " + r); }
  static IntPtr Box8(long v) { IntPtr p = Marshal.AllocHGlobal(8); Marshal.WriteInt64(p, v); return p; }
  static IntPtr Box4(int v) { IntPtr p = Marshal.AllocHGlobal(8); Marshal.WriteInt32(p, v); return p; }
  public static string Run(string co, int iters) {
    var sb = new System.Text.StringBuilder();
    C(hipInit(0), "init");
    IntPtr m; C(hipModuleLoad(out m, co), "modload");
    IntPtr o; C(hipMalloc(out o, (UIntPtr)256), "malloc");
    IntPtr e0, e1; C(hipEventCreate(out e0), "ev"); C(hipEventCreate(out e1), "ev");
    foreach (string k in new string[] { "f16", "bf16", "iu8", "iu4" }) {
      IntPtr f; C(hipModuleGetFunction(out f, m, k), "getfn " + k);
      uint blocks = 40 * 16; uint threads = 256;
      IntPtr[] args = new IntPtr[] { Box8(o.ToInt64()), Box4(iters) };
      C(hipModuleLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, IntPtr.Zero, args, IntPtr.Zero), "warm");
      C(hipDeviceSynchronize(), "sync");
      C(hipEventRecord(e0, IntPtr.Zero), "rec");
      C(hipModuleLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, IntPtr.Zero, args, IntPtr.Zero), "launch");
      C(hipEventRecord(e1, IntPtr.Zero), "rec"); C(hipEventSynchronize(e1), "evsync");
      float ms; C(hipEventElapsedTime(out ms, e0, e1), "elapsed");
      double waves = blocks * (threads / 32.0);
      double ops = waves * iters * 8.0 * 2.0 * 16 * 16 * 16;
      sb.AppendLine(String.Format("{0,-5}: {1,7:F1} T(FL)OPS dense  ({2:F1} ms)", k, ops / (ms / 1000.0) / 1e12, ms));
    }
    return sb.ToString();
  }
}
"@
Add-Type -TypeDefinition $src
[W]::Run("C:\llm\amd-port\wmma.co", $Iters)
