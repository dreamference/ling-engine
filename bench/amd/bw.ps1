# Host for bw.hip on Windows without MSVC: C# compiled by PowerShell, calling amdhip64_7.dll through P/Invoke.
# Build the code object first: clang++ -x hip --offload-arch=gfx1151 --cuda-device-only --no-gpu-bundle-output -nogpuinc -nogpulib -O3 -o bw.co bw.hip
# (specs/DREAMFERENCE_LING_ENGINE_AMD.md §1.1, §6.1)
param([int]$GiB = 4, [int]$Iters = 20)
$ErrorActionPreference = "Stop"
$env:Path = "C:\TheRock\build\bin;" + $env:Path
$src = @"
using System; using System.Runtime.InteropServices;
public static class H {
  const string D = "amdhip64_7.dll";
  [DllImport(D)] public static extern int hipInit(uint f);
  [DllImport(D)] public static extern int hipSetDevice(int d);
  [DllImport(D)] public static extern int hipMalloc(out IntPtr p, UIntPtr n);
  [DllImport(D)] public static extern int hipFree(IntPtr p);
  [DllImport(D)] public static extern int hipMemset(IntPtr p, int v, UIntPtr n);
  [DllImport(D)] public static extern int hipMemcpy(IntPtr d, IntPtr s, UIntPtr n, int kind);
  [DllImport(D)] public static extern int hipDeviceSynchronize();
  [DllImport(D)] public static extern int hipModuleLoad(out IntPtr m, string path);
  [DllImport(D)] public static extern int hipModuleGetFunction(out IntPtr f, IntPtr m, string name);
  [DllImport(D)] public static extern int hipModuleLaunchKernel(IntPtr f, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint shm, IntPtr stream, IntPtr[] args, IntPtr extra);
  [DllImport(D)] public static extern int hipEventCreate(out IntPtr e);
  [DllImport(D)] public static extern int hipEventRecord(IntPtr e, IntPtr s);
  [DllImport(D)] public static extern int hipEventSynchronize(IntPtr e);
  [DllImport(D)] public static extern int hipEventElapsedTime(out float ms, IntPtr a, IntPtr b);
  [DllImport(D)] public static extern int hipMemGetInfo(out UIntPtr free, out UIntPtr total);
  static void C(int r, string w) { if (r != 0) throw new Exception(w + " -> hipError " + r); }
  static IntPtr Box(long v) { IntPtr p = Marshal.AllocHGlobal(8); Marshal.WriteInt64(p, v); return p; }
  public static string Run(string co, long bytes, int iters) {
    var sb = new System.Text.StringBuilder();
    C(hipInit(0), "init"); C(hipSetDevice(0), "dev");
    UIntPtr fr, tot; C(hipMemGetInfo(out fr, out tot), "meminfo");
    sb.AppendLine(String.Format("hipMemGetInfo free {0:F1} GiB total {1:F1} GiB", (double)(ulong)fr / (1L<<30), (double)(ulong)tot / (1L<<30)));
    IntPtr m; C(hipModuleLoad(out m, co), "modload");
    IntPtr a, b, o; C(hipMalloc(out a, (UIntPtr)bytes), "malloc a"); C(hipMalloc(out b, (UIntPtr)bytes), "malloc b"); C(hipMalloc(out o, (UIntPtr)256), "malloc o");
    { IntPtr ff; C(hipModuleGetFunction(out ff, m, "fill"), "getfn fill"); long nn = bytes / 16; IntPtr[] fa = new IntPtr[] { Box(a.ToInt64()), Box(nn), Box(5120L*256), Box(o.ToInt64()) }; C(hipModuleLaunchKernel(ff, 5120, 1, 1, 256, 1, 1, 0, IntPtr.Zero, fa, IntPtr.Zero), "fill"); } C(hipMemset(b, 0, (UIntPtr)bytes), "memset b"); C(hipDeviceSynchronize(), "sync");
    long n = bytes / 16;
    IntPtr e0, e1; C(hipEventCreate(out e0), "ev"); C(hipEventCreate(out e1), "ev");
    foreach (uint blocks in new uint[] { 320, 1280, 5120 }) {
      long stride = (long)blocks * 256;
      foreach (string k in new string[] { "rd", "rd4", "wr", "cp" }) {
        IntPtr f; C(hipModuleGetFunction(out f, m, k), "getfn " + k);
        IntPtr[] args = k == "cp" ? new IntPtr[] { Box(a.ToInt64()), Box(b.ToInt64()), Box(n), Box(stride) }
                                  : new IntPtr[] { Box(a.ToInt64()), Box(n), Box(stride), Box(o.ToInt64()) };
        C(hipModuleLaunchKernel(f, blocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, args, IntPtr.Zero), "warm " + k);
        C(hipDeviceSynchronize(), "sync");
        C(hipEventRecord(e0, IntPtr.Zero), "rec");
        for (int i = 0; i < iters; i++) C(hipModuleLaunchKernel(f, blocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, args, IntPtr.Zero), "launch " + k);
        C(hipEventRecord(e1, IntPtr.Zero), "rec"); C(hipEventSynchronize(e1), "evsync");
        float ms; C(hipEventElapsedTime(out ms, e0, e1), "elapsed");
        double moved = (double)bytes * iters * (k == "cp" ? 2 : 1);
        sb.AppendLine(String.Format("{0,-4} blocks {1,5}: {2,7:F1} GB/s", k, blocks, moved / (ms / 1000.0) / 1e9));
      }
    }
    C(hipEventRecord(e0, IntPtr.Zero), "rec");
    for (int i = 0; i < iters; i++) C(hipMemcpy(b, a, (UIntPtr)bytes, 3), "memcpy");
    C(hipEventRecord(e1, IntPtr.Zero), "rec"); C(hipEventSynchronize(e1), "evsync");
    float ms2; C(hipEventElapsedTime(out ms2, e0, e1), "elapsed");
    sb.AppendLine(String.Format("hipMemcpy DtoD: {0:F1} GB/s (read+write)", 2.0 * bytes * iters / (ms2 / 1000.0) / 1e9));
    hipFree(a); hipFree(b); hipFree(o);
    return sb.ToString();
  }
}
"@
Add-Type -TypeDefinition $src
[H]::Run("C:\llm\amd-port\bw.co", [long]$GiB * 1GB, $Iters)
