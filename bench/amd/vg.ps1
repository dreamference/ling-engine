# Host for vg.hip, the first milestone's verify-GEMM probe: correctness against an FP32 reference, then us and GB/s per matrix
# for M = 1, 16, 32 rows, and the plain read stream over the same bytes (specs/DREAMFERENCE_LING_ENGINE_AMD.md §6.5).
param([int]$TB = 256, [int]$L = 8, [int]$Iters = 10, [string]$Co = "C:\llm\amd-port\vg.co", [int]$Quick = 0)
$ErrorActionPreference = "Stop"
$src = @"
using System; using System.Runtime.InteropServices;
public static class V {
  const string D = "amdhip64_7.dll";
  [DllImport(D)] static extern int hipInit(uint f);
  [DllImport(D)] static extern int hipMalloc(out IntPtr p, UIntPtr n);
  [DllImport(D)] static extern int hipMemcpy(float[] d, IntPtr s, UIntPtr n, int kind);
  [DllImport(D)] static extern int hipDeviceSynchronize();
  [DllImport(D)] static extern int hipModuleLoad(out IntPtr m, string path);
  [DllImport(D)] static extern int hipModuleGetFunction(out IntPtr f, IntPtr m, string name);
  [DllImport(D)] static extern int hipModuleLaunchKernel(IntPtr f, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint shm, IntPtr stream, IntPtr[] args, IntPtr extra);
  [DllImport(D)] static extern int hipEventCreate(out IntPtr e);
  [DllImport(D)] static extern int hipEventRecord(IntPtr e, IntPtr s);
  [DllImport(D)] static extern int hipEventSynchronize(IntPtr e);
  [DllImport(D)] static extern int hipEventElapsedTime(out float ms, IntPtr a, IntPtr b);
  static void C(int r, string w) { if (r != 0) throw new Exception(w + " -> hipError " + r); }
  static IntPtr P(long v) { IntPtr p = Marshal.AllocHGlobal(8); Marshal.WriteInt64(p, v); return p; }
  static IntPtr U(uint v) { IntPtr p = Marshal.AllocHGlobal(8); Marshal.WriteInt32(p, (int)v); return p; }
  static IntPtr F(float v) { IntPtr p = Marshal.AllocHGlobal(8); Marshal.WriteInt32(p, BitConverter.ToInt32(BitConverter.GetBytes(v), 0)); return p; }
  static IntPtr mod; static IntPtr e0, e1;
  static IntPtr Fn(string n) { IntPtr f; C(hipModuleGetFunction(out f, mod, n), "fn " + n); return f; }
  static void Go(string n, uint g, IntPtr[] a) { C(hipModuleLaunchKernel(Fn(n), g, 1, 1, 256, 1, 1, 0, IntPtr.Zero, a, IntPtr.Zero), "launch " + n); }
  static IntPtr Alloc(long n) { IntPtr p; C(hipMalloc(out p, (UIntPtr)n), "malloc"); return p; }
  static float[] Get(IntPtr p, int n) { float[] a = new float[n]; C(hipMemcpy(a, p, (UIntPtr)(4L * n), 2), "d2h"); return a; }
  static string Cmp(float[] a, float[] r) {
    double mx = 0, err = 0; for (int i = 0; i < a.Length; i++) { mx = Math.Max(mx, Math.Abs(r[i])); err = Math.Max(err, Math.Abs(a[i] - r[i])); }
    return String.Format("max |err| {0:E2} of max |ref| {1:E2} (rel {2:E2})", err, mx, err / mx);
  }
  public static uint TB = 256;
  public static string Run(string co, int L, int iters, int quick) {
    var sb = new System.Text.StringBuilder();
    C(hipInit(0), "init"); C(hipModuleLoad(out mod, co), "modload");
    C(hipEventCreate(out e0), "ev"); C(hipEventCreate(out e1), "ev");
    uint K = 5120;
    // FFN gate|up (NVFP4) and DeltaNet in-projection (FP8), L copies each so the MALL cannot hold them.
    uint N4 = 34816, N8 = 16384;
    long cb = (long)N4 * K / 2, sbz = (long)N4 * K / 16, fb = (long)N8 * K;
    IntPtr codes = Alloc(cb * L), scales = Alloc(sbz * L), w8 = Alloc(fb * L);
    IntPtr x = Alloc(32L * K * 2), y = Alloc(32L * N4 * 4), yr = Alloc(32L * N4 * 4), o = Alloc(256);
    Go("fill_bytes", 4096, new IntPtr[] { P(codes.ToInt64()), P(cb * L), U(1), U(0) });
    Go("fill_bytes", 4096, new IntPtr[] { P(scales.ToInt64()), P(sbz * L), U(2), U(1) });
    Go("fill_bytes", 4096, new IntPtr[] { P(w8.ToInt64()), P(fb * L), U(3), U(2) });
    Go("fill_x", 640, new IntPtr[] { P(x.ToInt64()), P(32L * K) });
    C(hipDeviceSynchronize(), "sync fill");
    // Correctness against the naive FP32 reference.
    foreach (var t in new object[][] { new object[] { "fp4_m16", 16u, true }, new object[] { "fp8_m16", 16u, false }, new object[] { "fp4t_m16", 16u, true }, new object[] { "fp4t_m32", 32u, true }, new object[] { "fp8t_m16", 16u, false }, new object[] { "fp8t_m32", 32u, false }, new object[] { "fp4w_m16", 16u, true }, new object[] { "fp4w_m32", 32u, true }, new object[] { "fp8w_m16", 16u, false }, new object[] { "fp8w_m32", 32u, false } }) {
      string kn = (string)t[0]; uint M = (uint)t[1]; bool f4 = (bool)t[2]; uint N = f4 ? N4 : N8;
      if (quick == 1 && !(kn == "fp4t_m16" || kn == "fp8t_m16")) continue;
      IntPtr w = f4 ? codes : w8;
      C(hipModuleLaunchKernel(Fn(kn), kn.Contains("t_") ? N / (TB / 2) : (kn.Contains("w_") ? N / 256 : N / 128), 1, 1, kn.Contains("t_") ? TB : 256u, 1, 1, 0, IntPtr.Zero, new IntPtr[] { P(w.ToInt64()), P(scales.ToInt64()), P(x.ToInt64()), P(y.ToInt64()), U(K), U(N), U(M), F(0.5f) }, IntPtr.Zero), "launch " + kn);
      Go("ref", (uint)((M * N + 255) / 256), new IntPtr[] { P(w.ToInt64()), P(scales.ToInt64()), P(x.ToInt64()), P(yr.ToInt64()), U(K), U(N), U(M), U(f4 ? 1u : 0u), F(0.5f), U((kn.Contains("t_") || kn.Contains("w_")) ? 1u : 0u) });
      C(hipDeviceSynchronize(), "sync check");
      sb.AppendLine(String.Format("check {0,-8} M={1,2}: {2}", kn, M, Cmp(Get(y, (int)(M * N)), Get(yr, (int)(M * N)))));
    }
    // Timing: each launch streams one copy; one timed loop walks all L copies, iters times.
    foreach (var t in new object[][] { new object[] { "fp4t_m16", 1u, true }, new object[] { "fp4t_m16", 16u, true }, new object[] { "fp4t_m32", 32u, true },
                                       new object[] { "fp8t_m16", 1u, false }, new object[] { "fp8t_m16", 16u, false }, new object[] { "fp8t_m32", 32u, false },
                                       new object[] { "fp4w_m16", 1u, true }, new object[] { "fp4w_m16", 16u, true }, new object[] { "fp4w_m32", 32u, true },
                                       new object[] { "fp8w_m16", 1u, false }, new object[] { "fp8w_m16", 16u, false }, new object[] { "fp8w_m32", 32u, false } }) {
      string kn = (string)t[0]; uint M = (uint)t[1]; bool f4 = (bool)t[2]; uint N = f4 ? N4 : N8;
      if (quick == 1 && !(kn == "fp4t_m16" || kn == "fp8t_m16")) continue;
      long bytes = f4 ? cb + sbz : fb;
      IntPtr f = Fn(kn);
      IntPtr[][] args = new IntPtr[L][];
      for (int c = 0; c < L; c++) {
        long wo = (f4 ? codes.ToInt64() + c * cb : w8.ToInt64() + c * fb), so = scales.ToInt64() + c * sbz;
        args[c] = new IntPtr[] { P(wo), P(so), P(x.ToInt64()), P(y.ToInt64()), U(K), U(N), U(M), F(0.5f) };
      }
      for (int c = 0; c < L; c++) C(hipModuleLaunchKernel(f, kn.Contains("t_") ? N / (TB / 2) : (kn.Contains("w_") ? N / 256 : N / 128), 1, 1, kn.Contains("t_") ? TB : 256u, 1, 1, 0, IntPtr.Zero, args[c], IntPtr.Zero), "warm");
      C(hipDeviceSynchronize(), "sync");
      C(hipEventRecord(e0, IntPtr.Zero), "rec");
      for (int i = 0; i < iters; i++) for (int c = 0; c < L; c++) C(hipModuleLaunchKernel(f, kn.Contains("t_") ? N / (TB / 2) : (kn.Contains("w_") ? N / 256 : N / 128), 1, 1, kn.Contains("t_") ? TB : 256u, 1, 1, 0, IntPtr.Zero, args[c], IntPtr.Zero), "launch");
      C(hipEventRecord(e1, IntPtr.Zero), "rec"); C(hipEventSynchronize(e1), "evsync");
      float ms; C(hipEventElapsedTime(out ms, e0, e1), "el");
      double per = ms / (iters * L);
      sb.AppendLine(String.Format("{0} {1,-8} M={2,2}: {3,7:F1} us per matrix ({4:F1} MB), {5,6:F1} GB/s", f4 ? "NVFP4 5120x34816" : "FP8   5120x16384", kn, M, per * 1000, bytes / 1e6, bytes / (per / 1000) / 1e9));
    }
    if (quick == 1) return sb.ToString();
    // The same bytes read by the plain read stream.
    foreach (var t in new object[][] { new object[] { codes, cb * L, "NVFP4 codes" }, new object[] { scales, sbz * L, "NVFP4 scales" }, new object[] { w8, fb * L, "FP8 weights" } }) {
      IntPtr p = (IntPtr)t[0]; long n = (long)t[1]; IntPtr f = Fn("rd");
      IntPtr[] a = new IntPtr[] { P(p.ToInt64()), P(n / 16), P(2560L * 256), P(o.ToInt64()) };
      C(hipModuleLaunchKernel(f, 2560, 1, 1, 256, 1, 1, 0, IntPtr.Zero, a, IntPtr.Zero), "warm"); C(hipDeviceSynchronize(), "s");
      C(hipEventRecord(e0, IntPtr.Zero), "rec");
      for (int i = 0; i < iters; i++) C(hipModuleLaunchKernel(f, 2560, 1, 1, 256, 1, 1, 0, IntPtr.Zero, a, IntPtr.Zero), "rd");
      C(hipEventRecord(e1, IntPtr.Zero), "rec"); C(hipEventSynchronize(e1), "evsync");
      float ms; C(hipEventElapsedTime(out ms, e0, e1), "el");
      sb.AppendLine(String.Format("read stream over {0}: {1:F1} GB/s", (string)t[2], (double)n * iters / (ms / 1000) / 1e9));
    }
    return sb.ToString();
  }
}
"@
Add-Type -TypeDefinition $src
[V]::TB = $TB
[V]::Run($Co, $L, $Iters, $Quick)
