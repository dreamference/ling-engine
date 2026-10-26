# Loads bw.co through the driver's System32 HIP runtime with TheRock off PATH (specs/DREAMFERENCE_LING_ENGINE_AMD.md §6.4).
$ErrorActionPreference = "Stop"
$src = @"
using System; using System.Runtime.InteropServices;
public static class R {
  const string D = "amdhip64_7.dll";
  [DllImport(D)] public static extern int hipInit(uint f);
  [DllImport(D)] public static extern int hipRuntimeGetVersion(out int v);
  [DllImport(D)] public static extern int hipModuleLoad(out IntPtr m, string path);
  [DllImport(D)] public static extern int hipModuleGetFunction(out IntPtr f, IntPtr m, string name);
  [DllImport(D)] public static extern int hipMalloc(out IntPtr p, UIntPtr n);
  [DllImport(D)] public static extern int hipModuleLaunchKernel(IntPtr f, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint shm, IntPtr stream, IntPtr[] args, IntPtr extra);
  [DllImport(D)] public static extern int hipDeviceSynchronize();
  static IntPtr Box(long v) { IntPtr p = Marshal.AllocHGlobal(8); Marshal.WriteInt64(p, v); return p; }
  public static string Run(string co) {
    int v; int a = hipInit(0); hipRuntimeGetVersion(out v);
    IntPtr m, f, b, o; int l = hipModuleLoad(out m, co); int g = hipModuleGetFunction(out f, m, "rd");
    hipMalloc(out b, (UIntPtr)(1 << 20)); hipMalloc(out o, (UIntPtr)256);
    int k = hipModuleLaunchKernel(f, 4, 1, 1, 256, 1, 1, 0, IntPtr.Zero, new IntPtr[] { Box(b.ToInt64()), Box(65536), Box(1024), Box(o.ToInt64()) }, IntPtr.Zero);
    int s = hipDeviceSynchronize();
    return String.Format("init {0} runtime {1} modload {2} getfn {3} launch {4} sync {5}", a, v, l, g, k, s);
  }
}
"@
Add-Type -TypeDefinition $src
[R]::Run("C:\llm\amd-port\bw.co")
(Get-Process -Id $PID).Modules | ? { $_.ModuleName -match "amdhip|comgr|kpack" } | % { $_.FileName + "  " + $_.FileVersionInfo.FileVersion }
"PATH has TheRock: " + ($env:Path -match "TheRock")
