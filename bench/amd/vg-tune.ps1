# Compiles vg.hip with several layouts and block sizes and times the 16-row kernels (specs/DREAMFERENCE_LING_ENGINE_AMD.md §6.5).
$d = "C:\llm\amd-port"; $cl = "C:\TheRock\build\lib\llvm\bin\clang++.exe"
$cfgs = @(@("-DKC=256 -DNT -DKMAJOR", 256), @("-DKC=256 -DNT -DKMAJOR -DBWAVES=16", 512), @("-DKC=512 -DNT -DKMAJOR -DBWAVES=16", 512), @("-DKC=256 -DNT -DKMAJOR -DBWAVES=32 -DXROWS=16", 1024))
$i = 20
foreach ($c in $cfgs) {
  $co = "$d\vgq$i.co"
  & $cl -x hip -std=c++17 --offload-arch=gfx1151 --cuda-device-only --no-gpu-bundle-output -nogpuinc -nogpulib -O3 ($c[0] -split " ") -o $co "$d\vg.hip" 2>&1 | Select -First 3
  "== $($c[0])"
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$d\vg.ps1" -Co $co -Quick 1 -TB $c[1] 2>&1 | ? { $_ -match "us per matrix|check|rror" } | % { "   $_" }
  $i++
}
