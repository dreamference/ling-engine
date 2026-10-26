# Times the 16-row NVFP4 kernel with one part switched off at a time: staging (NOX), decode (NODEC), WMMA (NOWMMA) (specs/DREAMFERENCE_LING_ENGINE_AMD.md §6.5).
$d = "C:\llm\amd-port"; $cl = "C:\TheRock\build\lib\llvm\bin\clang++.exe"
$cfgs = @("-DKC=256 -DNT -DKMAJOR", "-DKC=256 -DNT -DKMAJOR -DNOX", "-DKC=256 -DNT -DKMAJOR -DNODEC", "-DKC=256 -DNT -DKMAJOR -DNOWMMA")
$i = 30
foreach ($c in $cfgs) {
  $co = "$d\vgq$i.co"
  & $cl -x hip -std=c++17 --offload-arch=gfx1151 --cuda-device-only --no-gpu-bundle-output -nogpuinc -nogpulib -O3 ($c -split " ") -o $co "$d\vg.hip" 2>&1 | Select -First 3
  "== $c"
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$d\vg.ps1" -Co $co -Quick 1 2>&1 | ? { $_ -match "us per matrix|check" } | % { "   $_" }
  $i++
}
