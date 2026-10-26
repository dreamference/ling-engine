# The first four agentlong turns (bench/agentlong; bodies from bench-sets, not in the repository) through llama-server's /v1/responses
# with Mightling's patched chat template (specs/DREAMFERENCE_LING_ENGINE_AMD.md §6.3). cmp.ps1 compares the answers' content.
param([string]$Backend = "rocm", [string]$Tag = "none", [string]$Extra = "--spec-type none", [int]$Port = 18766)
$ErrorActionPreference = "Continue"
$dir = "C:\llm\amd-port"
if ($Backend -eq "rocm") { $env:Path = "C:\TheRock\build\bin;C:\llm\rocm;" + $env:Path; $env:HIP_PATH = "C:\TheRock\build" }
$exe = "C:\llm\$Backend\llama-server.exe"
$log = "$dir\ag-$Backend-$Tag.log"
$a = @("-m","C:\llm\models\Qwen3.8-27B-UD-Q4_K_XL.gguf","-ngl","99","-fa","on","-lm","none","-c","40960","-np","1","--host","127.0.0.1","--port","$Port","--jinja","--chat-template-file","$dir\qwen38-patched.jinja","--min-p","0","--temp","0","--cache-ram","0") + ($Extra -split " " | ? { $_ })
$p = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardError $log -RedirectStandardOutput "$log.out" -PassThru -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 180; $i++) { Start-Sleep 2; try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.status -eq "ok") { $ok = $true; break } } catch {} ; if ($p.HasExited) { break } }
if (-not $ok) { "SERVER FAILED (exited=$($p.HasExited))"; Get-Content $log -Tail 25; if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; exit 1 }
foreach ($t in @("t00","t01","t02","t03")) {
  $body = [IO.File]::ReadAllBytes("$dir\ag-$t.json")
  $sw = [Diagnostics.Stopwatch]::StartNew()
  try {
    $r = Invoke-WebRequest "http://127.0.0.1:$Port/v1/responses" -Method Post -Body $body -ContentType "application/json; charset=utf-8" -TimeoutSec 1800 -UseBasicParsing
    $txt = $r.Content
  } catch { $txt = "ERROR: $_" }
  $sw.Stop()
  [IO.File]::WriteAllText("$dir\agout-$Backend-$Tag-$t.json", $txt)
  $j = $null; try { $j = $txt | ConvertFrom-Json } catch {}
  $outs = if ($j) { ($j.output | ConvertTo-Json -Depth 12 -Compress) } else { $txt }
  $sha = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($outs))).Replace("-","").Substring(0,12)
  Start-Sleep 1
  $tim = Select-String -Path $log -Pattern "prompt eval time|       eval time|draft acceptance" | Select -Last 3 | % { ($_.Line -replace '^.*print_timing: ','') -replace '\s+',' ' }
  "{0} {1} {2} wall {3:F1}s usage in={4} out={5} sha {6}" -f $Backend, $Tag, $t, $sw.Elapsed.TotalSeconds, $j.usage.input_tokens, $j.usage.output_tokens, $sha
  $tim | % { "   $_" }
}
Stop-Process -Id $p.Id -Force
Start-Sleep 3
