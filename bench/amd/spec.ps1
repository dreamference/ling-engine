# llama-server (release zips) on prompts.json at temperature 0: decode, draft acceptance, answer hashes (specs/DREAMFERENCE_LING_ENGINE_AMD.md §6.2).
param([string]$Backend = "vulkan", [string]$Tag = "none", [string]$Extra = "", [int]$Port = 18765)
$ErrorActionPreference = "Continue"
$dir = "C:\llm\amd-port"
if ($Backend -eq "rocm") { $env:Path = "C:\TheRock\build\bin;C:\llm\rocm;" + $env:Path; $env:HIP_PATH = "C:\TheRock\build" }
$exe = "C:\llm\$Backend\llama-server.exe"
$log = "$dir\server-$Backend-$Tag.log"
$a = @("-m","C:\llm\models\Qwen3.8-27B-UD-Q4_K_XL.gguf","-ngl","99","-fa","on","-lm","none","-c","16384","-np","1","--host","127.0.0.1","--port","$Port","--jinja","--min-p","0","--temp","0") + ($Extra -split " " | ? { $_ })
$p = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardError $log -RedirectStandardOutput "$log.out" -PassThru -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 180; $i++) { Start-Sleep 2; try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.status -eq "ok") { $ok = $true; break } } catch {} ; if ($p.HasExited) { break } }
if (-not $ok) { "SERVER FAILED (exited=$($p.HasExited))"; Get-Content $log -Tail 25; if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; exit 1 }
$prompts = Get-Content "$dir\prompts.json" -Raw | ConvertFrom-Json
$wb = @{ messages = @(@{ role = "user"; content = "Say hello in five words." }); max_tokens = 16; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 6 -Compress
try { Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -Body $wb -ContentType "application/json" -TimeoutSec 600 | Out-Null } catch { "warmup failed: $_" }
foreach ($name in @("prose","code","edit")) {
  $max = if ($name -eq "edit") { 900 } else { 450 }
  $body = @{ messages = @(@{ role = "user"; content = $prompts.$name }); max_tokens = $max; temperature = 0; top_k = 1; seed = 1; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 6 -Compress
  $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ContentType "application/json; charset=utf-8" -TimeoutSec 900
  $t = $r.timings
  $txt = $r.choices[0].message.content
  [IO.File]::WriteAllText("$dir\out-$Backend-$Tag-$name.txt", $txt)
  $sha = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($txt))).Replace("-","").Substring(0,12)
  $acc = if ($t.draft_n) { "{0}/{1} = {2:P0}" -f $t.draft_n_accepted, $t.draft_n, ($t.draft_n_accepted / [double]$t.draft_n) } else { "-" }
  "{0,-7} {1,-6} {2,-14} n={3,4} decode {4,6:F1} tok/s  prefill {5,6:F1} tok/s  drafts accepted {6}  sha {7}" -f $Backend, $name, $Tag, $t.predicted_n, $t.predicted_per_second, $t.prompt_per_second, $acc, $sha
}
Stop-Process -Id $p.Id -Force
Start-Sleep 3
Select-String -Path $log -Pattern "accept|dflash|draft" | Select -Last 6 | % { "  log: " + $_.Line }
