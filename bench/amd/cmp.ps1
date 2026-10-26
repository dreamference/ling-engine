# Compares the content (messages, function calls) of ag.ps1's answers across configurations (specs/DREAMFERENCE_LING_ENGINE_AMD.md §6.3).
$d = "C:\llm\amd-port"
foreach ($t in @("t00","t01","t02","t03")) {
  $line = "$t"
  foreach ($tag in @("none","dflash7","dflash7+ngram")) {
    $f = "$d\agout-rocm-$tag-$t.json"
    if (-not (Test-Path $f)) { $line += "  ${tag}:-"; continue }
    $j = Get-Content $f -Raw | ConvertFrom-Json
    $parts = @()
    foreach ($o in $j.output) {
      if ($o.type -eq "message") { foreach ($c in $o.content) { $parts += "M:" + $c.text } }
      elseif ($o.type -eq "function_call") { $parts += "F:" + $o.name + ":" + $o.arguments }
      elseif ($o.type -eq "reasoning") { foreach ($c in $o.content) { $parts += "R:" + $c.text } }
    }
    $s = $parts -join "`n"
    $sha = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($s))).Replace("-","").Substring(0,12)
    $kinds = ($j.output | % { $_.type }) -join ","
    $line += "  {0}:{1} [{2}] {3}ch" -f $tag, $sha, $kinds, $s.Length
  }
  $line
}
