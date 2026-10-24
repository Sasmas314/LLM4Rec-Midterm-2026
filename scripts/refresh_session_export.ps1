param([string]$Source = 'C:\Users\Asus\.codex\sessions\2026\10\24\rollout-2026-10-24T13-14-33-01a16d67-ac13-7e63-a0a5-f70142368393.jsonl')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$destination = Join-Path $projectRoot 'session-export'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$target = Join-Path $destination (Split-Path $Source -Leaf)
Copy-Item -LiteralPath $Source -Destination $target -Force
$sourceStream = [IO.File]::Open($Source, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
try {
    $hasher = [Security.Cryptography.SHA256]::Create()
    $sourceHash = [Convert]::ToHexString($hasher.ComputeHash($sourceStream))
} finally { $sourceStream.Dispose() }
$targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
if ($sourceHash -ne $targetHash) { throw 'Source changed during copying. Rerun after the chat finishes.' }
@{ source=$Source; file=(Split-Path $target -Leaf); sha256=$targetHash; bytes=(Get-Item -LiteralPath $target).Length; copied_utc=[DateTime]::UtcNow.ToString('o'); transformation='none; byte-for-byte native Codex JSONL'; finalized_after_chat=$false } | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $destination 'integrity_manifest.json')
Write-Output "Original session preserved: $target"
