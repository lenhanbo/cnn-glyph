param(
    [string]$Output = "vatr-feature-map-source.zip"
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destination = [System.IO.Path]::GetFullPath((Join-Path $repo $Output))
if (-not $destination.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output must stay inside the repository: $destination"
}
$temporaryZip = Join-Path ([System.IO.Path]::GetTempPath()) ("vatr-feature-map-" + [guid]::NewGuid() + ".zip")
try {
    # Archive the current working tree (including intentional uncommitted source)
    # but never Git metadata, caches, datasets, checkpoints, or a prior archive.
    tar.exe -a -c -f $temporaryZip --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache `
        --exclude='*.pyc' --exclude='*.pth' --exclude='*.pt' --exclude='*.zip' `
        --exclude='*-32.pickle' -C $repo .
    if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
    Move-Item -LiteralPath $temporaryZip -Destination $destination -Force
} finally {
    if (Test-Path -LiteralPath $temporaryZip) { Remove-Item -LiteralPath $temporaryZip -Force }
}
Write-Host "Created $destination"
