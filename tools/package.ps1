param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location -LiteralPath $projectRoot
try {
    & python tools\bundle_assets.py
    if ($LASTEXITCODE -ne 0) { throw 'Validated models/dataset missing; complete model preparation before packaging.' }
    if (-not $SkipBuild) {
        & .\gradlew.bat :app:testDebugUnitTest :app:lintDebug :app:assembleDebug --console=plain
        if ($LASTEXITCODE -ne 0) { throw 'Build/test/lint failed; artifacts not packaged.' }
    }
    $apk = Join-Path $projectRoot 'app\build\outputs\apk\debug\app-debug.apk'
    if (-not (Test-Path -LiteralPath $apk)) { throw 'APK missing; run the build first.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $apkArchive = [System.IO.Compression.ZipFile]::OpenRead($apk)
    try {
        foreach ($name in @('gridformer','yolo_detect','yolo_segment')) {
            $entry = $apkArchive.GetEntry("assets/models/$name.tflite")
            if ($null -eq $entry) { throw "APK does not contain $name; rebuild after bundling." }
            $entryStream = $entry.Open()
            try { $sha = [System.Security.Cryptography.SHA256]::Create(); try { $actual = [System.BitConverter]::ToString($sha.ComputeHash($entryStream)).Replace('-','').ToLowerInvariant() } finally { $sha.Dispose() } } finally { $entryStream.Dispose() }
            $expected = (Get-FileHash -LiteralPath (Join-Path $projectRoot "tools\exports\$name.tflite") -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $expected) { throw "APK model hash mismatch: $name" }
        }
    } finally { $apkArchive.Dispose() }
    $distDir = Join-Path $projectRoot 'dist'
    New-Item -ItemType Directory -Path $distDir -Force | Out-Null
    Copy-Item -LiteralPath $apk -Destination (Join-Path $distDir 'GridRestore-debug.apk') -Force
    $evidenceDir = Join-Path $distDir 'validation'
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
    foreach ($item in @('app\build\test-results\testDebugUnitTest', 'app\build\reports\tests\testDebugUnitTest', 'app\build\reports\lint-results-debug.html', 'app\build\reports\lint-results-debug.xml')) {
        $source = Join-Path $projectRoot $item
        if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $evidenceDir -Recurse -Force }
    }
    Add-Type -AssemblyName System.IO.Compression
    $sourceZip = Join-Path $distDir 'GridRestore-source.zip'
    $fileStream = [System.IO.File]::Open($sourceZip, [System.IO.FileMode]::Create)
    try {
        $archive = [System.IO.Compression.ZipArchive]::new($fileStream, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            $roots = @('app\src', 'gradle', 'docs', 'tools')
            $sourceFiles = foreach ($path in $roots) { Get-ChildItem -LiteralPath (Join-Path $projectRoot $path) -File -Recurse }
            $sourceFiles += Get-ChildItem -LiteralPath $projectRoot -File
            $sourceFiles += Get-Item -LiteralPath (Join-Path $projectRoot 'app\build.gradle.kts')
            foreach ($file in $sourceFiles | Sort-Object FullName -Unique) {
                $relative = $file.FullName.Substring($projectRoot.Length + 1).Replace('\', '/')
                if ($relative -eq 'local.properties' -or $relative -match '^tools/(downloads|vendor|work|exports|\.venv[^/]*)(/|$)' -or $relative -match '(^|/)__pycache__(/|$)' -or $file.Extension -in @('.pth','.pt','.tflite','.onnx','.pyc','.zip')) { continue }
                $entry = $archive.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
                $stream = $entry.Open()
                try { $inputStream = $file.OpenRead(); try { $inputStream.CopyTo($stream) } finally { $inputStream.Dispose() } } finally { $stream.Dispose() }
            }
        } finally { $archive.Dispose() }
    } finally { $fileStream.Dispose() }
    $modelZip = Join-Path $distDir 'GridRestore-models.zip'
    $modelStream = [System.IO.File]::Open($modelZip,[System.IO.FileMode]::Create)
    try {
        $modelArchive = [System.IO.Compression.ZipArchive]::new($modelStream,[System.IO.Compression.ZipArchiveMode]::Create)
        try {
            foreach ($modelFile in Get-ChildItem -LiteralPath (Join-Path $projectRoot 'app\src\main\assets\models') -File) {
                $entry = $modelArchive.CreateEntry($modelFile.Name,[System.IO.Compression.CompressionLevel]::Optimal)
                $entryStream = $entry.Open()
                try { $inputStream = $modelFile.OpenRead(); try { $inputStream.CopyTo($entryStream) } finally { $inputStream.Dispose() } } finally { $entryStream.Dispose() }
            }
        } finally { $modelArchive.Dispose() }
    } finally { $modelStream.Dispose() }
    Copy-Item -LiteralPath (Join-Path $projectRoot 'tools\test-data.zip') -Destination (Join-Path $distDir 'GridRestore-test-data.zip') -Force
    $hashes = @('GridRestore-debug.apk','GridRestore-source.zip','GridRestore-models.zip','GridRestore-test-data.zip') | ForEach-Object {
        $hash = Get-FileHash -LiteralPath (Join-Path $distDir $_) -Algorithm SHA256
        "$($hash.Hash.ToLowerInvariant())  $_"
    }
    $hashes | Set-Content -LiteralPath (Join-Path $distDir 'SHA256SUMS.txt') -Encoding utf8
    Write-Output "Packaged APK, source ZIP and checksums in $distDir"
    Write-Output 'Validated model hashes verified inside APK. Runtime/device acceptance status is recorded separately in docs/validation.md.'
} finally { Pop-Location }
