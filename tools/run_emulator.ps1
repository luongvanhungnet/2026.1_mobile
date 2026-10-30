$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sdkRoot = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$imageDir = Join-Path $sdkRoot 'system-images\android-30\default\x86_64'
if (-not (Test-Path -LiteralPath (Join-Path $imageDir 'system.img'))) { throw 'Install Android 30 AOSP x86_64 system image first.' }
$avdRoot = Join-Path $projectRoot 'tools\work\avd'
$avdDir = Join-Path $avdRoot 'GridRestore30.avd'
New-Item -ItemType Directory -Path $avdDir -Force | Out-Null
@("avd.ini.encoding=UTF-8", "path=$avdDir", 'target=android-30') | Set-Content -LiteralPath (Join-Path $avdRoot 'GridRestore30.ini') -Encoding ascii
@('AvdId=GridRestore30','avd.ini.displayname=GridRestore Android 30','abi.type=x86_64','hw.cpu.arch=x86_64',"image.sysdir.1=$imageDir\",'tag.id=default','tag.display=Default','hw.ramSize=2048','vm.heapSize=512','hw.lcd.width=1080','hw.lcd.height=1920','hw.lcd.density=420','hw.gpu.enabled=yes','hw.gpu.mode=swiftshader_indirect','disk.dataPartition.size=4G','hw.keyboard=yes','showDeviceFrame=no','fastboot.forceColdBoot=yes','PlayStore.enabled=false') | Set-Content -LiteralPath (Join-Path $avdDir 'config.ini') -Encoding ascii
$env:ANDROID_AVD_HOME = $avdRoot
$env:ANDROID_HOME = $sdkRoot
$arguments = @('-avd','GridRestore30','-no-window','-no-audio','-no-snapshot','-no-boot-anim','-gpu','swiftshader_indirect','-memory','2048','-cores','2','-port','5580')
$process = Start-Process -FilePath (Join-Path $sdkRoot 'emulator\emulator.exe') -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot 'tools\work\emulator.log') -RedirectStandardError (Join-Path $projectRoot 'tools\work\emulator-error.log')
Write-Output "Started task-owned emulator PID $($process.Id), ADB serial emulator-5580"
