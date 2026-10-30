"""Capture task-owned 1080x1920 emulator UI only; never targets the phone."""
import subprocess,time
from pathlib import Path

serial="emulator-5580"
out=Path("dist/validation/emulator-5580");out.mkdir(parents=True,exist_ok=True)
def adb(*args):return subprocess.check_output(["adb","-s",serial,*args])
def capture(name):(out/name).write_bytes(adb("exec-out","screencap","-p"))
adb("shell","am","start","-W","-n","vn.edu.gridrestore/.MainActivity")
recorder=subprocess.Popen(["adb","-s",serial,"shell","screenrecord","--time-limit","22","/sdcard/gridrestore-walkthrough.mp4"],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
time.sleep(1)
capture("10-restoration.png")
adb("shell","input","swipe","800","1100","230","1100","900");time.sleep(.4)
capture("11-comparison-slider.png")
adb("shell","input","swipe","230","1100","800","1100","900")
for x,name in [(410,"12-evaluation.png"),(665,"14-benchmark-pending.png"),(900,"15-models-bundled.png")]:
 adb("shell","input","tap",str(x),"300");time.sleep(.7);capture(name)
 if x==410:
  adb("shell","input","swipe","950","1670","950","950","500");time.sleep(.5);capture("13-evaluation-results.png")
adb("shell","input","tap","170","300")
recorder.wait(timeout=35)
if recorder.returncode:raise RuntimeError(recorder.stderr.read().decode(errors="replace"))
adb("pull","/sdcard/gridrestore-walkthrough.mp4",str(out/"walkthrough.mp4"))
print("Captured final emulator screenshots and silent app walkthrough")
