"""Fetch only GridRestore's own evidence from an explicitly selected ADB target."""
import argparse, io, json, re, subprocess, tarfile
from pathlib import Path, PurePosixPath

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--serial",required=True);p.add_argument("--output",type=Path,required=True)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
# Windows adb shell can translate line endings even with -T. exec-out is
# binary-safe, but remote failures must be recognized from content as well.
base=["adb","-s",args.serial,"exec-out","run-as","vn.edu.gridrestore"]
for name in ["evaluation.json","benchmark.json","device-validation.json","edge-size-validation.json","yolo-device-validation.json","native-probe-validation.json","device-progress.txt","input.png","restored.png"]:
    result=subprocess.run(base+["cat","files/"+name],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    valid=result.returncode==0 and not result.stdout.startswith(b"cat: ")
    if valid and name.endswith(".json"):
        try:json.loads(result.stdout)
        except ValueError:valid=False
    if valid and name.endswith(".png"):valid=result.stdout.startswith(b"\x89PNG\r\n\x1a\n")
    if valid:(args.output/name).write_bytes(result.stdout)
    else:
        (args.output/name).unlink(missing_ok=True)
        print(f"Not available: {name}")
archive=subprocess.run(base+["tar","-cf","-","files/evaluation-images"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if archive.returncode==0 and not archive.stdout.startswith(b"tar:"):
    destination=args.output/"evaluation-images";destination.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        for item in tar.getmembers():
            if not item.isfile():continue
            path=PurePosixPath(item.name)
            if path.is_absolute() or ".." in path.parts or not re.fullmatch(r"sample_\d+\.png",path.name):raise ValueError(item.name)
            image=tar.extractfile(item).read()
            if not image.startswith(b"\x89PNG\r\n\x1a\n"):raise ValueError("Invalid PNG in evidence archive: "+item.name)
            (destination/path.name).write_bytes(image)
if (args.output/"evaluation.json").exists():
    report=json.loads((args.output/"evaluation.json").read_bytes())
    summary={"device":report["device"],"images":report["imageCount"],"results":[{"degradation":r["degradation"],"detectionBefore":r["detection"]["before"]["mAP50_95"],"detectionAfter":r["detection"]["after"]["mAP50_95"],"segmentationBefore":r["segmentation"]["before"]["mAP50_95"],"segmentationAfter":r["segmentation"]["after"]["mAP50_95"]} for r in report["results"]]}
    print(json.dumps(summary,indent=2))
