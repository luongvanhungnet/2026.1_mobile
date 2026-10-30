"""Compare the actual Android-restored PNG against host LiteRT on identical pixels."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

evidence=Path("dist/validation/emulator-5580")
source=np.asarray(Image.open(evidence/"input.png").convert("RGB"),dtype=np.float32)/255
height,width=source.shape[:2]
assert width<=256 and height<=256,"This check covers the single-tile smoke sample"
tile=np.pad(source,((0,256-height),(0,256-width),(0,0)),mode="edge")[None]
model=Path("tools/exports/gridformer.tflite")
runner=Interpreter(model_path=str(model),num_threads=4);runner.allocate_tensors()
runner.set_tensor(runner.get_input_details()[0]["index"],tile)
runner.invoke()
result=runner.get_tensor(runner.get_output_details()[0]["index"])[0,:height,:width]
reference=np.rint(np.clip(result,0,1)*255).astype(np.int16)
android=np.asarray(Image.open(evidence/"restored.png").convert("RGB"),dtype=np.int16)
assert android.shape==reference.shape
difference=np.abs(android-reference)
report={"modelSha256":hashlib.sha256(model.read_bytes()).hexdigest(),"inputPngSha256":hashlib.sha256((evidence/"input.png").read_bytes()).hexdigest(),"width":width,"height":height,"max8bitChannelDifference":int(difference.max()),"mean8bitChannelDifference":float(difference.mean()),"exactChannelFraction":float(np.mean(difference==0)),"policy":"max difference <=1 level for the 8-bit app output","passed":bool(difference.max()<=1),"androidDevice":"Android 11 AOSP x86_64 emulator; LiteRT 1.4.2","hostRuntime":"ai-edge-litert 2.1.0"}
(evidence/"host-android-comparison.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert report["passed"],"Android and host restoration pixels differ beyond the display budget"
