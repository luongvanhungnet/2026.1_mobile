"""Finish a cached/inlined SavedModel conversion; output remains a candidate."""
import argparse, hashlib, json
from pathlib import Path
import tensorflow as tf

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("source",type=Path);p.add_argument("candidate",type=Path)
args=p.parse_args()
if args.candidate.exists():raise FileExistsError(args.candidate)
converter=tf.lite.TFLiteConverter.from_saved_model(str(args.source))
converter._experimental_enable_composite_direct_lowering=True
converter._experimental_enable_dynamic_update_slice=True
converter.model_origin_framework="PYTORCH"
converter.optimizations=[]
converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
blob=converter.convert()
args.candidate.write_bytes(blob)
args.candidate.with_suffix(".provenance.json").write_text(json.dumps({"sourceSavedModelSha256":hashlib.sha256((args.source/"saved_model.pb").read_bytes()).hexdigest(),"candidateSha256":hashlib.sha256(blob).hexdigest(),"inlining":json.loads((args.source/"inlining.json").read_text()) if (args.source/"inlining.json").is_file() else None,"numericalValidationPassed":False},indent=2))
print(f"CANDIDATE ONLY: {args.candidate} ({len(blob)} bytes); run export_models.py --reuse-candidate for original-PyTorch validation",flush=True)
