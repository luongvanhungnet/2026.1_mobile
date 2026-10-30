"""Export pretrained GridFormer / YOLOv8n to the Android app's strict FP32 ABI.

Run on Linux x86_64 / Colab with Python 3.11 and requirements-convert.txt.
No training, quantization, remote inference, or placeholder model is performed.
Successful files are published only after independent TFLite numerical checks.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import types
import itertools
import tempfile
import math

REFERENCE_FORWARD = {}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def package_versions() -> dict:
    names = ("torch", "litert-torch", "ai-edge-litert", "ultralytics", "einops", "numpy", "Pillow")
    return {name: importlib.metadata.version(name) for name in names}


def load_gridformer(source_dir: Path, state_key: str):
    import torch

    source = source_dir / "GridFormer_arch.py"
    checkpoint = source_dir / "net_g_best.pth"
    manifest = json.loads((source_dir / "download_manifest.json").read_text(encoding="utf-8"))
    if digest(source) != manifest["architecture_sha256"] or digest(checkpoint) != manifest["checkpoint"]["sha256"]:
        raise ValueError("Source/checkpoint hash differs from download manifest. Re-download or investigate.")
    # Bypass only BasicSR's registration side effect. Architecture math is unchanged.
    # The official YAML calls the architecture GridFormer_v15; the pinned repository
    # actually defines only GridFormer. Strict state_dict loading is mandatory below.
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "basicsr.utils.registry"
    )]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "GridFormer":
            node.decorator_list = []
    module = types.ModuleType("official_gridformer_arch")
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    model = module.GridFormer(
        in_channels=3, out_channels=3, dim=48, kernel_size=3, stride=2,
        height=3, width=6, num_blocks=[1, 2, 4], heads=[2, 4, 8], attention=True,
        windows=4, sample_rate_trunk=[4, 2, 2], scale=1,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state_key not in state:
        raise ValueError(f"Checkpoint has no requested state_dict key {state_key}; keys: {list(state)[:10]}")
    state = state[state_key]
    if state and all(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)

    # The converter does not lower depthwise ConvTranspose2d correctly. When
    # kernel=stride and padding=0, no output samples overlap: a depthwise 1x1
    # channel expansion followed by pixel shuffle is exactly the same operation.
    replacements = []
    checks = []
    class ExpandedDepthwiseUpsample(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.weight = original.weight
            self.bias = original.bias
            self.channels = original.in_channels
            self.factor = original.stride[0]
        def forward(self, x):
            s = self.factor
            weights = self.weight.reshape(self.channels*s*s, 1, 1, 1)
            out = torch.nn.functional.conv2d(x, weights, groups=self.channels)
            out = torch.nn.functional.pixel_shuffle(out, s)
            return out if self.bias is None else out+self.bias.reshape(1,-1,1,1)
    for name, layer in list(model.named_modules()):
        if isinstance(layer, torch.nn.ConvTranspose2d) and layer.groups > 1:
            s = layer.stride[0]
            if not (layer.in_channels == layer.out_channels == layer.groups and layer.kernel_size == (s,s) and layer.stride == (s,s) and layer.padding == (0,0) and layer.output_padding == (0,0) and layer.dilation == (1,1)):
                raise ValueError(f"Unrecognized grouped transposed convolution: {name}")
            replacement = ExpandedDepthwiseUpsample(layer)
            sample = torch.randn(1, layer.in_channels, 5, 7, generator=torch.Generator().manual_seed(2026))
            with torch.no_grad():
                actual, expected = replacement(sample), layer(sample)
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
            checks.append({"module":name,"max_absolute_error":float((actual-expected).abs().max())})
            parent_name, attribute = name.rsplit('.',1)
            parent = model.get_submodule(parent_name)
            replacements.append((parent, attribute, layer, replacement))
            setattr(parent, attribute, replacement)

    class RestorationNHWC(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            # Official forward() reverses [full,half,quarter]; last is full resolution.
            full = self.network(image.permute(0, 3, 1, 2))[-1]
            return full.permute(0, 2, 3, 1).contiguous()

    wrapped = RestorationNHWC(model).eval()
    def original_forward(image):
        for parent, attribute, original, replacement in replacements: setattr(parent, attribute, original)
        try: return wrapped(image)
        finally:
            for parent, attribute, original, replacement in replacements: setattr(parent, attribute, replacement)
    REFERENCE_FORWARD[id(wrapped)] = original_forward
    print(f"STRICT_LOAD_OK; equivalent depthwise upsample rewrites: {len(replacements)}", flush=True)
    return wrapped, {
        "architecture": "GridFormer (official YAML alias: GridFormer_v15)",
        "source_commit": manifest["source_commit"],
        "checkpoint_sha256": digest(checkpoint),
        "state_dict_key": state_key,
        "strict_load_passed": True,
        "conversion_rewrite": "Depthwise ConvTranspose kernel=stride -> depthwise 1x1 expansion + pixel shuffle; numerical validation reference uses original unmodified modules",
        "rewrite_local_checks": checks,
    }


def load_yolo(checkpoint: str, segmentation: bool):
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    yolo = YOLO(checkpoint)
    expected_task = "segment" if segmentation else "detect"
    if yolo.task != expected_task:
        raise ValueError(f"Checkpoint is task {yolo.task}, expected {expected_task}")
    model = yolo.model.cpu().float().eval().fuse().requires_grad_(False)
    if len(model.names) != 80:
        raise ValueError("Android decoder currently expects the COCO 80-class YOLOv8 ABI.")
    for layer in model.modules():
        if isinstance(layer, Detect):
            layer.dynamic = False
            layer.export = True
            layer.format = "tflite"  # normalized xywh, raw class probabilities
    # Materialize anchors and shape cache before torch.export tracing.
    with torch.no_grad():
        for _ in range(2):
            model(torch.zeros(1, 3, 640, 640))

    class YoloNHWC(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            result = self.network(image.permute(0, 3, 1, 2).contiguous())
            if segmentation:
                predictions, prototypes = result
                return predictions, prototypes.permute(0, 2, 3, 1).contiguous()
            return result

    resolved = Path(getattr(yolo, "ckpt_path", checkpoint))
    return YoloNHWC(model).eval(), {
        "architecture": "YOLOv8n-seg" if segmentation else "YOLOv8n",
        "checkpoint": str(resolved),
        "checkpoint_sha256": digest(resolved) if resolved.is_file() else None,
        "classes": model.names,
        "coordinate_format": "normalized_xywh",
        "nms": "external",
        "license": "Ultralytics AGPL-3.0; retain attribution and comply when redistributing",
    }


def sample_arrays(size: int, images: list[Path]):
    import numpy as np
    from PIL import Image, ImageOps

    yield "zeros", np.zeros((1, size, size, 3), dtype=np.float32)
    yield "gray", np.full((1, size, size, 3), 0.5, dtype=np.float32)
    yield "seeded_noise", np.random.default_rng(2026).random((1, size, size, 3), dtype=np.float32)
    for image_path in images:
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            if size == 256:
                # First tile, pad using replicated edge as the Android tiler does.
                pixels = np.asarray(image, dtype=np.float32) / 255.0
                pixels = pixels[:size, :size]
                pixels = np.pad(pixels, ((0, max(0, size - pixels.shape[0])), (0, max(0, size - pixels.shape[1])), (0, 0)), mode="edge")
            else:
                ratio = min(size / image.width, size / image.height)
                resized = image.resize((round(image.width * ratio), round(image.height * ratio)), Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (size, size), (114, 114, 114))
                canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
                pixels = np.asarray(canvas, dtype=np.float32) / 255.0
            yield str(image_path), pixels[None].copy()


def export_and_check(model, target: Path, inputs: list[int], outputs: list[list[int]], metadata: dict, args):
    import litert_torch
    import numpy as np
    import torch
    from ai_edge_litert.interpreter import Interpreter

    if target.exists() and args.reuse_candidate:
        report = target.with_suffix(".validation.json")
        if report.is_file() and json.loads(report.read_text())["sha256"] == digest(target):
            print(f"REUSING VALIDATED EXPORT: {target}", flush=True)
            return
    if target.exists() or target.with_suffix(".validation.json").exists():
        raise FileExistsError(f"Refusing to overwrite {target}. Use a new output directory.")
    candidate = target.with_suffix(".candidate.tflite")
    if candidate.exists() and not args.reuse_candidate:
        raise FileExistsError(f"An unverified candidate already exists: {candidate}")
    sample = torch.zeros(inputs, dtype=torch.float32)
    display_budget=args.restoration_display_budget and target.stem == "gridformer"
    if not candidate.exists():
        print(f"CONVERTING {metadata['architecture']}", flush=True)
        # v0.8.0 truncates random lowering names to 32 bits. Large graphs can
        # collide. A local monotonic namespace changes names, not operators.
        from litert_torch.odml_torch.jax_bridge import _wrap
        counter = itertools.count()
        _wrap.uuid = types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex=f"{next(counter):08x}"))
        metadata["converter_symbol_workaround"] = "Monotonic unique 32-bit lowering symbols instead of truncated random UUIDs (LiteRT Torch 0.8.0)"
        from litert_torch.lowertools import odml_torch_utils
        from litert_torch.odml_torch import export_utils
        from jaxlib.mlir import ir
        base_lower = odml_torch_utils.exported_program_to_mlir
        def lower_and_inline(*call_args, **kwargs):
            lowered=base_lower(*call_args, **kwargs)
            if target.stem == "gridformer":
                with lowered.ctx, ir.Location.unknown():
                    symbols=ir.SymbolTable(lowered.module.operation)
                    main=symbols["main"]
                    count=len(list(lowered.module.body.operations))
                    export_utils.inline(symbols,main.entry_block)
                    for operation in list(lowered.module.body.operations):
                        if operation.operation != main.operation: operation.operation.erase()
                    lowered.module.operation.verify()
                    metadata["lowering_helpers_inlined"]=count-1
                    print(f"INLINED_LOWERING_HELPERS {count-1}",flush=True)
            return lowered
        work=Path("tools/work");work.mkdir(parents=True,exist_ok=True)
        saved_model_dir=tempfile.mkdtemp(prefix=target.stem+"-saved-",dir=work)
        metadata["saved_model_cache"]=saved_model_dir
        with torch.no_grad():
            odml_torch_utils.exported_program_to_mlir=lower_and_inline
            try:
                edge_model = litert_torch.convert(model, (sample,), _saved_model_dir=saved_model_dir)
                edge_model.export(str(candidate))
                del edge_model
            finally: odml_torch_utils.exported_program_to_mlir=base_lower
    print(f"VALIDATING {candidate}", flush=True)
    with candidate.open("rb") as handle:
        header = handle.read(8)
    if header[4:8] != b"TFL3":
        raise ValueError("Converter output is not a TFLite FlatBuffer.")
    interpreter = Interpreter(model_path=str(candidate), num_threads=4)
    interpreter.allocate_tensors()
    incoming = interpreter.get_input_details()
    outgoing = interpreter.get_output_details()
    if len(incoming) != 1 or incoming[0]["shape"].tolist() != inputs or incoming[0]["dtype"] != np.float32:
        raise ValueError(f"Unexpected input ABI: {incoming}")
    actual_outputs = [item["shape"].tolist() for item in outgoing]
    if sorted(actual_outputs) != sorted(outputs) or any(item["dtype"] != np.float32 for item in outgoing):
        raise ValueError(f"Unexpected output ABI: {outgoing}")
    ordered_outgoing = [next(item for item in outgoing if item["shape"].tolist() == shape) for shape in outputs]
    if any(item["dtype"] == np.float16 for item in interpreter.get_tensor_details()):
        raise ValueError("Found FP16 tensors in requested FP32 export.")
    checks = []
    for name, array in sample_arrays(inputs[1], args.sample_image):
        with torch.no_grad():
            reference = REFERENCE_FORWARD.get(id(model), model)(torch.from_numpy(array))
        reference = reference if isinstance(reference, (tuple, list)) else (reference,)
        interpreter.set_tensor(incoming[0]["index"], array)
        interpreter.invoke()
        comparisons = []
        for expected, details in zip(reference, ordered_outgoing, strict=True):
            actual = interpreter.get_tensor(details["index"])
            baseline = expected.detach().cpu().numpy()
            if not np.isfinite(actual).all() or not np.isfinite(baseline).all():
                raise ValueError(f"Non-finite model result on {name}")
            difference=actual.astype(np.float64)-baseline.astype(np.float64)
            max_error=float(np.max(np.abs(difference)))
            mse=float(np.mean(difference*difference))
            strict_pass=bool(np.allclose(actual,baseline,atol=args.atol,rtol=args.rtol))
            details_report={"max_absolute_error":max_error,"mean_absolute_error":float(np.mean(np.abs(difference))),"mse":mse,"psnr_db":-10*math.log10(mse) if mse>0 else None,"strict_allclose_passed":strict_pass}
            if display_budget:
                if inputs != [1,256,256,3] or outputs != [[1,256,256,3]]:raise ValueError("Display error budget applies only to restoration RGB images")
                # FP32 reductions can differ across backends in this deep model.
                # Fixed app-output contract: <= half one 8-bit level raw error,
                # >=70dB PSNR, and <=one level after the app's clamp/round step.
                quantized=np.rint(np.clip(actual,0,1)*255).astype(np.int16)
                reference_quantized=np.rint(np.clip(baseline,0,1)*255).astype(np.int16)
                pixel_error=int(np.max(np.abs(quantized-reference_quantized)))
                details_report["max_8bit_channel_difference"]=pixel_error
                if max_error>0.5/255 or mse>1e-7 or pixel_error>1:
                    raise AssertionError(f"Restoration display budget exceeded on {name}: {details_report}")
            else:
                np.testing.assert_allclose(actual, baseline, atol=args.atol, rtol=args.rtol)
            comparisons.append(details_report)
        print(f"CHECKED {name}: {comparisons}",flush=True)
        checks.append({"sample": name, "outputs": comparisons})
    metadata.update({
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": inputs, "outputs": outputs, "dtype": "float32", "input_range": [0, 1],
        "actual_output_order": actual_outputs,
        "python": sys.version, "platform": platform.platform(), "packages": package_versions(),
        "numerical_checks": checks, "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "validation_policy": "RGB restoration: max raw error<=0.5/255, PSNR>=70dB, max clamped/rounded 8-bit channel difference<=1; strict allclose outcome retained" if display_budget else "strict_allclose",
        "operator_names": sorted({item["op_name"] for item in interpreter._get_ops_details()}),
        "operator_count_without_runtime_delegates": sum(item["op_name"] != "DELEGATE" for item in interpreter._get_ops_details()),
        "sha256": digest(candidate), "bytes": candidate.stat().st_size,
        "host_cpu_validation_passed": True,
        "android_validation_passed": False,
        "gpu_partition_verified": False,
        "restoration_quality_verified": False,
    })
    provenance=candidate.with_suffix(".provenance.json")
    if provenance.is_file():
        conversion=json.loads(provenance.read_text())
        if conversion["candidateSha256"] != metadata["sha256"]:raise ValueError("Candidate provenance hash mismatch")
        conversion["numericalValidationPassed"]=True
        metadata["conversionProvenance"]=conversion
    del interpreter
    candidate.replace(target)
    target.with_suffix(".validation.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"VALIDATED HOST EXPORT: {target} ({metadata['bytes']:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["all", "gridformer", "yolo"], default="all")
    parser.add_argument("--source-dir", type=Path, default=Path("models/source"))
    parser.add_argument("--state-key", choices=["params", "params_ema"], default="params", help="Official test loader defaults to params; use EMA only as a recorded experiment")
    parser.add_argument("--output-dir", type=Path, default=Path("models/exported"))
    parser.add_argument("--detect-checkpoint", default="yolov8n.pt")
    parser.add_argument("--segment-checkpoint", default="yolov8n-seg.pt")
    parser.add_argument("--sample-image", type=Path, action="append", default=[])
    parser.add_argument("--atol", type=float, default=0.0002)
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument("--reuse-candidate", action="store_true", help="Numerically validate an existing candidate against freshly loaded original weights")
    parser.add_argument("--restoration-display-budget",action="store_true",help="Use explicit 8-bit restoration output error budget; retain strict-allclose results for transparency")
    args = parser.parse_args()
    for image in args.sample_image:
        if not image.is_file(): parser.error(f"Sample image not found: {image}")
    if sys.platform != "linux":
        parser.error("litert-torch conversion requires Linux. Use WSL2, a Linux machine, or Colab. Download/checkpoint inspection works on Windows.")
    if args.atol <= 0 or args.rtol < 0:
        parser.error("Tolerances must be valid positive numerical bounds.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.kind in ("all", "gridformer"):
        model, metadata = load_gridformer(args.source_dir, args.state_key)
        export_and_check(model, args.output_dir / "gridformer.tflite", [1, 256, 256, 3], [[1, 256, 256, 3]], metadata, args)
        del model
    if args.kind in ("all", "yolo"):
        for segmentation, filename, outputs in [
            (False, "yolo_detect.tflite", [[1, 84, 8400]]),
            (True, "yolo_segment.tflite", [[1, 116, 8400], [1, 160, 160, 32]]),
        ]:
            checkpoint = args.segment_checkpoint if segmentation else args.detect_checkpoint
            model, metadata = load_yolo(checkpoint, segmentation)
            export_and_check(model, args.output_dir / filename, [1, 640, 640, 3], outputs, metadata, args)
            del model
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"EXPORT FAILED ({type(error).__name__}): {error}\nNo successful export is claimed for the failed artifact. Keep the traceback/log for diagnosis.", file=sys.stderr)
        raise
