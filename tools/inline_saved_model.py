"""Inline StableHLO helper calls in a converter-created SavedModel cache.

Large graphs otherwise spend excessive time renaming thousands of private
functions in TF's XlaCallModule importer. This changes IR structure only;
the final TFLite must still pass original-PyTorch numerical validation.
"""
from pathlib import Path
import argparse, ast, hashlib, json, shutil
from typing import Sequence, cast
import jax._src.interpreters.mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func, stablehlo

MAGIC=b"ML\xefR"

def varint(data, position):
    value=0;shift=0
    while True:
        byte=data[position];position+=1;value|=(byte&127)<<shift
        if byte<128:return value,position
        shift+=7
        if shift>63:raise ValueError("Invalid protobuf varint")

def encode(value):
    result=bytearray()
    while value>127:result.append((value&127)|128);value>>=7
    result.append(value);return bytes(result)

def inline_module(blob):
    context=jax._src.interpreters.mlir.make_ir_context()
    context.allow_unregistered_dialects=True
    # Reuse the pinned converter's Apache-2.0 inliner without importing its
    # TensorFlow/PyTorch packages into this small preprocessing process.
    import sys
    source=Path(sys.prefix)/"lib/python3.11/site-packages/litert_torch/odml_torch/export_utils.py"
    tree=ast.parse(source.read_text())
    selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("inline","clone_func_body_ops")],type_ignores=[])
    namespace={"ir":ir,"func":func,"Sequence":Sequence,"cast":cast}
    exec(compile(selected,str(source),"exec"),namespace)
    with context,ir.Location.unknown():
        module=stablehlo.deserialize_portable_artifact(context,blob)
        symbols=ir.SymbolTable(module.operation)
        main=symbols["main"]
        count=len(list(module.body.operations))
        namespace["inline"](symbols,main.entry_block)
        for operation in list(module.body.operations):
            if operation.operation!=main.operation:operation.operation.erase()
        module.operation.verify()
        result=stablehlo.serialize_portable_artifact(module,stablehlo.get_current_version())
        print(json.dumps({"functionsBefore":count,"functionsAfter":1,"irVerified":True}),flush=True)
        return result

def rewrite(data):
    if data.startswith(MAGIC):return inline_module(data),1
    if MAGIC not in data:return data,0
    result=bytearray();position=0;changed=0
    while position<len(data):
        start=position;tag,position=varint(data,position);wire=tag&7
        if wire==2:
            length,position=varint(data,position);end=position+length
            if end>len(data):raise ValueError("Truncated protobuf")
            payload,count=rewrite(data[position:end]);changed+=count
            result.extend(encode(tag)+encode(len(payload))+payload);position=end
        else:
            if wire==0:_,position=varint(data,position)
            elif wire==1:position+=8
            elif wire==5:position+=4
            else:raise ValueError(f"Unsupported protobuf wire {wire}")
            result.extend(data[start:position])
    return bytes(result),changed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path);p.add_argument("destination",type=Path)
    args=p.parse_args()
    if args.destination.exists():raise FileExistsError(args.destination)
    original=(args.source/"saved_model.pb").read_bytes()
    result,count=rewrite(original)
    if count<1:raise ValueError("No StableHLO module found")
    args.destination.mkdir(parents=True)
    shutil.copytree(args.source/"variables",args.destination/"variables")
    shutil.copytree(args.source/"assets",args.destination/"assets")
    (args.destination/"saved_model.pb").write_bytes(result)
    (args.destination/"inlining.json").write_text(json.dumps({"sourceSha256":hashlib.sha256(original).hexdigest(),"inlinedSha256":hashlib.sha256(result).hexdigest(),"modules":count,"numericallyValidated":False},indent=2))

if __name__=="__main__":main()
