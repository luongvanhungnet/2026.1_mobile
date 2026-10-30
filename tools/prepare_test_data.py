"""Prepare a reproducible labelled COCO val2017 weather subset (supplemental test).
No model predictions are used as ground truth; labels are original COCO polygons.
"""
import argparse, hashlib, io, json, urllib.request, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BUCKET = "https://s3.amazonaws.com/images.cocodataset.org/"
ARCHIVE = BUCKET + "annotations/annotations_trainval2017.zip"

class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            self.length = int(r.headers["Content-Length"])
        self.position = 0
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        return self.position
    def read(self, count=-1):
        count = self.length-self.position if count < 0 else min(count,self.length-self.position)
        if count <= 0: return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{self.position+count-1}"})
        with urllib.request.urlopen(req, timeout=90) as r:
            if r.status != 206: raise RuntimeError("COCO server must support HTTP ranges; refusing accidental full archive download")
            data = r.read(count)
        self.position += len(data)
        return data

def weather(image, category, seed):
    rng = np.random.default_rng(seed)
    result = image.copy()
    if category in ("haze", "rain_haze", "multi_weather"):
        array = np.asarray(result,dtype=np.float32)
        transmission = np.linspace(.42,.72,image.height,dtype=np.float32)[:,None,None]
        result=Image.fromarray(np.clip(array*transmission+225*(1-transmission),0,255).astype(np.uint8))
    if category in ("rain", "rain_haze", "multi_weather"):
        overlay=Image.new("RGB",image.size); draw=ImageDraw.Draw(overlay)
        for _ in range(140):
            x=int(rng.integers(-10,image.width));y=int(rng.integers(-15,image.height))
            draw.line((x,y,x+3,y+12),fill=(160,175,190),width=1)
        from PIL import ImageChops
        result=ImageChops.add(result,overlay.filter(ImageFilter.GaussianBlur(.35)),scale=1.12)
    if category in ("snow", "multi_weather"):
        overlay=result.copy(); draw=ImageDraw.Draw(overlay)
        for _ in range(90):
            x=int(rng.integers(0,image.width));y=int(rng.integers(0,image.height));r=int(rng.integers(1,3))
            draw.ellipse((x-r,y-r,x+r,y+r),fill=(240,245,250))
        result=Image.blend(result,overlay.filter(ImageFilter.GaussianBlur(.4)),.75)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=Path("tools/test-data"))
    p.add_argument("--count",type=int,default=3)
    p.add_argument("--max-side",type=int,default=256)
    args=p.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    cache=Path("tools/downloads/coco");cache.mkdir(parents=True,exist_ok=True)
    annotations=cache/"instances_val2017.json"
    if not annotations.exists():
        print("Downloading original COCO val2017 annotations via HTTP ranges",flush=True)
        with zipfile.ZipFile(RangeReader(ARCHIVE)) as z:
            annotations.write_bytes(z.read("annotations/instances_val2017.json"))
    data=json.loads(annotations.read_text())
    by_image={}
    for ann in data["annotations"]: by_image.setdefault(ann["image_id"],[]).append(ann)
    selected=[]
    for image in sorted(data["images"],key=lambda x:x["id"]):
        labels=by_image.get(image["id"],[])
        if 1<=len(labels)<=8 and all(not a["iscrowd"] and isinstance(a["segmentation"],list) and a["segmentation"] for a in labels): selected.append(image)
        if len(selected)==args.count:break
    category_map={c["id"]:i for i,c in enumerate(sorted(data["categories"],key=lambda x:x["id"]))}
    entries=[]; sources=[]
    (out/"images").mkdir(exist_ok=True);(out/"clean").mkdir(exist_ok=True)
    for image in selected:
        url=BUCKET+"val2017/"+image["file_name"]
        source=cache/image["file_name"]
        if not source.exists():urllib.request.urlretrieve(url,source)
        clean=Image.open(source).convert("RGB")
        ratio=min(1.,args.max_side/max(clean.size)); size=(round(clean.width*ratio),round(clean.height*ratio))
        clean=clean.resize(size,Image.Resampling.LANCZOS);clean.save(out/"clean"/(str(image["id"])+".png"))
        sx=size[0]/image["width"];sy=size[1]/image["height"]
        labels=[]
        for ann in by_image[image["id"]]:
            x,y,w,h=ann["bbox"]
            polygons=[[[float(poly[i]*sx),float(poly[i+1]*sy)] for i in range(0,len(poly),2)] for poly in ann["segmentation"]]
            labels.append({"classId":category_map[ann["category_id"]],"box":[max(0,x*sx),max(0,y*sy),min(size[0],(x+w)*sx),min(size[1],(y+h)*sy)],"polygons":polygons,"cocoAnnotationId":ann["id"]})
        for category in ["rain","haze","rain_haze","snow","multi_weather"]:
            name=f"{category}_{image['id']}"; file=f"images/{name}.png"
            weather(clean,category,2026+image["id"]).save(out/file)
            entries.append({"id":name,"degradation":category,"file":file,"annotations":labels,"cleanFile":f"clean/{image['id']}.png"})
        sources.append({"cocoImageId":image["id"],"url":url,"sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"originalSize":[image["width"],image["height"]],"testSize":list(size)})
    provenance={"dataset":"COCO val2017 (held-out YOLO validation images)","sourceAnnotations":ARCHIVE,"annotationSha256":hashlib.sha256(annotations.read_bytes()).hexdigest(),"groundTruth":"Original COCO instance polygons and boxes; contiguous COCO-80 class mapping; no pseudo labels","degradation":"Deterministic synthetic weather; source code and seed recorded","seed":2026,"scope":"Five GridFormer weather tasks selected by user","supplemental":True,"originalGridFormerTestSplit":False,"limitations":"Small supplemental subset, not a replacement for the paper's restoration benchmark or evidence of generalization.","images":sources,"imageLicense":"COCO images have individual Flickr licenses; original image records retained in source-subset.json"}
    manifest={"schemaVersion":1,"name":"COCO val2017 · 5 weather tasks · supplemental", "provenance":provenance,"images":entries}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (out/"source-subset.json").write_text(json.dumps({"images":selected,"licenses":data.get("licenses",[]),"categories":data["categories"],"annotations":[a for image in selected for a in by_image[image["id"]]]}))
    target=out.with_suffix(".zip")
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for file in out.rglob("*"):
            if file.is_file():z.write(file,file.relative_to(out).as_posix())
    print(json.dumps({"zip":str(target),"images":len(entries),"sourceImages":len(selected),"sha256":hashlib.sha256(target.read_bytes()).hexdigest()}),flush=True)

if __name__=="__main__":main()
