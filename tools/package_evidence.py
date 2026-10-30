"""Preserve the measured evidence and hash the final local deliverables."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile,xml.etree.ElementTree as ET

root=Path('.')
dist=root/'dist';validation=dist/'validation';validation.mkdir(parents=True,exist_ok=True)
emulator=validation/'emulator-5580'
assert json.loads((emulator/'device-validation.json').read_text())['status']=='passed'
assert 'OK (1 test)' in (root/'tools/work/emulator-full-workflow.log').read_text()
assert 'OK (2 tests)' in (root/'tools/work/emulator-final-probe-test.log').read_text()
models=validation/'models';models.mkdir(exist_ok=True)
for file in (root/'tools/exports').glob('*.validation.json'):shutil.copy2(file,models/file.name)
shutil.copy2(root/'tools/requirements-convert-lock.txt',models/'conversion-environment.txt')
shutil.copy2(root/'tools/download_verification.json',models/'checkpoint-download-verification.json')
logs=validation/'logs';logs.mkdir(exist_ok=True)
for name in ['gridformer-export.log','gridformer-export-rewrite.log','gridformer-inlined-tflite.log','gridformer-numerical-validation.log','gridformer-display-validation.log','yolo-export-validated.log','emulator-full-workflow.log','emulator-final-probe-test.log']:
    source=root/'tools/work'/name
    if source.exists():shutil.copy2(source,logs/name)
shutil.copy2(root/'app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk',dist/'GridRestore-androidTest.apk')
tests=[ET.parse(f).getroot() for f in (root/'app/build/test-results/testDebugUnitTest').glob('TEST-*.xml')]
summary={'generatedAtUtc':datetime.now(timezone.utc).isoformat(),'build':'passed','unitTests':sum(int(t.attrib['tests']) for t in tests),'unitFailures':sum(int(t.attrib['failures']) for t in tests),'unitErrors':sum(int(t.attrib['errors']) for t in tests),'androidFunctionalTests':3,'androidTarget':'Android 11 AOSP x86_64 emulator','androidRealModelEvaluation':True,'evaluationImages':15,'supplementalDataset':True,'originalGridFormerTestSplit':False,'modelConversionValidated':True,'emulatorUiVerified':True,'physicalDeviceMeasurement':'deferred_by_user','physicalCpuGpuBenchmarkMeasured':False,'gpuFullMixedOnPhoneVerified':False,'gpuEmulator':'unavailable_OpenCL_EGL_failure_recorded','benchmarkTimingPublished':False,'apkSignatureEvidence':'apk-signature.txt','artifacts':{}}
for name in ['GridRestore-debug.apk','GridRestore-source.zip','GridRestore-models.zip','GridRestore-test-data.zip','GridRestore-androidTest.apk']:
    p=dist/name;digest=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    summary['artifacts'][name]={'bytes':p.stat().st_size,'sha256':digest.hexdigest()}
(validation/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
evidence=dist/'GridRestore-evidence.zip'
with zipfile.ZipFile(evidence,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for folder in [validation,root/'docs']:
        for p in folder.rglob('*'):
            if p.is_file():z.write(p,p.as_posix())
    z.write(root/'README.md','README.md')
checks=[]
for name,info in summary['artifacts'].items():checks.append(f"{info['sha256']}  {name}")
checks.append(f"{hashlib.sha256(evidence.read_bytes()).hexdigest()}  {evidence.name}")
(dist/'SHA256SUMS.txt').write_text('\n'.join(checks)+'\n')
print(json.dumps({'unitTests':summary['unitTests'],'androidFunctionalTests':3,'artifacts':list(summary['artifacts'])+[evidence.name],'physicalMeasurement':'deferred_by_user'},indent=2))
