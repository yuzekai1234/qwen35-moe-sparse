"""Portable version/source/optional installed-RECORD guard; never initializes CUDA."""
import argparse,ast,base64,hashlib,importlib.metadata as md,importlib.util,json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('flashqla-source-manifest.json'))
p.add_argument('--output',type=Path)
p.add_argument('--verify-records',action='store_true',help='Full install-time check, not needed on every inference call')
a=p.parse_args()
expected={'flash-qla':'0.1.3','tilelang':'0.1.12','apache-tvm-ffi':'0.1.11',
          'torch-c-dlpack-ext':'0.1.5','cloudpickle':'3.1.2','ml-dtypes':'0.6.0','z3-solver':'4.15.4.0'}
patched={
 'flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py':('2b6a0ad49194881f5e3eda7812ea3ec07b42f41f3745e4239dbe767598dc92be','8a3678ee34810d1a88728fc02f368689b1ea0596a4b00a43de36fa838d3f89cc'),
 'flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py':('9a129fc89d22264fdacfcaf1e616af78e9b33bd2f91c7ca1a96cba12be7992d2','6a0e5d35f4b0ad57ada1b9d302f18fd731b6b5efc641bd1f0134f1f40ca7bfa4'),
}
versions={name:md.version(name) for name in [*expected,'torch','triton']}
for name,version in expected.items():assert versions[name]==version,(name,versions[name],version)
assert versions['torch'].split('+')[0]=='2.8.0' and versions['triton']=='3.4.0',versions
torch_spec=importlib.util.find_spec('torch');assert torch_spec and torch_spec.origin
torch_root=Path(torch_spec.origin).parent
constants={}
for node in ast.parse((torch_root/'version.py').read_text()).body:
    if isinstance(node,ast.Assign):targets=node.targets;value=node.value
    elif isinstance(node,ast.AnnAssign):targets=[node.target];value=node.value
    else:continue
    for target in targets:
        if isinstance(target,ast.Name) and target.id in ('__version__','cuda'):
            constants[target.id]=ast.literal_eval(value)
assert constants=={'__version__':'2.8.0+cu128','cuda':'12.8'},constants
spec=importlib.util.find_spec('flash_qla');assert spec and spec.origin
package=Path(spec.origin).parent
manifest=json.loads(a.manifest.read_text())
actual={str(f.relative_to(package)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(package.rglob('*.py'))}
assert actual==manifest,'FlashQLA source manifest mismatch'
report=dict(status='ok',versions=versions,torch_runtime=constants,flashqla_source_files=len(actual),
            flashqla_source_manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),record_checks={})
if a.verify_records:
    for name in expected:
        distribution=md.distribution(name);checks={};intentional=[]
        for entry in distribution.files or []:
            if entry.hash is None:continue
            file=Path(distribution.locate_file(entry));raw=file.read_bytes()
            digest=hashlib.new(entry.hash.mode,raw).digest()
            encoded=base64.urlsafe_b64encode(digest).decode().rstrip('=')
            relative=str(entry).replace('\\','/')
            if encoded!=entry.hash.value:
                assert name=='flash-qla' and relative in patched,(name,relative)
                upstream,current=patched[relative]
                assert hashlib.sha256(raw).hexdigest()==current
                expected_record=base64.urlsafe_b64encode(bytes.fromhex(upstream)).decode().rstrip('=')
                assert entry.hash.mode=='sha256' and entry.hash.value==expected_record
                intentional.append(relative)
            checks[relative]=hashlib.sha256(raw).hexdigest()
        report['record_checks'][name]=dict(files=len(checks),intentional_D64_patches=intentional,
            installed_file_hash_manifest_sha256=hashlib.sha256(json.dumps(checks,sort_keys=True).encode()).hexdigest(),
            file_hashes=checks)
if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='record_checks'},indent=2))
print('Verified installed RECORD files:',{k:v['files'] for k,v in report['record_checks'].items()})
