"""Reproduce the qualified local FlashQLA 0.1.3 SM90 D64 specialization."""
import argparse,hashlib,importlib.util,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--package',type=Path);p.add_argument('--record',type=Path);a=p.parse_args()
if a.package is None:
    spec=importlib.util.find_spec('flash_qla')
    assert spec and spec.origin,'Install flash-qla==0.1.3 in the isolated interpreter first'
    package=Path(spec.origin).parent
else:package=a.package
hopper=package/'ops/gated_delta_rule/chunk/hopper'
expected={
    'kkt_solve.py':('2b6a0ad49194881f5e3eda7812ea3ec07b42f41f3745e4239dbe767598dc92be','8a3678ee34810d1a88728fc02f368689b1ea0596a4b00a43de36fa838d3f89cc'),
    'fused_fwd.py':('9a129fc89d22264fdacfcaf1e616af78e9b33bd2f91c7ca1a96cba12be7992d2','6a0e5d35f4b0ad57ada1b9d302f18fd731b6b5efc641bd1f0134f1f40ca7bfa4'),
}
records=[]
for name,(before,after) in expected.items():
    path=hopper/name;source=path.read_text();digest=hashlib.sha256(source.encode()).hexdigest()
    assert digest in (before,after),'Unexpected upstream source; refuse fuzzy patch: '+str(path)
    if digest==before:
        if name=='kkt_solve.py':
            assert source.count('    assert K == 128')==1
            updated=source.replace('    assert K == 128','    assert K in (64, 128)')
        else:
            assert source.count('    assert K == V == 128')==1
            updated=source.replace('    assert K == V == 128','    assert K == V and K in (64, 128)')
            marker='    tilelang_fused_chunk_gdr_fwd_kernel = tilelang_fused_chunk_gdr_fwd('
            assert updated.count(marker)==1
            updated=updated.replace(marker,'    block_DV = min(block_DV, V)\n\n'+marker)
        assert hashlib.sha256(updated.encode()).hexdigest()==after
        compile(updated,str(path),'exec');path.write_text(updated)
    records.append(dict(file=str(path),upstream_sha256=before,patched_sha256=after))
report=dict(package='flash-qla==0.1.3',scope='local_SM90_D64_fixed_length_inference_auto_CP_disabled',files=records,
            note='This only reproduces source changes. Run the numerical/memcheck/model gates on the target system.')
if a.record:a.record.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
