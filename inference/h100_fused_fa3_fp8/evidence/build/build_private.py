import os, pathlib, subprocess, json, time
P=pathlib.Path('/data/q38/experiments/c4-lowprecision-20261031/fp8-attention')
R=P/'round2'
py='/data/q38/environments/q38-flashqla-torch280-cu128-py312/bin/python'
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='1', CUDA_HOME=str(R/'cuda-12.8'), FLASH_ATTENTION_FORCE_BUILD='TRUE', FLASH_ATTENTION_OFFLINE_BUILD='TRUE', BUILD_TARGET='cuda', MAX_JOBS='4', NVCC_THREADS='2')
for feature in ['BACKWARD','SPLIT','PAGEDKV','APPENDKV','LOCAL','SOFTCAP','FP16','VARLEN','CLUSTER','HDIM96','HDIM128','HDIM192','HDIM256','SM80','HDIMDIFF64','HDIMDIFF192']:
    env['FLASH_ATTENTION_DISABLE_'+feature]='TRUE'
env['PATH']=str(R/'deps/bin')+':'+str(R/'cuda-12.8/bin')+':'+env['PATH']
env['PYTHONPATH']=str(R/'deps')+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
nvidia=pathlib.Path('/data/q38/environments/q38-dense-c1-torch280-cu128-py312/lib/python3.12/site-packages/nvidia')
includes=['/data/q38/environments/python312-development-headers/usr/include/python3.12']+[str(x) for x in sorted(nvidia.glob('*/include'))]
env['CPATH']=':'.join(includes)
env['CPLUS_INCLUDE_PATH']=env['CPATH']
env['FLASH_ATTENTION_HOME']=str(R/'build-cache')
cwd=R/'src/flash-attention-4616f5116d33e3ba3cbb71d582caffcfcb49649c/hopper'
cmd=[py,'setup.py','build_ext','--inplace']
with open(R/'guard.log','w') as f:
    guard=subprocess.run([py,'-c',"import torch; from torch.utils.cpp_extension import _check_cuda_version; from torch.torch_version import TorchVersion; _check_cuda_version('g++',TorchVersion('13.3.0')); print('NORMAL_CUDA_VERSION_GUARD_PASSED',torch.version.cuda)"],env=env,stdout=f,stderr=subprocess.STDOUT)
assert guard.returncode==0, 'normal CUDA version guard failed'
with open(R/'build.log','w') as f:
    child=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
    (P/'status.json').write_text(json.dumps({'phase':'round2_build_fa3','candidate':'FA3 Hopper FP8','running':True,'pid':child.pid,'gpu':1,'updated_at':time.time()},indent=2))
    code=child.wait()
(R/'build_metadata.json').write_text(json.dumps({'command':cmd,'env':{k:v for k,v in env.items() if k.startswith('FLASH_') or k in ['CUDA_HOME','CUDA_VISIBLE_DEVICES','MAX_JOBS','NVCC_THREADS','CPATH','CPLUS_INCLUDE_PATH']},'returncode':code,'updated_at':time.time()},indent=2))
(P/'status.json').write_text(json.dumps({'phase':'round2_fa3_built' if code==0 else 'round2_fa3_build_failed','candidate':'FA3 Hopper FP8','running':False,'pid':None,'gpu':1,'returncode':code,'updated_at':time.time()},indent=2))
raise SystemExit(code)
