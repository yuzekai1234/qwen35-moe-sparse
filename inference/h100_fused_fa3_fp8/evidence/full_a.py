"""Unchanged frozen gzy scoring loop with existing FA3 SDPA callback and verified cache."""
import argparse,os,sys,json,hashlib,time
from common import R,F,E,MODEL,configure,verify_sources,code_hashes
ap=argparse.ArgumentParser();ap.add_argument('--batch',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args();from pathlib import Path
out=Path(a.out);out.mkdir(exist_ok=False);configure('fa3-full-a');os.environ['Q38_COMPILE_EVIDENCE']=str(out/'compile-evidence')
import fa3_overlay
fa3_overlay.install()
import eval as evaluation
cache=R.parent/'full-a-cache'
def pack(dataset,tokenizer,dest):
 from benchmark_identity import tokenizer_identity
 proof=json.loads((cache/'identity.json').read_text());assert proof['dataset_sha256']==hashlib.file_digest(dataset.open('rb'),'sha256').hexdigest()
 reference=Path('/data/q38/experiments/dense-c1-inference-h100-20261030/c1-infra-fullcompile-20261031/release/submission/models/FLOPs_3e17')
 assert tokenizer_identity(MODEL)==tokenizer_identity(reference)
 for name,item in proof['files'].items():
  p=cache/name;assert p.stat().st_size==item['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==item['sha256'];os.link(p,dest/name)
 assert proof['metadata']['total_tokens']==120142976 and proof['metadata']['total_bytes']==445910984
 (out/'cache-proof.json').write_text(json.dumps(proof,indent=2));return proof['metadata']
evaluation.pack_jsonl=pack
import runtime
original=runtime.build_forward
def build(model):
 f,paths=original(model);info=paths['full_compile'];assert info['trunk_chunk']==256 and info['deferred_scale'] and info['gdn']['merged_projection'] and info['gdn']['attention']['merged_qkv_modules']==5
 info['external_ops']=['q38fp8::fa3_fused_baseline_native' if x=='q38infra::cudnn_attention' else x for x in info['external_ops']]
 paths['fp8_attention_overlay']={'backend':'official_FA3_SM90_e4m3','head_dim':64,'Q_heads':44,'KV_heads':2,'pack_gqa':True,'all_quant_layout_in_forward':True,'quant':'existing per_batch_per_KV_group amax/cast in FX','output_dtype':'BF16','compiled_attention_preserved':True,'merged_projection_preserved':True,'deferred_scale_preserved':True,'trunk_chunk':256,'output_head':'unchanged native _scaled_mm.out'}
 return f,paths
runtime.build_forward=build
(out/'launch.json').write_text(json.dumps({'baseline':'c4-fused-trunk256-b938-20261031','baseline_commit':'c5fd8e8bbd8dd17e0ad29397581984e1731dc887','physical_gpu':1,'batch':a.batch,'frozen_sources':verify_sources(),'overlay_sha256':code_hashes(),'started_at':time.time()},indent=2))
sys.argv=['eval.py','--dataset-path','/data/q38/datasets/evaluation-a-v1/dolma3_mix_100M_a.jsonl','--model-path',str(MODEL),'--batch-size',str(a.batch),'--sequence-length','4096','--output',str(out/'result.json')]
evaluation.main()
