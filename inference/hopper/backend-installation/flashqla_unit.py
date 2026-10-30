"""C1 GDN-shape compatibility and latency gate; not a model/A-score benchmark."""
import argparse,json,math,time,traceback
from pathlib import Path
import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as reference
from flash_qla import chunk_gated_delta_rule as candidate

p=argparse.ArgumentParser();p.add_argument('--auto-cp',type=int,default=0);p.add_argument('--small-only',action='store_true');a=p.parse_args()
ROOT=Path(__file__).resolve().parent
report=dict(scope='GDN_core_unit_not_model_score',torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),auto_cp=bool(a.auto_cp),cases=[])
destination=ROOT/f'unit-auto-cp-{a.auto_cp}.json'
torch.manual_seed(20261031)

def difference(actual,expected):
    d=actual.float()-expected.float()
    rms=d.square().mean().sqrt().item()
    baseline=expected.float().square().mean().sqrt().item()
    return dict(max_abs=d.abs().max().item(),rms=rms,reference_rms=baseline,relative_rms=rms/max(baseline,1.e-12))

def measure(fn):
    for _ in range(3):z=fn();del z
    torch.cuda.synchronize();times=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();z=fn();end.record();torch.cuda.synchronize()
        times.append(start.elapsed_time(end));del z
    return times

try:
    with torch.inference_mode():
        for b,t,h0_enabled in ([(1,128,False)] if a.small_only else [(1,128,False),(1,4096,False),(1,4096,True),(64,4096,False)]):
            q=torch.randn(b,t,6,64,device='cuda',dtype=torch.bfloat16)
            k=torch.randn_like(q);v=torch.randn(b,t,18,64,device='cuda',dtype=torch.bfloat16)
            g=-(.001+.1*torch.rand(b,t,18,device='cuda',dtype=torch.float32))
            beta=torch.randn(b,t,18,device='cuda',dtype=torch.float32).sigmoid().to(torch.bfloat16)
            h0=torch.randn(b,18,64,64,device='cuda',dtype=torch.float32)*.1 if h0_enabled else None
            # Explicitly share the original FLA normalization rather than conflating core and norm changes.
            q,_=l2norm_fwd(q);k,_=l2norm_fwd(k)
            kwargs=dict(q=q,k=k,v=v,g=g,beta=beta,scale=.125,initial_state=h0,
                        output_final_state=True,use_qk_l2norm_in_kernel=False,state_v_first=False)
            print('UNIT_START',b,t,h0_enabled,'AUTO_CP',a.auto_cp,flush=True)
            expected,expected_state=reference(**kwargs)
            actual,actual_state=candidate(**kwargs,auto_cp=bool(a.auto_cp))
            assert actual.shape==expected.shape and actual.dtype==expected.dtype
            assert actual_state.shape==expected_state.shape
            out_error=difference(actual,expected);state_error=difference(actual_state,expected_state)
            assert torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
            assert out_error['relative_rms']<.02,out_error
            assert state_error['relative_rms']<.02,state_error
            record=dict(batch=b,sequence=t,qk_heads=6,v_heads=18,head_dimension=64,
                        scale=.125,normalization='shared_original_FLA_l2norm_then_both_flag_False',
                        g_dtype=str(g.dtype),beta_dtype=str(beta.dtype),initial_state=h0_enabled,
                        final_state_dtype=str(actual_state.dtype),output_error=out_error,state_error=state_error)
            if not h0_enabled:
                perf_args=dict(kwargs,output_final_state=False)
                record['reference_ms']=measure(lambda:reference(**perf_args))
                record['candidate_ms']=measure(lambda:candidate(**perf_args,auto_cp=bool(a.auto_cp)))
                record['speedup_median']=sorted(record['reference_ms'])[2]/sorted(record['candidate_ms'])[2]
            report['cases'].append(record);report['status']='running'
            destination.write_text(json.dumps(report,indent=2)+'\n')
            print('UNIT_RESULT',json.dumps(record),flush=True)
            del q,k,v,g,beta,h0,actual,actual_state,expected,expected_state,kwargs
    report['status']='ok'
except Exception as error:
    report.update(status='error',error_type=type(error).__name__,error=str(error));traceback.print_exc()
destination.write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['status']=='ok' else 1)
