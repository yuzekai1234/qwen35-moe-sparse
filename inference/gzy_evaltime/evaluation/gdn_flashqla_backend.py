"""Opt-in FlashQLA SM90 D64 core, retaining original FLA Q/K normalization."""
import inspect
import importlib.metadata
import types
import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as reference
from fla.ops.utils.cumsum import chunk_local_cumsum as fla_chunk_local_cumsum
from flash_qla import chunk_gated_delta_rule as flashqla
import flash_qla.ops.gated_delta_rule.chunk as flashqla_chunk

EVIDENCE={'calls':0,'backend':'flash-qla','version':importlib.metadata.version('flash-qla'),
          'local_patch':'SM90_D64_guards_and_block_DV_cap','auto_cp':False,
          'odd_head_cumsum':'fla_triton'}
_tilelang_chunk_local_cumsum=flashqla_chunk.chunk_local_cumsum


def _chunk_local_cumsum(g,chunk_size=64,cu_seqlens=None,**kwargs):
    """TileLang's gate cumsum finds no register layout for an odd head count (C2 27, C3 33).
    FLA's Triton cumsum is bitwise equal to it on the C1/C4 head counts but slower, so it
    serves only odd head counts."""
    if g.shape[-1]%2==0:
        return _tilelang_chunk_local_cumsum(g=g,chunk_size=chunk_size,cu_seqlens=cu_seqlens,**kwargs)
    assert cu_seqlens is None and not kwargs.get('reverse') and not kwargs.get('zero_last_chunk')
    return fla_chunk_local_cumsum(g,chunk_size,output_dtype=g.dtype)


flashqla_chunk.chunk_local_cumsum=_chunk_local_cumsum


def run(q,k,v,g,beta,scale=None,initial_state=None,output_final_state=False,
        use_qk_l2norm_in_kernel=False,**kwargs):
    if torch.is_grad_enabled():raise RuntimeError('Qualified FlashQLA path is inference-only')
    if kwargs.get('cu_seqlens') is not None or kwargs.get('state_v_first',False):
        raise RuntimeError('Qualified FlashQLA path is fixed-length, K-first state only')
    allowed={'cu_seqlens','state_v_first'}
    if set(kwargs)-allowed:raise RuntimeError('Unqualified GDN arguments: '+str(set(kwargs)-allowed))
    assert q.dtype==k.dtype==v.dtype==torch.bfloat16
    assert q.shape[-1]==k.shape[-1]==v.shape[-1]==64
    q=q.contiguous();k=k.contiguous();v=v.contiguous()
    if use_qk_l2norm_in_kernel:
        q,_=l2norm_fwd(q);k,_=l2norm_fwd(k)
    EVIDENCE['calls']+=1
    EVIDENCE.setdefault('first_shape',dict(q=list(q.shape),k=list(k.shape),v=list(v.shape),
        g_dtype=str(g.dtype),beta_dtype=str(beta.dtype),normalization='original_FLA_l2norm',
        scale=scale if scale is not None else 64**-.5))
    return flashqla(q=q,k=k,v=v,g=g,beta=beta,scale=scale,
        initial_state=initial_state,output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,cu_seqlens=None,state_v_first=False,
        auto_cp=False,enable_fwd_cp_cache=False)


def install(model):
    assert model.config.num_experts==0 and not model.training
    assert torch.cuda.get_device_capability()==(9,0)
    import common_gdn
    source=inspect.getsource(common_gdn._forward)
    needle='from fla.ops.gated_delta_rule import chunk_gated_delta_rule'
    assert source.count(needle)==1,'Unexpected GDN implementation'
    namespace=dict(common_gdn.__dict__)
    source=source.replace(needle,'from gdn_flashqla_backend import run as chunk_gated_delta_rule')
    exec(compile(source,__file__+':qualified_common_gdn','exec'),namespace)
    count=0
    for module in model.modules():
        if not type(module).__name__.endswith('GatedDeltaNet'):continue
        assert hasattr(module,'_common_original_gdn')
        module.forward=types.MethodType(namespace['_forward'],module)
        count+=1
    return dict(implementation='FlashQLA_SM90_D64_local_specialization',modules=count,
        core_dtype='BF16',state_dtype='FP32',normalization='original_FLA_l2norm',
        auto_cp=False,external_backend_version=EVIDENCE['version'],route_evidence=EVIDENCE,
        checkpoint_parameters_changed=False,experts_modified=False,router_modified=False)
