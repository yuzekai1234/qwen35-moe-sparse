"""Dense-reusable Gated DeltaNet: share input quantization across projections."""
import os,types
import torch
import torch.nn.functional as F
from low_precision import quant_rows


def _project(linear,q,scale,batch,sequence):
    y=torch._scaled_mm(q,linear._q8_weight.t(),scale_a=scale[:,None],
        scale_b=linear._q8_weight_scale[None,:],bias=linear._q8_bias,
        out_dtype=torch.bfloat16,use_fast_accum=True)
    return y[:,:linear.out_features].reshape(batch,sequence,linear.out_features)


def _forward(self,hidden_states,cache_params=None,attention_mask=None,**kwargs):
    if cache_params is not None or attention_mask is not None or kwargs.get('cu_seq_lens_q') is not None:
        return self._common_original_gdn(hidden_states,cache_params=cache_params,attention_mask=attention_mask,**kwargs)
    if self.training or torch.is_grad_enabled():raise RuntimeError('Shared GDN FP8 quantization is inference-only')
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    batch,sequence,_=hidden_states.shape
    limit=int(os.environ.get('Q38_GDN_BATCH_LIMIT','64'))
    if limit<1:raise ValueError('GDN batch tile must be positive')
    if batch>limit:
        return torch.cat([_forward(self,hidden_states[start:start+limit]) for start in range(0,batch,limit)],0)
    q,scale=quant_rows(hidden_states.reshape(-1,hidden_states.shape[-1]))
    projected=_project(self.in_proj_qkv,q,scale,batch,sequence)
    mixed,_=causal_conv1d(projected,weight=self.conv1d.weight.squeeze(1),bias=self.conv1d.bias,activation=self.activation)
    del projected
    query,key,value=torch.split(mixed,[self.key_dim,self.key_dim,self.value_dim],dim=-1)
    query=query.reshape(batch,sequence,-1,self.head_k_dim)
    key=key.reshape(batch,sequence,-1,self.head_k_dim)
    value=value.reshape(batch,sequence,-1,self.head_v_dim)
    z=_project(self.in_proj_z,q,scale,batch,sequence)
    beta=_project(self.in_proj_b,q,scale,batch,sequence).sigmoid()
    a=_project(self.in_proj_a,q,scale,batch,sequence)
    del q,scale
    g=-self.A_log.float().exp()*F.softplus(a.float()+self.dt_bias)
    del a
    output,_=chunk_gated_delta_rule(query,key,value,g=g,beta=beta,use_qk_l2norm_in_kernel=True)
    del query,key,value,mixed,g,beta
    output=self.norm(output.reshape(-1,self.head_v_dim),z.reshape(-1,self.head_v_dim))
    del z
    return self.out_proj(output.reshape(batch,sequence,-1))


def install(model):
    if model.training:raise ValueError('Load an eval model before GDN optimization')
    count=0
    for module in model.modules():
        if not type(module).__name__.endswith('GatedDeltaNet'):continue
        projections=('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b')
        if not all(hasattr(getattr(module,n),'_q8_weight') for n in projections):
            raise ValueError('Install general FP8 projections before shared GDN quantization')
        module._common_original_gdn=module.forward
        module.forward=types.MethodType(_forward,module);count+=1
    return dict(modules=count,input_quantizations_per_GDN_tile=1,previous_input_quantizations=4,
        intermediate_lifetimes='release_after_last_use',core_state_precision='unchanged',expert_or_router_changes=False)
