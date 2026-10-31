"""Row-wise native E4M3 projections with explicit optional control/head coverage."""
import os,types
import torch
from low_precision import quant_rows


def _forward(self,x):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('FP8 linear requires eval and inference_mode')
    shape=x.shape
    q,scale=quant_rows(x.reshape(-1,shape[-1]))
    output=torch._scaled_mm(q,self._q8_weight.t(),scale_a=scale.view(-1,1),
        scale_b=self._q8_weight_scale.view(1,-1),bias=self._q8_bias,
        out_dtype=torch.bfloat16,use_fast_accum=os.environ.get('Q38_FP8_FAST_ACCUM')=='1')
    return output[:,:self.out_features].reshape(*shape[:-1],self.out_features)


def install(model,scope='projections'):
    selected=[]
    protected={id(m.weight) for m in model.modules() if isinstance(m,torch.nn.Embedding)}
    with torch.no_grad():
        for name,module in model.named_modules():
            if not isinstance(module,torch.nn.Linear):
                continue
            eligible=(('.linear_attn.' in name and name.rsplit('.',1)[-1] in ('in_proj_qkv','in_proj_z','out_proj'))
                or ('.self_attn.' in name and name.rsplit('.',1)[-1] in ('q_proj','k_proj','v_proj','o_proj'))
                or ('.mlp.shared_expert.' in name and name.rsplit('.',1)[-1] in ('gate_proj','up_proj','down_proj')))
            if scope in ('head','maximal') and name=='lm_head':eligible=True
            if scope in ('maximal','general') and '.mhc.' not in name:eligible=True
            if scope=='general' and ('.experts.' in name or '.router' in name or
                name.endswith('.mlp.gate') or name.endswith('.shared_expert_gate')):eligible=False
            if not eligible:
                continue
            if module.in_features%16:
                raise ValueError('Unsupported FP8 projection shape: '+name)
            weight=module.weight.detach()
            padding=(-module.out_features)%16
            if padding:weight=torch.nn.functional.pad(weight,(0,0,0,padding))
            q,scale=quant_rows(weight)
            bias=module.bias
            if bias is not None and padding:bias=torch.nn.functional.pad(bias,(0,padding))
            module.register_buffer('_q8_bias',bias,persistent=False)
            module.register_buffer('_q8_weight',q,persistent=False)
            module.register_buffer('_q8_weight_scale',scale,persistent=False)
            module.forward=types.MethodType(_forward,module)
            if name!='lm_head' and id(module.weight) not in protected:
                module.weight.data=module.weight.detach().cpu()
            selected.append(name)
        if scope=='maximal':
            for name,module in model.named_modules():
                if type(module).__name__!='Qwen3_8MoeTopKRouter':continue
                q,scale=quant_rows(module.weight.detach())
                module.register_buffer('_q8_weight',q,persistent=False)
                module.register_buffer('_q8_weight_scale',scale,persistent=False)
                module.register_buffer('_q8_bias',None,persistent=False)
                module.out_features=module.num_experts
                module.weight.data=module.weight.detach().cpu()
                module.forward=types.MethodType(_router_forward,module)
                selected.append(name+'[router]')
    return dict(implementation='torch._scaled_mm_native_fp8',format='E4M3_W8A8',
                activation_scale='dynamic_per_row',weight_scale='per_output_row',
                selected=selected,scope=scope,use_fast_accum=os.environ.get('Q38_FP8_FAST_ACCUM')=='1')


def _router_forward(self,hidden_states):
    logits=_forward(self,hidden_states.reshape(-1,self.hidden_dim))
    probabilities=logits.softmax(-1,dtype=torch.float32)
    values,indices=probabilities.topk(self.top_k,dim=-1)
    values=values/values.sum(-1,keepdim=True)
    return logits,values.to(logits.dtype),indices
