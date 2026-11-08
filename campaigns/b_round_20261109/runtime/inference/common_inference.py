"""Dense-reusable inference changes: norm fusion and short tensor lifetimes."""
import sys,types
import torch


def _inference_layer(self,index,X,ids,cos,sin,dtype):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Tensor lifetime optimization is inference-only')
    layer=self.layers[index];kl=None
    xin,state=self.mhc[2*index].pre(X)
    x=layer.input_layernorm(xin).to(dtype)
    del xin
    if self.config.layer_types[index]=='linear_attention':
        y=layer.linear_attn(hidden_states=x,cache_params=None,attention_mask=None)
    else:
        y,kl=self._common_qsa(layer.self_attn,x,cos,sin,self.indexers[str(index)],
            self.config.qsa_blocks,False,False)
    del x
    X=self.mhc[2*index].post(X,y,state)
    del y,state
    xin,state=self.mhc[2*index+1].pre(X)
    x=layer.post_attention_layernorm(xin).to(dtype)
    del xin
    y=layer.mlp(x)
    del x
    X=self.mhc[2*index+1].post(X,y,state)
    del y,state
    if index==0 and self.ngram is not None:
        X=X+self.ngram.delta(X.mean(2).to(dtype),ids).unsqueeze(2)
    return X,kl


def install_memory_optimizations(model):
    if model.training:raise RuntimeError('Load an eval model before inference optimization')
    norms=0
    for module in model.modules():
        if isinstance(module,(torch.nn.RMSNorm,torch.nn.LayerNorm)):
            module.compile(fullgraph=True,dynamic=True,options={'emulate_precision_casts':True})
            norms+=1
    trunk=getattr(model,'model',None)
    implementation=sys.modules[type(model).__module__]
    applicable=(trunk is not None and hasattr(trunk,'_layer') and hasattr(trunk,'mhc')
        and hasattr(trunk,'layers') and len(trunk.mhc)>0 and hasattr(implementation,'qsa_attention'))
    if applicable:
        if len(trunk.mhc)!=2*len(trunk.layers):raise ValueError('Unexpected mHC layer layout')
        # Store a plain function on the instance, so nn.Module does not bind self.
        trunk._common_qsa=implementation.qsa_attention
        trunk._layer=types.MethodType(_inference_layer,trunk)
    return dict(torch_norms_compiled=norms,shortened_layer_temporaries=bool(applicable),
                scope='attention_GDN_mHC_dense_or_MoE_FFN_agnostic',expert_or_router_changes=False)
