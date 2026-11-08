"""SM90 FlashInfer/CUTLASS block-FP8 and Humming MXFP4 operators."""
import os
import torch


def _forward(self,hidden_states,top_k_index,top_k_weights):
    if torch.is_grad_enabled():
        raise RuntimeError('FlashInfer experts are inference-only')
    from flashinfer.fused_moe import MoEActivationPack
    limit=self._fi_token_chunk
    if hidden_states.shape[0] > limit:
        outputs=[]
        for start in range(0,hidden_states.shape[0],limit):
            stop=start+limit
            chunk=MoEActivationPack(hidden_states_q=hidden_states[start:stop],hidden_states_scale=None,
                topk_ids=top_k_index[start:stop].int(),topk_weights=top_k_weights[start:stop].float())
            outputs.append(self._fi_layer(chunk,self._fi_weights))
        return torch.cat(outputs,dim=0)
    pack=MoEActivationPack(hidden_states_q=hidden_states,hidden_states_scale=None,
        topk_ids=top_k_index.int(),topk_weights=top_k_weights.float())
    return self._fi_layer(pack,self._fi_weights)


def install(model,precision='fp8'):
    from flashinfer.fused_moe import (MoELayer,MoEConfig,RoutingConfig,QuantConfig,QuantFormat,
        ExpertConfig,BackendOptions,ExecutionConfig,MoEWeightPack,CutlassFp8BlockConfig,CutlassHummingConfig)
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    backend=CutlassFp8BlockConfig() if precision=='fp8' else CutlassHummingConfig()
    quant=QuantConfig(weight=QuantFormat.DeepSeekFp8,activation=QuantFormat.DeepSeekFp8) if precision=='fp8' else QuantConfig(
        weight=QuantFormat.MXFP4,activation=QuantFormat.FP8PerTensor)
    count=0;keys=[];layers={}
    token_chunk=int(os.environ.get('Q38_FLASHINFER_TOKEN_CHUNK','131072'))
    if token_chunk<1:raise ValueError('FlashInfer token chunk must be positive')
    for module in model.modules():
        if type(module).__name__!='Qwen3_8MoeExperts':
            continue
        experts=module.num_experts;hidden=module.hidden_dim;intermediate=module.intermediate_dim
        config=MoEConfig(routing=RoutingConfig(num_experts=experts,top_k=model.config.num_experts_per_tok),
            quant=quant,experts=ExpertConfig(intermediate_size=intermediate),
            backend=BackendOptions(candidates=(backend,)),
            execution=ExecutionConfig(tune_max_num_tokens=token_chunk))
        geometry=(experts,hidden,intermediate,model.config.num_experts_per_tok,precision)
        if geometry not in layers:layers[geometry]=MoELayer(config,device=torch.device('cuda',0))
        layer=layers[geometry]
        # FlashInfer's canonical gated weights are [up, gate]; Q38 is [gate, up].
        gate,up=module.gate_up_proj.detach().chunk(2,dim=1)
        canonical=torch.cat((up,gate),dim=1)
        view=backend.prepare_weights(canonical,module.down_proj.detach(),num_local_experts=experts,
            hidden_size=hidden,intermediate_size=intermediate,device=torch.device('cuda',0))
        pack=MoEWeightPack()
        for runner in layer.runners:
            pack.prepare_for(runner.backend_key,view)
            keys.append(runner.backend_key)
        module._fi_layer=layer;module._fi_weights=pack;module._fi_token_chunk=token_chunk
        module.gate_up_proj.data=module.gate_up_proj.detach().cpu()
        module.down_proj.data=module.down_proj.detach().cpu()
        count+=1
    name='q38_flashinfer_'+precision
    ALL_EXPERTS_FUNCTIONS.register(name,_forward)
    model.set_experts_implementation(name)
    return dict(modules=count,backends=sorted(set(keys)),precision=precision,
                routing='original_precomputed_top5',weight_order='canonical_up_gate',
                implementation='FlashInfer0.7.2.post1_CUTLASS_SM90',shared_runner_instances=len(layers),
                token_chunk=token_chunk,workspace_reuse='shared_sequential_same_geometry',
                activation_scale_scope='per_chunk_tensor' if precision=='fp4' else 'per_token_block128')
