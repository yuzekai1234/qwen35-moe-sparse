"""One frozen forward: own NN, optional causal SGD, canonical backoff and cache."""
import hashlib,json,os
from pathlib import Path
import torch
from backoff_lmc_gpu import BackoffLMCHead,validate_reference
from continuous_cache_head import CausalNeuralCacheHead
from causal_adaptive_head import CausalAdaptiveHead


def wrap_feature_forward(model,forward,paths,specification):
    spec=json.loads(Path(specification).read_text())
    proof=validate_reference(spec['root'],spec['neural_config'],model.config.to_dict())
    # Conservatively count one independently adapted output matrix per input model;
    # batch replication changes runtime state, not this per-sequence parameter contract.
    proof=dict(proof)
    proof['canonical_only_total_nonembedding_bound']=proof['total_nonembedding_parameter_upper_bound']
    fast_scalars=64*int(model.config.vocab_size)
    proof['temporary_adaptive_weights_included_in_totalN']=fast_scalars
    proof['total_nonembedding_parameter_upper_bound']+=fast_scalars
    train_cfg=json.loads(Path(spec['neural_config']).read_text())
    proof['train_active_ratio_lower_bound']=train_cfg['accounting']['active_params_excluding_embeddings']/proof['total_nonembedding_parameter_upper_bound']
    assert proof['train_active_ratio_lower_bound']>=.1
    static=BackoffLMCHead(spec,torch.device('cuda',torch.cuda.current_device()))
    adaptive=CausalAdaptiveHead(**spec['adaptive'])
    cache=CausalNeuralCacheHead(**spec['continuous'])
    model._conditional_count_head=static
    def combined(inputs):
        logits,features=forward(inputs)
        logits=adaptive(inputs,features,logits,inplace=True)
        if spec['temperature']!=1:logits.div_(spec['temperature'])
        logits=static(inputs,logits)
        return cache(inputs,features,logits)
    paths['conditional_count_head']=dict(reference=proof,spec=spec,targets_in_forward=False,output='All positions/all vocabulary BF16 logits')
    paths['sequence_local_cache']=dict(mixing=spec['local_models'],targets_in_forward=False,state='Visible input prefix only; reset each call/document')
    paths['continuous_neural_cache']=dict(spec=spec['continuous'],targets_in_forward=False,source_sha256=hashlib.sha256(Path(__file__).with_name('continuous_cache_head.py').read_bytes()).hexdigest())
    paths['causal_output_head_adaptation']=dict(spec=spec['adaptive'],targets_in_forward=False,
        state='Per-sequence temporary fast output weights; only earlier predicted input tokens update them; EOS/call reset; checkpoint immutable',
        source_sha256=hashlib.sha256(Path(__file__).with_name('causal_adaptive_head.py').read_bytes()).hexdigest(),
        temporary_fast_weights_conservatively_counted_in_totalN=True)
    return combined,paths
