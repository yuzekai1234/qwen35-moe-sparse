"""Apply the qualified dual-prefix graph to the unchanged A3 neural forward."""
import json, os
from pathlib import Path
import torch
from backoff_lmc_gpu import BackoffLMCHead,validate_reference
def wrap_forward(builder,model):
 forward,paths=builder(model);spec=json.loads(Path(os.environ['Q38_JOINT_SPEC']).read_text());proof=validate_reference(spec['root'],os.environ['Q38_JOINT_NEURAL_CONFIG'],model.config.to_dict());head=BackoffLMCHead(spec,torch.device('cuda',torch.cuda.current_device()));model._conditional_count_head=head
 def combined(inputs):
  logits=forward(inputs)
  if spec['temperature']!=1:logits.div_(spec['temperature'])
  return head(inputs,logits)
 paths['conditional_count_head']=dict(reference=proof,spec=spec,targets_in_forward=False,output='All positions/all vocabulary BF16 logits');paths['sequence_local_cache']=dict(mixing=spec['local_models'],targets_in_forward=False,state='Rebuilt from visible inputs; EOS resets; no cross-call state')
 return combined,paths
