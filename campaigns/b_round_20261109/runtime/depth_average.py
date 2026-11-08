"""DenseFormer depth weighted averaging, raw block outputs and embedded input."""
import torch
from torch import nn


class PaperDepthWeightedAverage(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.weights=nn.ParameterList([nn.Parameter(torch.cat([torch.zeros(i+1),torch.ones(1)])) for i in range(layers)])

    def mix(self,index,raw):
        if len(raw)!=index+2:raise ValueError('Expected embedding plus each raw block output')
        weights=self.weights[index]
        # No softmax/normalization: paper coefficients are independent learned scalars.
        result=raw[-1]*weights[-1].to(raw[-1].dtype)
        for value,weight in zip(raw[:-1],weights[:-1]):result=result+value*weight.to(value.dtype)
        return result
