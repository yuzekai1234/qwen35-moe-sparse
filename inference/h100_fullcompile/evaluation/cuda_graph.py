"""Forward-only CUDA graph replay with separate captures for batch shapes."""
import torch


class GraphForward:
    def __init__(self, model):
        self.model = model
        self.captures = {}

    def __call__(self, inputs):
        key = (tuple(inputs.shape), inputs.dtype, inputs.device)
        if key not in self.captures:
            static_input = inputs.clone()
            current = torch.cuda.current_stream(inputs.device)
            warmup = torch.cuda.Stream(device=inputs.device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    output = self.model(input_ids=static_input, use_cache=False, return_dict=True).logits
                    del output
            current.wait_stream(warmup)
            torch.cuda.synchronize(inputs.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.model(input_ids=static_input, use_cache=False, return_dict=True).logits
            self.captures[key] = graph, static_input, output
        graph, static_input, output = self.captures[key]
        static_input.copy_(inputs)
        graph.replay()
        return output
