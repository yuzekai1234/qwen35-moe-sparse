"""CPU checks of official single-GPU packing, warmup, scoring and timing scope."""
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from q38.evaluate_single_gpu import evaluate_model, pack_jsonl, packed_batch, parse_args


class ToyModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, input_ids, use_cache, return_dict):
        assert use_cache is False and return_dict is True
        self.inputs.append(input_ids.clone())
        scores = torch.arange(6, dtype=torch.float32)
        return SimpleNamespace(logits=input_ids[..., None].float()*scores/10)


class SingleGPU(unittest.TestCase):
    def setUp(self):
        self.tokens = np.array([0, 1, 2, 0, 3, 4, 5, 0, 1, 2], dtype=np.int32)
        self.mask = self.tokens != 0
        self.metadata = dict(total_tokens=7, total_bytes=11, pad_token_id=0)

    def run_toy(self, batch_size=2, metadata=None):
        model = ToyModel()
        with patch("torch.cuda.synchronize") as sync, \
                patch("q38.evaluate_single_gpu.time.perf_counter", side_effect=[10, 11, 20, 22]):
            result = evaluate_model(model, self.tokens, self.mask, metadata or self.metadata,
                                    batch_size, 4, "cpu", 100)
        return model, sync, result

    def test_matches_complete_unbatched_nll_and_normalized_time(self):
        model, _, result = self.run_toy()
        inputs, labels = packed_batch(self.tokens, self.mask, 0, 3, 4, 0)
        logits = ToyModel()(torch.from_numpy(inputs), False, True).logits
        nll = F.cross_entropy(logits.reshape(-1, 6), torch.from_numpy(labels).reshape(-1),
                              ignore_index=-100, reduction="none").double().sum().item()
        self.assertEqual(result["scored_tokens"], 7)
        self.assertEqual(result["total_nll_nats"], nll)
        self.assertEqual(result["bpb"], nll/(math.log(2)*11))
        self.assertEqual(result["forward_seconds"], 3)
        self.assertEqual(result["evaltime"], 3/((100/1e9)*(7/1e6)))
        self.assertEqual((result["gpu_count"], result["num_sequences"], result["num_steps"]), (1, 3, 2))
        self.assertTrue(torch.equal(torch.cat(model.inputs[-2:]), torch.from_numpy(inputs)))

    def test_warms_each_distinct_full_and_tail_batch_five_times(self):
        model, sync, _ = self.run_toy()
        self.assertEqual([tuple(x.shape) for x in model.inputs], [(2, 4)]*5+[(1, 4)]*5+[(2, 4), (1, 4)])
        self.assertEqual(sync.call_count, 2+2*2)

    def test_same_shape_warms_only_once(self):
        model = ToyModel()
        with patch("torch.cuda.synchronize"), \
                patch("q38.evaluate_single_gpu.time.perf_counter", side_effect=[10, 12]):
            evaluate_model(model, self.tokens, self.mask, self.metadata, 4, 4, "cpu", 100)
        self.assertEqual(len(model.inputs), 6)

    def test_rejects_wrong_scored_count_and_invalid_inputs(self):
        with self.assertRaisesRegex(RuntimeError, "Scored token count"):
            self.run_toy(metadata=dict(self.metadata, total_tokens=8))
        for batch, seq, active in ((0, 4, 100), (2, 0, 100), (2, 4, 0)):
            with self.assertRaises(ValueError):
                evaluate_model(ToyModel(), self.tokens, self.mask, self.metadata, batch, seq, "cpu", active)
        with self.assertRaises(ValueError):
            evaluate_model(ToyModel(), self.tokens, self.mask[:-1], self.metadata, 2, 4, "cpu", 100)

    def test_timer_does_not_include_loss(self):
        events = []
        ce = F.cross_entropy
        def loss(*args, **kwargs):
            events.append("loss")
            return ce(*args, **kwargs)
        def clock():
            events.append("clock")
            return len(events)
        with patch("torch.cuda.synchronize"), patch("q38.evaluate_single_gpu.time.perf_counter", clock), \
                patch("torch.nn.functional.cross_entropy", loss):
            evaluate_model(ToyModel(), self.tokens, self.mask, self.metadata, 2, 4, "cpu", 100)
        self.assertEqual(events, ["clock", "clock", "loss", "clock", "clock", "loss"])

    def test_packing_keeps_utf8_text_eos_context_and_padded_tail(self):
        class Tokenizer:
            eos_token_id, pad_token_id = 0, 0
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[1]*len(x) for x in texts]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root/"input.jsonl"
            data.write_text(json.dumps({"text": "中文"})+"\n\n"+json.dumps({"text": "abc"})+"\n")
            metadata = pack_jsonl(data, Tokenizer(), root)
            tokens = np.fromfile(root/"tokens.bin", dtype=np.int32)
            mask = np.fromfile(root/"is_text.bin", dtype=np.bool_)
        self.assertEqual(metadata, dict(total_tokens=5, total_bytes=9, pad_token_id=0))
        self.assertEqual(tokens.tolist(), [0, 1, 1, 0, 1, 1, 1])
        _, labels = packed_batch(tokens, mask, 0, 2, 4, 0)
        self.assertEqual(int(np.count_nonzero(labels != -100)), 5)
        self.assertEqual(labels.tolist(), [[1, 1, -100, 1], [1, 1, -100, -100]])

    def test_cli_rejects_overwrite_and_invalid_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"result.json"
            args = ["--dataset-path", "/unused/data", "--model-path", "/unused/model", "--output", str(output)]
            self.assertEqual(parse_args(args).sequence_length, 4096)
            with patch("sys.stderr"):
                for extra in (["--batch-size", "0"], ["--sequence-length", "0"]):
                    with self.assertRaises(SystemExit):
                        parse_args(args+extra)
                output.write_text("{}")
                with self.assertRaises(SystemExit):
                    parse_args(args)


if __name__ == "__main__":
    unittest.main()
