"""CPU-only qualification of the published snapshot; no Torch/GPU imports."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"


def pure_functions(filename, names, namespace=None):
    tree = ast.parse((EVAL / filename).read_text())
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in names], type_ignores=[])
    scope = {} if namespace is None else namespace
    exec(compile(module, filename, "exec"), scope)
    return scope


class PublishedContract(unittest.TestCase):
    def test_source_matches_full_A_snapshot(self):
        manifest = json.loads((ROOT / "validated-source-sha256.json").read_text())
        for name, expected in manifest.items():
            self.assertEqual(hashlib.sha256((EVAL / name).read_bytes()).hexdigest(), expected, name)

    def test_batch_priority(self):
        spec = importlib.util.spec_from_file_location("entry_defaults", EVAL / "entry_defaults.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(entry.explicit_batch())
            os.environ["EVAL_BATCH_SIZE"] = "1660"
            self.assertEqual(entry.explicit_batch(), 1660)
            self.assertEqual(entry.explicit_batch(942), 942)
            os.environ["EVAL_BATCH_SIZE"] = "auto"
            self.assertIsNone(entry.explicit_batch())
            with self.assertRaises(ValueError):
                entry.explicit_batch(0)

    def test_capacity_search_has_no_pilot_limit(self):
        functions = pure_functions("autobatch.py", {"largest_fitting", "full_backoff"})
        calls = []
        def probe(batch):
            calls.append(batch)
            return batch <= 1660
        self.assertEqual(functions["largest_fitting"](probe, 2000), 1660)
        self.assertIn(1661, calls)
        self.assertEqual(functions["full_backoff"](
            lambda b: "ok" if b <= 942 else "oom", 945), 942)
        with self.assertRaises(RuntimeError):
            functions["full_backoff"](lambda b: "error", 945)

    def test_bf16_residual_rejects_partial_prequant(self):
        validate = pure_functions("prequant_norm.py", {"validate_configuration"}, {"os": os})[
            "validate_configuration"]
        with patch.dict(os.environ, {"Q38_RESIDUAL_DTYPE": "bf16"}, clear=True):
            validate()
            for name in ("Q38_PREQUANT_NORM", "Q38_PREQUANT_INPUT", "Q38_PREQUANT_FFN"):
                os.environ[name] = "0"
                with self.assertRaises(ValueError):
                    validate()
                del os.environ[name]

    def test_full_A_evidence_and_accounting(self):
        result = json.loads((ROOT / "evidence/gpu7-result.json").read_text())
        metadata = json.loads((ROOT / "evidence/gpu7-run-metadata.json").read_text())
        self.assertEqual(metadata["actual_batch_sizes"], [942] * 31 + [146])
        self.assertEqual(metadata["scored_tokens"], 120142976)
        self.assertEqual(metadata["text_utf8_bytes"], 445910984)
        self.assertEqual(metadata["active_nonvocab_params"], 33776160)
        self.assertEqual(metadata["materialized_logits_dtype"], "torch.bfloat16")
        expected = metadata["forward_seconds"] / ((33776160 / 1e9) * (120142976 / 1e6))
        self.assertAlmostEqual(result["evaltime"], expected, places=10)
        self.assertLess(result["bpb"] - 1.121291825565294, .03)
        first = json.loads((ROOT / "evidence/gpu6-result.json").read_text())
        self.assertAlmostEqual(first["bpb"], result["bpb"], places=12)

    def test_timer_covers_forward_and_inputs_release_after_CE(self):
        tree = ast.parse((EVAL / "eval.py").read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and any(isinstance(x, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "logits" for t in x.targets)
                            for x in n.body))
        code = ast.unparse(loop)
        self.assertLess(code.index("start_event.record()"), code.index("logits = forward_only(inputs)"))
        self.assertLess(code.index("logits = forward_only(inputs)"), code.index("end_event.record()"))
        self.assertLess(code.index("end_event.record()"), code.index("cross_entropy"))
        released = {x.id for n in ast.walk(loop) if isinstance(n, ast.Delete)
                    for x in n.targets if isinstance(x, ast.Name)}
        self.assertTrue({"inputs", "labels", "batch", "logits"}.issubset(released))


if __name__ == "__main__":
    unittest.main()
