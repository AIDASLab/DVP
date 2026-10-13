"""CPU-only contract tests; the inference engine is replaced by a test double."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import LLMModel


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return json.dumps(messages)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.options = kwargs


class FakeEngine:
    instances = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.instances.append(self)

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompt, sampling_params, use_tqdm):
        self.calls.append((prompt, sampling_params.options, use_tqdm))
        text = "  answer:" + json.loads(prompt)[-1]["content"] + "  "
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)])]


class ModelInterfaceTests(unittest.TestCase):
    def setUp(self):
        FakeEngine.instances = []
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeEngine
        fake_vllm.SamplingParams = FakeSamplingParams
        previous = sys.modules.get("vllm")
        was_loaded = "vllm" in sys.modules
        sys.modules["vllm"] = fake_vllm

        def restore_dependency():
            if was_loaded:
                sys.modules["vllm"] = previous
            else:
                sys.modules.pop("vllm", None)

        self.addCleanup(restore_dependency)

    def test_import_does_not_require_inference_dependencies(self):
        code = """
import sys
sys.modules.update({'vllm': None, 'transformers': None, 'torch': None})
from model import LLMModel
assert callable(LLMModel)
"""
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=env, check=True)

    def test_arbitrary_compatible_model_id_and_ordered_text_contract(self):
        model = LLMModel(
            "my-organization/new-chat-model",
            max_new_tokens=73,
            temperature=0.4,
            top_p=0.82,
            dtype="float16",
            tensor_parallel_size=2,
            system_prompt="classify the text",
        )
        engine = FakeEngine.instances[0]
        self.assertEqual(engine.options, {
            "model": "my-organization/new-chat-model",
            "dtype": "float16",
            "tensor_parallel_size": 2,
        })
        self.assertEqual(model(["first", "second"]), ["answer:first", "answer:second"])
        self.assertEqual(engine.calls[0][1], {"max_tokens": 73, "temperature": 0.4, "top_p": 0.82})
        self.assertFalse(engine.calls[0][2])
        messages, template = engine.tokenizer.calls[0]
        self.assertEqual(messages, [
            {"role": "system", "content": "classify the text"},
            {"role": "user", "content": "first"},
        ])
        self.assertEqual(template, {"tokenize": False, "add_generation_prompt": True})
        self.assertEqual(model([]), [])
        with self.assertRaises(TypeError):
            model("a single string is not a prompt batch")

    def test_local_model_override_dtype_and_optional_engine_settings(self):
        local_path = ROOT / "test-only-no-model-download"
        model = LLMModel(
            "unused/identifier",
            model_dir=local_path,
            dtype="bfloat16",
            tensor_parallel_size=3,
            engine_kwargs={"max_model_len": 4096, "trust_remote_code": False},
        )
        self.assertEqual(model.model.model_name, str(local_path))
        self.assertEqual(FakeEngine.instances[0].options, {
            "model": str(local_path),
            "dtype": "bfloat16",
            "tensor_parallel_size": 3,
            "max_model_len": 4096,
            "trust_remote_code": False,
        })

    def test_per_call_sampling_overrides_do_not_change_defaults(self):
        model = LLMModel("org/chat", max_new_tokens=90, temperature=0.3, top_p=0.9)
        model(["one"], max_tokens=17, temperature=0.6, top_p=0.75, seed=123)
        model(["two"])
        first, second = FakeEngine.instances[0].calls
        self.assertEqual(first[1], {"max_tokens": 17, "temperature": 0.6, "top_p": 0.75, "seed": 123})
        self.assertEqual(second[1], {"max_tokens": 90, "temperature": 0.3, "top_p": 0.9})

    def test_template_controls_are_model_specific_and_overridable(self):
        plain = LLMModel("org/chat", chat_template_kwargs={"custom_mode": "short"})
        plain(["one"])
        self.assertEqual(FakeEngine.instances[-1].tokenizer.calls[0][1], {
            "tokenize": False, "add_generation_prompt": True, "custom_mode": "short",
        })
        qwen = LLMModel("Qwen/Qwen3-4B")
        qwen(["two"])
        self.assertFalse(FakeEngine.instances[-1].tokenizer.calls[0][1]["enable_thinking"])
        qwen_override = LLMModel("Qwen/Qwen3-4B", chat_template_kwargs={"enable_thinking": True})
        qwen_override(["three"])
        self.assertTrue(FakeEngine.instances[-1].tokenizer.calls[0][1]["enable_thinking"])

    def test_invalid_options_rejected_before_engine_creation(self):
        for kwargs in [
            {"device": "cuda:2"},
            {"device": "cpu"},
            {"model_dir": ""},
            {"tensor_parallel_size": 0},
            {"engine_kwargs": {"dtype": "float16"}},
            {"engine_kwargs": {"model": "different/model"}},
            {"chat_template_kwargs": {"tokenize": True}},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LLMModel("org/chat", **kwargs)
        self.assertEqual(FakeEngine.instances, [])

    def test_annotation_callers_forward_supported_settings(self):
        callers = [
            ("processing.extract_valued_text", "ExpressiveSentenceExtractor", {}),
            ("value_item_parser.extract_value_item", "ValueItemExtractor", {}),
            ("value_item_parser.generate_posneg_sentences_from_value_items", "PosNegGenerator", {}),
            ("intensity_parser.classify_surface_value_regimes", "SurfaceRegimeClassifier", {}),
            ("context_parser.extract_context_states", "SingleLLM", {"include_evidence": False}),
        ]
        for module_name, class_name, extra in callers:
            with self.subTest(caller=class_name), patch("model.LLMModel") as factory:
                cls = getattr(importlib.import_module(module_name), class_name)
                cls(
                    model_name="org/chat", tensor_parallel_size=2,
                    temperature=0.2, top_p=0.77, max_new_tokens=31,
                    device="auto", dtype="float16", model_dir="/test/local-chat", **extra,
                )
                supplied = factory.call_args.kwargs
                self.assertEqual(supplied["model_dir"], "/test/local-chat")
                self.assertEqual(supplied["top_p"], 0.77)
                self.assertEqual(supplied["dtype"], "float16")
                self.assertEqual(supplied["tensor_parallel_size"], 2)
        with patch("model.LLMModel") as factory:
            from relation_parser.extract_value_relationships import LLMRunner
            LLMRunner("/test/local-chat", 2, 0.2, 0.77, 31, "extract relations")
            self.assertEqual(factory.call_args.kwargs["top_p"], 0.77)
            self.assertEqual(factory.call_args.kwargs["model"], "/test/local-chat")

    def test_context_stage_switch_reuses_engine_and_updates_system_message(self):
        from context_parser.extract_context_states import SingleLLM
        parser = SingleLLM("org/chat", 1, 0, 0.81, 200, include_evidence=False)
        parser.set_stage("activate")
        self.assertEqual(parser.run(["first"], max_tokens=180), ["answer:first"])
        parser.set_stage("state")
        parser.run(["second"])
        self.assertEqual(len(FakeEngine.instances), 1)
        engine = FakeEngine.instances[0]
        self.assertEqual(engine.tokenizer.calls[0][0][0]["content"], parser.system_prompt_activate)
        self.assertEqual(engine.tokenizer.calls[1][0][0]["content"], parser.system_prompt_state)
        self.assertEqual(engine.calls[0][1]["max_tokens"], 180)
        self.assertEqual(engine.calls[1][1]["max_tokens"], 200)
        self.assertEqual(engine.calls[1][1]["top_p"], 0.81)


if __name__ == "__main__":
    unittest.main()
