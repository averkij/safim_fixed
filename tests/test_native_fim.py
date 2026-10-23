from argparse import Namespace
import types
import unittest
from unittest.mock import Mock
from model_support import load_models, Tensor


class NativeFimTests(unittest.TestCase):
    def prepare(self):
        module=load_models()
        module.torch.float32="float32"
        module.transformers.AutoConfig=Mock()
        module.transformers.AutoConfig.from_pretrained.return_value=types.SimpleNamespace(max_position_embeddings=8192)
        tokenizer=Mock(unk_token_id=0)
        tokenizer.encode.return_value=[3]
        tokenizer.init_kwargs={}
        module.AutoTokenizer.from_pretrained.return_value=tokenizer
        return module

    def test_native_marker_order_and_reverse_rejection(self):
        module=self.prepare()
        for name,expected in [("Qwen/Qwen2.5-Coder-3B","<|fim_prefix|>P<|fim_suffix|>S<|fim_middle|>"),
                              ("JetBrains/Mellum-4b-base","<fim_suffix>S<fim_prefix>P<fim_middle>")]:
            with self.subTest(model=name):
                wrapper=module.NativeFIMModel(name,device="cpu",dtype="float32")
                self.assertEqual(wrapper.assemble_infilling_prompt("P","S"),expected)
                with self.assertRaises(ValueError):wrapper.assemble_infilling_prompt("P","S",True)
                args=module.AutoModelForCausalLM.from_pretrained.call_args.kwargs
                self.assertEqual(args["revision"],module.NATIVE_FIM_MODELS[name]["revision"])
                self.assertFalse(args["trust_remote_code"])

    def test_reject_moving_revision_invalid_context_and_missing_markers(self):
        module=self.prepare();name="Qwen/Qwen2.5-Coder-3B"
        with self.assertRaises(ValueError):module.NativeFIMModel(name,revision="main")
        with self.assertRaises(ValueError):module.NativeFIMModel(name,context_limit=9000)
        module.AutoTokenizer.from_pretrained.return_value.encode.return_value=[3,4]
        with self.assertRaises(ValueError):module.NativeFIMModel(name)
        module.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_native_options_are_not_silently_ignored_by_legacy_models(self):
        module=self.prepare()
        with self.assertRaises(ValueError):
            module.build_model(Namespace(model_name="bigcode/starcoder",block_comments=False,device="cpu"))

    def test_invocation_keeps_attention_and_output_only_decoding(self):
        module=self.prepare()
        wrapper=module.NativeFIMModel("JetBrains/Mellum-4b-base",device="cpu",dtype="float32")
        class Batch(dict):
            def to(self,*args):return self
        wrapper.tokenizer.return_value=Batch(input_ids=Tensor(),attention_mask=Tensor())
        wrapper.tokenizer.eos_token_id=0;wrapper.tokenizer.pad_token_id=None
        wrapper.model.generate.return_value=Tensor()
        wrapper.max_new_tokens=7
        wrapper.invoke("prompt")
        args=wrapper.model.generate.call_args.kwargs
        self.assertEqual((args["max_new_tokens"],args["top_k"],args["pad_token_id"]),(7,0,0))
        self.assertIn("attention_mask",args)
        self.assertFalse(wrapper.tokenizer.call_args.kwargs["truncation"])
        self.assertFalse(wrapper.tokenizer.decode.call_args.kwargs["clean_up_tokenization_spaces"])


if __name__=="__main__":unittest.main()
