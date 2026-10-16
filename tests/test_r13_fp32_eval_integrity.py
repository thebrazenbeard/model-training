"""Evaluation must restore exact saved FP32 LoRA tensors, not BF16-rounded copies."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import unittest

from successor.experiments.r13_frozen_lora_inference import restore_frozen_fp32_lora


class FrozenInferenceRestoration(unittest.TestCase):
    def test_saved_fp32_exact_recovery_to_frozen_model(self):
        import torch
        from safetensors.torch import save_file
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A=torch.nn.ModuleDict({"default":torch.nn.Linear(3,2,bias=False)})
                self.lora_B=torch.nn.ModuleDict({"default":torch.nn.Linear(2,4,bias=False)})
        model=Stub()
        source={}
        for k,p in model.named_parameters():
            v=0.01012345+torch.arange(p.numel(),dtype=torch.float32).reshape(p.shape)*0.000151713
            source[k.replace(".default.",".")]=v
            p.data=v.to(torch.bfloat16)
            p.requires_grad_(False)
        with TemporaryDirectory() as td:
            path=Path(td)/"adapter_model.safetensors"
            save_file(source,path)
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            outcome=restore_frozen_fp32_lora(model,path,expected_sha=sha,expected_tensors=2,torch=torch)
            self.assertEqual(outcome["restored_tensors"],2)
            self.assertEqual(outcome["adapter_model_sha256"],sha)
            for name,p in model.named_parameters():
                self.assertFalse(p.requires_grad)
                self.assertEqual(p.dtype,torch.float32)
                self.assertTrue(torch.equal(p,source[name.replace(".default.",".")]))

    def test_refuses_name_or_digest_conflicts(self):
        import torch
        from safetensors.torch import save_file
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor([0.3]),requires_grad=False)
        m=Stub()
        with TemporaryDirectory() as td:
            path=Path(td)/"adapter_model.safetensors"
            save_file({"lora_A.weight":torch.zeros(1)},path)
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,"SHA"):
                restore_frozen_fp32_lora(m,path,expected_sha="0"*64,expected_tensors=1,torch=torch)
            with self.assertRaisesRegex(ValueError,"count"):
                restore_frozen_fp32_lora(m,path,expected_sha=sha,expected_tensors=1,torch=torch)

    def test_eval_calls_restore_before_first_generate(self):
        from successor.experiments import probe_vera_r13_heldout_retry2 as probe
        source=Path(probe.__file__).read_text(encoding="utf8")
        self.assertLess(source.index("restore_frozen_fp32_lora("),source.index("model.generate("))
        self.assertEqual(len(probe.CASES),10)


if __name__=="__main__":
    unittest.main()
