"""FP32 saved LoRA tensors must survive PEFT's implicit BF16 reload conversion."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import hashlib

from successor.experiments.r13_resume_fp32_integrity import restore_exact_lora

class R13ExactResumeTests(unittest.TestCase):
    def make(self):
        import torch
        from safetensors.torch import save_file
        t = torch.nn.Module()
        t.lora_A=torch.nn.ModuleDict({"default":torch.nn.Linear(3,2,bias=False)})
        t.lora_B=torch.nn.ModuleDict({"default":torch.nn.Linear(2,3,bias=False)})
        saved={}
        for name,p in t.named_parameters():
            p.requires_grad=True
            source = torch.arange(p.numel(),dtype=torch.float32).reshape(p.shape) * 0.0001731234 + 0.01234567
            p.data=source.clone()
            saved[name.replace(".default.","." )]=source
        return t,saved

    def test_restore_runs_after_trl_constructor_before_optimizer(self):
        text=(Path(__file__).resolve().parents[1] /
              "successor/experiments/train_v10r2_dev.py").read_text(encoding="utf8")
        trainer=text.index('    trainer = stack["SFTTrainer"](')
        restore=text.index("exact_resume_rehydration = restore_exact_lora(",trainer)
        optimizer=text.index("    optimizer = trainer.create_optimizer()",restore)
        self.assertLess(trainer,restore)
        self.assertLess(restore,optimizer)
        self.assertIn('"exact_resume_rehydration": exact_resume_rehydration',text)

    def test_restores_exact_float32_after_bfloat16_rounding(self):
        import torch
        from safetensors.torch import save_file
        m,saved=self.make()
        with TemporaryDirectory() as td:
            path=Path(td)/"adapter_model.safetensors"
            save_file(saved,path)
            expected=hashlib.sha256(path.read_bytes()).hexdigest()
            for _,p in m.named_parameters():
                p.data=p.data.to(torch.bfloat16).float()
            self.assertFalse(all(torch.equal(p.data,saved[n.replace(".default.",".")])
                                 for n,p in m.named_parameters()))
            result=restore_exact_lora(m,path,expected_sha=expected,expected_tensors=2,torch=torch)
            self.assertEqual(result["restored_tensors"],2)
            for n,p in m.named_parameters():
                self.assertEqual(p.dtype,torch.float32)
                self.assertTrue(torch.equal(p.data,saved[n.replace(".default.",".")]))
            self.assertEqual(result["adapter_model_sha256"],expected)

    def test_refuses_wrong_sha_without_changing_parameters(self):
        import torch
        from safetensors.torch import save_file
        m,saved=self.make()
        with TemporaryDirectory() as td:
            path=Path(td)/"adapter_model.safetensors"
            save_file(saved,path)
            before=[p.detach().clone() for p in m.parameters()]
            with self.assertRaisesRegex(ValueError,"SHA"):
                restore_exact_lora(m,path,expected_sha="0"*64,expected_tensors=2,torch=torch)
            self.assertTrue(all(torch.equal(x,p) for x,p in zip(before,m.parameters())))

    def test_refuses_unexpected_trainable_parameters_and_topology(self):
        import torch
        from safetensors.torch import save_file
        m,saved=self.make()
        with TemporaryDirectory() as td:
            path=Path(td)/"adapter_model.safetensors"
            save_file(saved,path)
            expected=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,"count"):
                restore_exact_lora(m,path,expected_sha=expected,expected_tensors=496,torch=torch)
            m.base=torch.nn.Parameter(torch.ones(2))
            with self.assertRaisesRegex(ValueError,"non-LoRA"):
                restore_exact_lora(m,path,expected_sha=expected,expected_tensors=2,torch=torch)


if __name__=="__main__":
    unittest.main()
