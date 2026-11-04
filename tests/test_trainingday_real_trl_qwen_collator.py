"""Opt-in real TRL 1.17.1 dataloader gate; no Qwen weights or training.

Run with ``TRAININGDAY_REAL_TRL_QWEN=1`` and a verified local tokenizer.json.
The public Calc TRAIN source must be available on PYTHONPATH; DEV is excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trainingday import calc_curriculum, calc_tokenization
from trainingday.pinned_base import BASE_ID, BASE_REVISION


@unittest.skipUnless(os.environ.get("TRAININGDAY_REAL_TRL_QWEN") == "1",
                     "real TRL/Qwen tokenizer dependency gate is opt-in")
class RealTrlQwenCalcBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from importlib.metadata import version

        if version("trl") != "1.17.1" or version("tokenizers") != "0.23.3":
            raise RuntimeError("TRL/tokenizers versions differ from hosted runtime candidate")
        path = Path(os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != (
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
        ):
            raise RuntimeError("tokenizer bytes not the pinned official artifact")
        cls.raw = Tokenizer.from_str(raw.decode("utf-8"))
        if cls.raw.token_to_id("<|im_end|>") != 248046:
            raise RuntimeError("pinned Qwen EOS ID changed")

        class Adapter:
            eos_token_id = 248046

            def encode(self, text, *, add_special_tokens):
                return cls.raw.encode(text, add_special_tokens=add_special_tokens).ids

            def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
                assert clean_up_tokenization_spaces is False
                return cls.raw.decode(ids, skip_special_tokens=skip_special_tokens)

        cls.rows = calc_curriculum.train_rows()
        cls.compiled = calc_tokenization.compile_calc_train_tokens(
            cls.rows, calc_curriculum.source_commitment(), Adapter(),
            model_id=BASE_ID, model_revision=BASE_REVISION,
        )

    def test_all_1024_real_tokenizer_and_actual_trl_collator_labels(self):
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        examples = self.compiled["examples"]
        self.assertEqual(len(examples), 1024)
        serialized = json.dumps(
            [{"input_ids": x["input_ids"], "labels": x["labels"]} for x in examples],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        # Reproduced independently with pinned raw tokenizers 0.22.1 and with
        # the runtime-compatible 0.23.3; this fixes the 1,024-row parity claim.
        self.assertEqual(hashlib.sha256(serialized).hexdigest(),
                         "6fa2258c53bd65c642463575cd06706a693f51985c454ffef3ad897a8f860fb7")
        lengths = [len(example["input_ids"]) for example in examples]
        self.assertEqual((min(lengths), max(lengths)), (33, 42))
        self.assertTrue(all(0 < len(example["input_ids"]) <= 64 for example in examples))
        self.assertEqual(sum(row["answer"].startswith("-") for row in self.rows), 520)
        collator = DataCollatorForLanguageModeling(pad_token_id=248044)
        for index in range(0, 1024, 16):
            source = examples[index:index + 16]
            batch = collator(source)
            for j, example in enumerate(source):
                count = len(example["input_ids"])
                self.assertEqual(batch["input_ids"][j, :count].tolist(), example["input_ids"])
                self.assertEqual(batch["labels"][j, :count].tolist(), example["labels"])
                self.assertTrue(all(label == -100 for label in batch["labels"][j, count:].tolist()))
                self.assertEqual(batch["labels"][j, count - 1].item(), 248046)
                self.assertGreater(sum(label != -100 for label in example["labels"]), 1)
        print("REAL_TRL_QWEN_CALC_MASK_SOURCE_PASS count=1024 negatives=520 "
              "length_range=33-42 token_label_sha256="
              "6fa2258c53bd65c642463575cd06706a693f51985c454ffef3ad897a8f860fb7")

    def test_negative_legacy_split_detects_bpe_boundary_merges(self):
        collisions = 0
        for row in self.rows:
            legacy_prompt = f"Q: {row['prompt']}\nA: "
            legacy_answer = row["answer"]
            legacy_ids = self.raw.encode(legacy_prompt, add_special_tokens=False).ids
            legacy_ids += self.raw.encode(legacy_answer, add_special_tokens=False).ids
            canonical = self.raw.encode(legacy_prompt + legacy_answer,
                                        add_special_tokens=False).ids
            collisions += legacy_ids != canonical
        self.assertEqual(collisions, 520)

    def test_actual_sfttrainer_dataloader_retains_prebuilt_labels(self):
        import torch
        from datasets import Dataset
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from trl import SFTConfig, SFTTrainer

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"],
            eos_token="<|im_end|>", pad_token="<|endoftext|>",
        )
        self.assertEqual(tokenizer.eos_token_id, 248046)
        self.assertEqual(tokenizer.pad_token_id, 248044)
        self.assertFalse(torch.cuda.is_available())
        selected = self.compiled["examples"][:8]
        dataset = Dataset.from_list(selected)
        cfg = GPT2Config(vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=1,
                         n_positions=64, eos_token_id=248046, pad_token_id=248044)
        torch.manual_seed(42)
        model = GPT2LMHeadModel(cfg)  # random tiny CPU model; no weights fetched
        with TemporaryDirectory(prefix="trainingday-trl-mask-") as output_dir:
            args = SFTConfig(output_dir=output_dir,
                             max_length=64, per_device_train_batch_size=8,
                             dataset_kwargs={"skip_prepare_dataset": True},
                             completion_only_loss=True, report_to="none", seed=42,
                             packing=False, fp16=False, bf16=False,
                             dataloader_num_workers=0)
            trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                                 processing_class=tokenizer)
            batch = next(iter(trainer.get_train_dataloader()))
        observed = sorted(
            (batch["input_ids"][i, :int(batch["attention_mask"][i].sum())].tolist(),
             batch["labels"][i, :int(batch["attention_mask"][i].sum())].tolist())
            for i in range(8)
        )
        expected = sorted((example["input_ids"], example["labels"]) for example in selected)
        self.assertEqual(observed, expected)
        self.assertEqual(int((batch["labels"] != -100).sum()),
                         sum(sum(label != -100 for label in x["labels"]) for x in selected))

    def test_actual_sfttrainer_builds_labels_from_completion_masks(self):
        """Separate from preservation: actual trainer derives -100 from masks."""
        from datasets import Dataset
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from trl import SFTConfig, SFTTrainer

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"],
            eos_token="<|im_end|>", pad_token="<|endoftext|>",
        )
        # Include positive and negative arithmetic answers in the 8-row test.
        positives = [x for x, row in zip(self.compiled["examples"], self.rows)
                     if not row["answer"].startswith("-")][:4]
        negatives = [x for x, row in zip(self.compiled["examples"], self.rows)
                     if row["answer"].startswith("-")][:4]
        selected = positives + negatives
        data = [{"input_ids": x["input_ids"],
                 "completion_mask": [int(y != -100) for y in x["labels"]]}
                for x in selected]
        dataset = Dataset.from_list(data)  # intentionally NO `labels`
        config = GPT2Config(vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=1,
                            n_positions=64, eos_token_id=248046, pad_token_id=248044)
        with TemporaryDirectory(prefix="trainingday-trl-derived-") as output_dir:
            args = SFTConfig(output_dir=output_dir, max_length=64,
                             per_device_train_batch_size=8, completion_only_loss=True,
                             report_to="none", seed=42, packing=False,
                             fp16=False, bf16=False, dataloader_num_workers=0)
            trainer = SFTTrainer(model=GPT2LMHeadModel(config), args=args,
                                 train_dataset=dataset, processing_class=tokenizer)
            batch = next(iter(trainer.get_train_dataloader()))
        observed = sorted(
            (batch["input_ids"][i, :int(batch["attention_mask"][i].sum())].tolist(),
             batch["labels"][i, :int(batch["attention_mask"][i].sum())].tolist())
            for i in range(8)
        )
        self.assertEqual(observed,
                         sorted((example["input_ids"], example["labels"]) for example in selected))
        print("REAL_TRL_SFTTRAINER_DERIVED_COMPLETION_LABELS_PASS "
              "examples=8 positives=4 negatives=4 no_optimizer_steps=true")


if __name__ == "__main__":
    unittest.main()
