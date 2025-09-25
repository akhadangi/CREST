
import os, json, math, random
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

def load_sft_jsonl(paths: List[Path]):
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            yield json.loads(line)

class ChatDataset(Dataset):
    def __init__(self, files:List[Path], tokenizer, max_len=2048):
        self.samples = list(load_sft_jsonl(files))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def _format(self, msgs):
        # Simple chat concat. You can adopt ChatML if desired.
        text = ""
        for m in msgs:
            role = m["role"]
            text += f"<{role}>: {m['content'].strip()}\n"
        text += "<assistant>:"
        return text

    def __getitem__(self, idx):
        msgs = self.samples[idx]["messages"]
        prompt = self._format(msgs[:-1])
        target = msgs[-1]["content"].strip()
        full = prompt + " " + target
        toks = self.tok(
            full, truncation=True, max_length=self.max_len, return_tensors=None
        )
        labels = toks["input_ids"].copy()
        # Mask the prompt portion so loss only on assistant final answer? Here we keep full supervision.
        return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"], "labels": labels}

def main():
    model_id = "apple/OpenELM-450M"
    out_dir = "calm_openelm/artifacts/openelm450m_cot_sft"
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit QLoRA to keep memory low
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none"
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Build datasets (union of all sft_train.jsonl files)
    root = Path("calm_openelm/data")
    train_files, eval_files = [], []
    for ds in ["german","australian","lendingclub","ccf","ccfraud","polish","taiwan","portoseguro","travelinsurace"]:
        t = root/ds/"sft_train.jsonl"
        v = root/ds/"sft_validation.jsonl"
        if t.exists(): train_files.append(t)
        if v.exists(): eval_files.append(v)
    train_ds = ChatDataset(train_files, tokenizer, max_len=2048)
    eval_ds  = ChatDataset(eval_files, tokenizer, max_len=2048) if eval_files else None

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=50,
        save_steps=1000,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=500 if eval_ds else None,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
