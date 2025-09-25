# Credit-Risk CoT with Verification

This repo trains a **small LLM**—`apple/OpenELM-450M`—to perform **credit & risk assessment** using **logical, numeric, and verifiable chain-of-thought (CoT)** on the **CALM** benchmark.  
We convert each tabular sample into a compact **STRUCTURED_COT** (feature contributions → sum-to-logit → sigmoid → threshold), then use a **tutor/verifier** to check the math and feed back **detailed errors** for a verification-in-the-loop (ViL) pass.

> small model + verified CoT ≈ big model performance on several CALM tasks, with auditable rationales.

---

## Contents

- `01_fetch_data.py` — Downloads all 9 CALM datasets from Hugging Face to `calm_openelm/data/<dataset>/`.
- `02_tutor.py` — Trains a transparent **logistic scorecard** on the train split (per dataset) and emits **STRUCTURED_COT** JSON for each split.
- `03_make_sft.py` — Builds **instruction-tuning** (chat) data that includes the numeric CoT and final choice.
- `04_sft_train.py` — Fine-tunes `apple/OpenELM-450M` with **QLoRA** using **Accelerate** (4 GPUs).
- `05_vil_iter.py` — **Verifier-in-the-loop** augmentation: generates CoT, checks arithmetic/probability/threshold consistency, and writes feedback examples.
- `06_eval.py` — Evaluates on test/validation splits: **Acc, F1, MCC, Miss** (+ **EOD/AOD** when a binary sensitive attribute is detectable).

Output folders:
- `calm_openelm/data/…` — cached raw splits + CoT jsonl + SFT jsonl
- `calm_openelm/artifacts/…` — scorecard params, fine-tuned model
- `calm_openelm/logs/…` — evaluation JSON, training logs

---

## Environment

```bash
conda create -n calm-openelm python=3.11 -y
conda activate calm-openelm

pip install -U "torch>=2.3" "transformers>=4.42" "datasets>=2.20" "accelerate>=0.33" \
  "peft>=0.11" bitsandbytes scikit-learn evaluate "numpy<2.0" pandas fastparquet \
  "huggingface_hub>=0.23" rich typer pydantic pyarrow ujson
````

If any dataset/model is gated on Hugging Face:

```bash
huggingface-cli login
```

GPU: the training/eval scripts assume **4 GPUs** and launch via `accelerate`.

---

## Datasets (from CALM)

All nine datasets load from the `daishen` namespace on Hugging Face:

* Credit Scoring: `daishen/cra-german`, `daishen/cra-australian`, `daishen/cra-lendingclub`
* Fraud Detection: `daishen/cra-ccf`, `daishen/cra-ccfraud`
* Financial Distress: `daishen/cra-polish`, `daishen/cra-taiwan`
* Insurance Claim Analysis: `daishen/cra-portoseguro`, `daishen/cra-travelinsurace` *(spelling as-is)*

> Note: CALM resamples extremely imbalanced sets to \~**1:2** (minority\:majority). The released splits already reflect their choices; we reuse those directly.

---

## Quickstart (end-to-end)

> Run these from the repo root. Paths in scripts are already set (e.g., `calm_openelm/data`).

### 1) Fetch all data

```bash
python 01_fetch_data.py
```

Writes Parquet + JSONL to `calm_openelm/data/<dataset>/`.

### 2) Train the tutor & generate numeric CoT

```bash
python 02_tutor.py
```

* Trains a **balanced logistic** scorecard per dataset (train split only).
* Saves `scorecard.json` to `calm_openelm/artifacts/<dataset>/`.
* Emits `cot_train.jsonl`, `cot_validation.jsonl`, `cot_test.jsonl` with **STRUCTURED\_COT** under each dataset folder.

### 3) Build instruction-tuning data (SFT)

```bash
python 03_make_sft.py
```

Creates `sft_{train,validation,test}.jsonl` files next to each CoT jsonl, using the pattern:

* system prompt,
* user prompt with `Choices=[…]` + original `Text:`,
* assistant JSON: `{"STRUCTURED_COT": …}`,
* assistant final answer: one of the provided `choices`.

### 4) Fine-tune OpenELM-450M (QLoRA, 4 GPUs)

```bash
accelerate launch --multi_gpu --num_processes 4 04_sft_train.py
```

Defaults:

* Base model: `apple/OpenELM-450M` (loaded in 4-bit).
* LoRA: `r=16, alpha=32, dropout=0.05` (attention + MLP proj modules).
* Seq len 2048; AdamW (8-bit); cosine LR 2e-4; warmup 3%; 3 epochs.
* Saves PEFT adapter + tokenizer to `calm_openelm/artifacts/openelm450m_cot_sft/`.

> To mirror CALM’s **generalization** setting, you can exclude `lendingclub`, `polish`, `portoseguro` from training (the script includes a line you can tweak).

### 5) (Optional) Verifier-in-the-loop augmentation

```bash
python 05_vil_iter.py
```

* Samples model outputs on training prompts.
* **Verifies** that `bias + sum(terms) == sum_logit`, `sigmoid(sum_logit) == probability`, and `prob ≥ τ ⇒ final_index`.
* Any failure generates a **feedback** message and is saved to `calm_openelm/data/_augmented_vil.jsonl`.

You can then append the augmented file to your SFT data and re-run step 4 for a short extra epoch.

### 6) Evaluate

```bash
accelerate launch --multi_gpu --num_processes 4 06_eval.py
```

* Runs greedy decoding to produce final choices.
* Logs per-dataset **Acc, F1, MCC, Miss** to `calm_openelm/logs/eval_results.json`.
* If a binary sensitive attribute is detectable (e.g., **foreign worker** in German, **female** in a fraud variant), also reports **EOD/AOD**.

---

## Method (short)

* **Parser → features:** deterministic regex turns `"key: value"` spans into numeric features; booleans → {0,1}, missing → 0 for scoring.
* **Tutor (transparent surrogate):** z-scores + balanced logistic regression. Pick threshold **τ** that maximizes **MCC** on train.
* **STRUCTURED\_COT JSON:** shows top per-feature contributions, bias, `sum_logit`, `sigmoid_prob`, `threshold_tau`, and the `final_choice`.
* **Verifier:** recomputes logit/prob/threshold decision and returns **detailed error strings** if any mismatch occurs.
* **Training:** Phase-I SFT on CoT + final answer; optional Phase-II verification-augmented SFT (feedback appended).

---

## Files & Artifacts

* `calm_openelm/data/<dataset>/train.parquet` — cached splits
* `calm_openelm/data/<dataset>/cot_*.jsonl` — numeric CoT per split
* `calm_openelm/data/<dataset>/sft_*.jsonl` — chat SFT data per split
* `calm_openelm/artifacts/<dataset>/scorecard.json` — tutor metadata
* `calm_openelm/artifacts/openelm450m_cot_sft/` — PEFT adapter & tokenizer
* `calm_openelm/logs/eval_results.json` — metrics per dataset

---

## Tips & Troubleshooting

* **HF auth**: if `load_dataset` or the base model errors with 401/403, run `huggingface-cli login`.
* **CUDA OOM**: lower `per_device_train_batch_size` or increase `gradient_accumulation_steps` in `04_sft_train.py`.
* **Tokenization length**: if truncation seems aggressive, reduce prompt verbosity or increase `max_length` (watch VRAM).
* **Generalization**: to replicate CALM’s held-out tests, exclude `lendingclub`, `polish`, and `portoseguro` from SFT train set.
* **Fairness**: the simple parser checks only a couple of attributes; extend `parse_sensitive_flags` in `06_eval.py` for your needs.

---

## Metrics

* **MCC**:

  $$\mathrm{MCC}=\frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
* **Miss**: $$FN / {y=1}$$
* **EOD/AOD**: computed when a binary sensitive attribute can be inferred from the text.

---

## Acknowledgments

* **CALM** benchmark and dataset cards.
* **OpenELM-450M** base model.
* Inspiration from **logical CoT + verifier** work in symbolic planning.

---

```
```
