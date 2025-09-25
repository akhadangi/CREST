
import json, math, re
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_choice(text, choices):
    text = text.strip().lower()
    # find exact match in choices (case-insensitive)
    for c in choices:
        if text.endswith(c.lower()) or text == c.lower():
            return c
    # fallback: grab 'yes/no/good/bad' heuristically
    for c in ["yes","no","good","bad"]:
        if c in text.lower().split():
            return c
    return choices[0]

def generate_label(tokenizer, model, query, text, choices):
    prompt = (
        "You are precise. Emit a JSON named STRUCTURED_COT computing the numeric score as described; "
        "then output ONLY the final choice.\n\n"
        f"Choices={choices}\n\nText:\n{text}\n"
    )
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, do_sample=False, max_new_tokens=192, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_choice(decoded, choices), decoded

def miss_rate(y_true, y_pred):
    # Miss: false negatives over positive class count (if positive is index 1)
    y_true = np.array(y_true); y_pred=np.array(y_pred)
    pos = (y_true==1)
    if pos.sum()==0: return 0.0
    fn = ((y_pred==0) & pos).sum()
    return fn/pos.sum()

def fairness_metrics(y_true, y_pred, sens):
    # sens is a binary indicator array for unprivileged vs privileged (e.g., female vs male, foreign worker vs not)
    # DI is a data metric (not applicable here post-prediction); report EOD/AOD for preds
    def rate(y, s, label):
        return ((y==label) & (s==1)).sum() / max(1, (s==1).sum())
    # True positive rates per group
    tpr_un = ((y_pred==1) & (y_true==1) & (sens==1)).sum() / max(1, ((y_true==1) & (sens==1)).sum())
    tpr_pr = ((y_pred==1) & (y_true==1) & (sens==0)).sum() / max(1, ((y_true==1) & (sens==0)).sum())
    fpr_un = ((y_pred==1) & (y_true==0) & (sens==1)).sum() / max(1, ((y_true==0) & (sens==1)).sum())
    fpr_pr = ((y_pred==1) & (y_true==0) & (sens==0)).sum() / max(1, ((y_true==0) & (sens==0)).sum())
    eod = tpr_un - tpr_pr
    aod = 0.5*((tpr_un - tpr_pr) + (fpr_un - fpr_pr))
    return {"EOD": float(eod), "AOD": float(aod)}

def parse_sensitive_flags(dsname, text):
    t = text.lower()
    if dsname=="german":
        # foreign worker yes/no; age in years
        fw = 1 if "foreign_worker_is_yes" in t or "foreign worker is yes" in t else 0
        return {"foreign_worker": fw}
    if dsname=="ccfraud":
        # "The client is a female/male"
        female = 1 if "the client is a female" in t else 0
        return {"female": female}
    if dsname=="travelinsurace":
        # Age may exist depending on card; often not; set empty
        return {}
    return {}

def main():
    model_dir = "calm_openelm/artifacts/openelm450m_cot_sft"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
    model.eval()

    HF = {
        "german": "daishen/cra-german",
        "australian": "daishen/cra-australian",
        "lendingclub": "daishen/cra-lendingclub",
        "ccf": "daishen/cra-ccf",
        "ccfraud": "daishen/cra-ccfraud",
        "polish": "daishen/cra-polish",
        "taiwan": "daishen/cra-taiwan",
        "portoseguro": "daishen/cra-portoseguro",
        "travelinsurace": "daishen/cra-travelinsurace",
    }

    results = {}
    for name, hf in HF.items():
        try:
            ds = load_dataset(hf, split="test")
        except:
            # some datasets only have test; otherwise use validation
            try:
                ds = load_dataset(hf, split="validation")
            except:
                continue
        y_true, y_pred, decodes = [], [], []
        sens_flags = []
        for ex in ds:
            choices = ex["choices"] if "choices" in ex else ["no","yes"]
            pred, decoded = generate_label(tokenizer, model, ex["query"], ex["text"], choices)
            pred_idx = choices.index(pred)
            gold = ex["gold"] if "gold" in ex and ex["gold"] in (0,1) else choices.index(ex["answer"])
            y_true.append(int(gold))
            y_pred.append(int(pred_idx))
            decodes.append(decoded)
            sens = parse_sensitive_flags(name, ex["text"])
            sens_flags.append(next(iter(sens.values())) if sens else -1)
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="binary")
        mcc = matthews_corrcoef(y_true, y_pred)
        miss = miss_rate(y_true, y_pred)
        metrics = {"acc": float(acc), "f1": float(f1), "mcc": float(mcc), "miss": float(miss)}
        # fairness if we have binary sensitive attr
        sf = np.array(sens_flags)
        if (sf!=-1).any():
            fm = fairness_metrics(np.array(y_true), np.array(y_pred), (sf==1).astype(int))
            metrics.update(fm)
        results[name] = metrics
        print(name, metrics)
    Path("calm_openelm/logs/eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
