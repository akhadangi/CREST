
from __future__ import annotations
import re, json, math, random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

R_FLOAT = re.compile(r"([-+]?\d*\.\d+|[-+]?\d+)")
# key: either token: or token (no colon) followed by number; allow spaces, slashes, parentheses, % etc.
R_PAIR = re.compile(r"([A-Za-z0-9_\-\(\)\/\*\+\s\%\.']+):\s*([-+]?\d*\.\d+|[-+]?\d+|yes|no|None)", re.I)

YESNO = {"yes": 1.0, "no": 0.0}

# Simple mapping from dataset id to the positive label string in `choices`
POSITIVE_LABEL_BY_SET = {
    "german": "good",
    "australian": "good",
    "lendingclub": "yes",   # lending club often uses 'yes/no' for approve/deny in CALM card
    "ccf": "yes",
    "ccfraud": "bad",       # card uses 'good'/'bad' text; positive might be 'bad' (fraud). We'll infer from 'gold' too.
    "polish": "yes",
    "taiwan": "yes",
    "portoseguro": "yes",
    "travelinsurace": "yes",
}

def parse_text_to_features(text:str) -> Dict[str, Any]:
    # Extract "key: value" tokens; convert yes/no; leave non-numeric aside
    feats: Dict[str, Any] = {}
    for m in R_PAIR.finditer(text):
        k = m.group(1).strip().lower().replace(" ", "_")
        v = m.group(2).strip()
        if v.lower() in YESNO:
            feats[k] = float(YESNO[v.lower()])
        elif v == "None":
            feats[k] = np.nan
        else:
            try:
                feats[k] = float(v)
            except ValueError:
                pass
    # Also capture bare A1: 1.0 style already handled; if nothing parsed, fallback: extract all floats as indexed features
    if not feats:
        floats = [float(x) for x in R_FLOAT.findall(text)]
        feats = {f"f{i}": val for i, val in enumerate(floats)}
    return feats

def load_local(name:str) -> DatasetDict:
    droot = Path("calm_openelm/data")/name
    if (droot/"train.parquet").exists() or (droot/"train.jsonl").exists():
        # loaded earlier
        try:
            # Try reading the parquet as dataset
            ds = DatasetDict({
                split: load_dataset("parquet", data_files=str((droot/f"{split}.parquet").resolve()))[split]
                for split in ["train","validation","test"] if (droot/f"{split}.parquet").exists()
            })
            # test-only datasets
            if not ds:
                ds = DatasetDict({"test": load_dataset("parquet", data_files=str((droot/"test.parquet").resolve()))["train"]})
            return ds
        except Exception:
            pass
    # fallback: reload from hub
    HF_DATASETS = {
        "german":          "daishen/cra-german",
        "australian":      "daishen/cra-australian",
        "lendingclub":     "daishen/cra-lendingclub",
        "ccf":             "daishen/cra-ccf",
        "ccfraud":         "daishen/cra-ccfraud",
        "polish":          "daishen/cra-polish",
        "taiwan":          "daishen/cra-taiwan",
        "portoseguro":     "daishen/cra-portoseguro",
        "travelinsurace":  "daishen/cra-travelinsurace",
    }
    return load_dataset(HF_DATASETS[name])

@dataclass
class Scorecard:
    pipe: Pipeline
    feat_names: List[str]
    threshold: float  # threshold on predicted prob for positive class

    def explain(self, row: Dict[str, float], topk:int=10) -> Dict[str, Any]:
        # Standard logistic regression: prob = sigmoid(w · x + b)
        X = np.array([[row.get(k, 0.0) for k in self.feat_names]])
        # get contributions by taking scaler & coef into account
        lr: LogisticRegression = self.pipe[-1]
        scaler: StandardScaler = self.pipe[0]
        x_scaled = scaler.transform(X)
        coef = lr.coef_[0]
        bias = lr.intercept_[0]
        logits = float(np.dot(x_scaled[0], coef) + bias)
        prob = 1.0 / (1.0 + math.exp(-logits))
        contribs = {k: float(coef[i] * x_scaled[0, i]) for i,k in enumerate(self.feat_names)}
        # top features by |contribution|
        top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:topk]
        decision = 1 if prob >= self.threshold else 0
        return {
            "features_used": self.feat_names,
            "top_contribs": top,
            "bias": float(bias),
            "logit": logits,
            "probability": prob,
            "threshold": self.threshold,
            "decision": decision,
        }

def train_scorecard(df: pd.DataFrame, y: np.ndarray) -> Tuple[Scorecard, Dict[str, float]]:
    # Keep numeric columns only; fill NaNs; z-score
    num_cols = [c for c in df.columns if df[c].dtype != "O"]
    X = df[num_cols].fillna(0.0).values
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(X, y)
    # choose threshold to maximize MCC on train
    proba = pipe.predict_proba(X)[:,1]
    candidates = np.linspace(0.2, 0.8, 25)
    best_tau, best_mcc = 0.5, -2
    for tau in candidates:
        mcc = matthews_corrcoef(y, (proba>=tau).astype(int))
        if mcc > best_mcc:
            best_tau, best_mcc = tau, mcc
    card = Scorecard(pipe=pipe, feat_names=num_cols, threshold=float(best_tau))
    return card, {"mcc_train": float(best_mcc)}

def build_frame(split):
    # Each example has fields: query (prompt), text (input), choices (list), answer (string), gold (index)
    # We'll parse `text` to features, keep `choices` and `answer`.
    feats, ys, raw = [], [], []
    for ex in split:
        f = parse_text_to_features(ex["text"])
        feats.append(f)
        raw.append(ex)
        # label index is often in 'gold', else map via 'choices'
        if "gold" in ex and ex["gold"] in (0,1):
            ys.append(int(ex["gold"]))
        else:
            # fallback: answer index inside choices
            ys.append(int(ex["choices"].index(ex["answer"])))
    # Harmonize columns (union of keys)
    all_keys = sorted({k for d in feats for k in d.keys()})
    df = pd.DataFrame([{k: d.get(k, np.nan) for k in all_keys} for d in feats])
    y = np.array(ys, dtype=int)
    return df, y, raw

def make_structured_cot(raw_ex:Dict[str,Any], card:Scorecard, df_row:pd.Series)->Dict[str,Any]:
    # Compose a compact, numeric, verifiable CoT
    row = {k: float(df_row[k]) if pd.notna(df_row[k]) else 0.0 for k in card.feat_names}
    expl = card.explain(row, topk=10)
    # Ensure the final textual label aligns with example's 'choices'
    pos_idx = 1  # by CALM convention gold 1 is often "bad"/"yes"; we stick with gold index at training time
    final_idx = int(expl["decision"])
    return {
        "structured_cot": {
            "parse_ok": True,
            "row_excerpt": {k: row[k] for k,_ in expl["top_contribs"]},
            "zscore_logit_sum": {
                "bias": expl["bias"],
                "terms": expl["top_contribs"],
                "sum_logit": expl["logit"],
                "sigmoid_prob": expl["probability"],
                "threshold_tau": expl["threshold"],
            },
            "final_index": final_idx,
            "final_choice": raw_ex["choices"][final_idx] if "choices" in raw_ex else None
        }
    }

def process_dataset(name:str):
    ds = load_local(name)
    # Many sets have {train, validation, test}; some have only test
    train_split = ds["train"] if "train" in ds else None
    val_split   = ds["validation"] if "validation" in ds else None
    test_split  = ds["test"] if "test" in ds else (ds["train"] if "train" in ds else None)

    # Build tutor on *train only* (avoid leakage)
    if train_split is None:
        print(f"[WARN] {name} has no train split; skipping tutor training.")
        return

    Xtr, ytr, raw_tr = build_frame(train_split)
    card, metrics = train_scorecard(Xtr, ytr)
    print(f"[{name}] trained scorecard; best MCC(train)={metrics['mcc_train']:.3f}, tau={card.threshold:.3f}")

    # Save tutor
    out = Path("calm_openelm/artifacts")/name
    out.mkdir(parents=True, exist_ok=True)
    with open(out/"scorecard.json","w") as f:
        json.dump({"feat_names": card.feat_names, "tau": card.threshold}, f, indent=2)

    # Generate numeric CoT for all splits available
    def annotate(split, split_name):
        if split is None: return
        X, y, raw = build_frame(split)
        cot_rows = []
        for i in range(len(raw)):
            cot = make_structured_cot(raw[i], card, X.iloc[i])
            packed = {
                "query": raw[i]["query"],
                "input": raw[i]["text"],
                "choices": raw[i].get("choices", ["no","yes"]),
                "answer": raw[i]["answer"],
                "gold": raw[i].get("gold", None),
                **cot
            }
            cot_rows.append(packed)
        Path(f"calm_openelm/data/{name}/cot_{split_name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in cot_rows), encoding="utf-8"
        )
        print(f"[OK] wrote {name} {split_name} with CoT: {len(cot_rows)}")

    annotate(train_split, "train")
    annotate(val_split,   "validation")
    annotate(test_split,  "test")

if __name__ == "__main__":
    for n in ["german","australian","lendingclub","ccf","ccfraud","polish","taiwan","portoseguro","travelinsurace"]:
        process_dataset(n)
