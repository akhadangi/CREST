
import os, json, math, random
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 2025
random.seed(SEED)

def verify_structured_cot(cot:Dict[str,Any]):
    ok=True; notes=[]
    try:
        bias = float(cot["zscore_logit_sum"]["bias"])
        terms = cot["zscore_logit_sum"]["terms"]  # list of [name, contrib]
        sum_terms = sum(float(t[1]) for t in terms)
        sum_logit = float(cot["zscore_logit_sum"]["sum_logit"])
        if abs((bias + sum_terms) - sum_logit) > 1e-3:
            ok=False; notes.append(f"Logit mismatch: bias+sum(terms)={bias+sum_terms:.4f} != {sum_logit:.4f}")
        p = 1/(1+math.exp(-sum_logit))
        if abs(p - float(cot["zscore_logit_sum"]["sigmoid_prob"])) > 1e-3:
            ok=False; notes.append("Sigmoid probability does not match sum_logit.")
        tau = float(cot["zscore_logit_sum"]["threshold_tau"])
        decision = 1 if p >= tau else 0
        if decision != int(cot["final_index"]):
            ok=False; notes.append("Final index does not match threshold comparison.")
    except Exception as e:
        ok=False; notes.append(f"Malformed STRUCTURED_COT: {e}")
    return ok, notes

def sample_structured_cot(tokenizer, model, prompt:str, max_new=256):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(**inputs, do_sample=False, max_new_tokens=max_new, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    # naive extract a JSON blob (first {...}) then the final choice (last token on the line)
    start = text.find("{")
    end = text.find("}") + 1
    resp_json = text[start:end]
    try:
        j = json.loads(resp_json)
    except Exception:
        j = {}
    # final choice is after JSON; find last line
    tail = text[end:].strip().splitlines()[-1].strip().strip("'").strip('"')
    return j.get("STRUCTURED_COT", {}), tail, text

def main():
    model_dir = "calm_openelm/artifacts/openelm450m_cot_sft"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
    model.eval()

    augmented = []
    for ds in ["german","australian","lendingclub","ccf","ccfraud","polish","taiwan","portoseguro","travelinsurace"]:
        path = Path(f"calm_openelm/data/{ds}/sft_train.jsonl")
        if not path.exists(): continue
        for line in path.read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            msgs = ex["messages"]
            # use the same user message, ask the model to respond
            user_msg = next(m for m in msgs if m["role"]=="user")["content"]
            prompt = f"<system>:You are precise, numeric, and honest.\n<user>:{user_msg}\n<assistant>:"
            cot, final_choice, full = sample_structured_cot(tokenizer, model, prompt)
            ok, notes = verify_structured_cot(cot)
            if not ok:
                feedback = "Detailed feedback: " + "; ".join(notes) + " Correct the STRUCTURED_COT and the final choice."
                augmented.append({
                    "messages":[
                        {"role":"system","content":"You are precise, numeric, and honest."},
                        {"role":"user","content": user_msg},
                        {"role":"assistant","content": json.dumps({"STRUCTURED_COT": cot}, ensure_ascii=False)},
                        {"role":"assistant","content": final_choice},
                        {"role":"user","content": feedback}
                    ]
                })
    out = Path("calm_openelm/data/_augmented_vil.jsonl")
    if augmented:
        out.write_text("\n".join(json.dumps(r) for r in augmented), encoding="utf-8")
        print(f"[OK] VIL augmented: {len(augmented)} -> {out}")
    else:
        print("[OK] Model produced fully verifiable chains on sampled set.")

if __name__ == "__main__":
    main()
