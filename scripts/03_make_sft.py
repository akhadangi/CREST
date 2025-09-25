import json
from pathlib import Path

INSTRUCT = (
"You're a risk analyst. Parse the applicant's features from the text, compute a linear score using z-scored features, "
"show the top contributions and the exact logit sum and sigmoid probability in a JSON object named STRUCTURED_COT, "
"then output only the final choice from the provided choices. Keep the JSON compact and strictly valid."
)

def convert_one_file(inpath:Path, outpath:Path):
    out = []
    for line in inpath.read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        cot = ex["structured_cot"]
        messages = [
            {"role":"system","content":"You are precise, numeric, and honest."},
            {"role":"user","content": f"{INSTRUCT}\n\nChoices={ex['choices']}\n\nText:\n{ex['input']}"},
            {"role":"assistant","content": json.dumps({"STRUCTURED_COT": cot}, ensure_ascii=False)},
            {"role":"assistant","content": cot["final_choice"]},
        ]
        out.append({"messages": messages})
    outpath.write_text("\n".join(json.dumps(r) for r in out), encoding="utf-8")

def main():
    root = Path("calm_openelm/data")
    for ds in ["german","australian","lendingclub","ccf","ccfraud","polish","taiwan","portoseguro","travelinsurace"]:
        for split in ["train","validation","test"]:
            f = root/ds/f"cot_{split}.jsonl"
            if f.exists():
                convert_one_file(f, root/ds/f"sft_{split}.jsonl")
                print("[OK]", ds, split)

if __name__ == "__main__":
    main()
