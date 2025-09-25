
from datasets import load_dataset, DatasetDict
from pathlib import Path
import json, os

HF_DATASETS = {
    "german":          "daishen/cra-german",
    "australian":      "daishen/cra-australian",
    "lendingclub":     "daishen/cra-lendingclub",
    "ccf":             "daishen/cra-ccf",
    "ccfraud":         "daishen/cra-ccfraud",
    "polish":          "daishen/cra-polish",
    "taiwan":          "daishen/cra-taiwan",
    "portoseguro":     "daishen/cra-portoseguro",
    "travelinsurace":  "daishen/cra-travelinsurace",  # (typo is in the HF id)
}

def dump_local(dsname:str, ds:DatasetDict, outdir:Path):
    outdir.mkdir(parents=True, exist_ok=True)
    for split, d in ds.items():
        # Persist to Arrow/JSON for transparency
        d.to_parquet(outdir/f"{split}.parquet")
        (outdir/f"{split}.jsonl").write_text("\n".join(map(json.dumps, d.to_list())), encoding="utf-8")
    print(f"[OK] Saved {dsname} to {outdir}")

def main():
    root = Path("calm_openelm/data")
    for name, hf_id in HF_DATASETS.items():
        print(f"==> Loading {name} ({hf_id})")
        try:
            ds = load_dataset(hf_id)
        except Exception:
            # Some datasets only expose a single split (e.g., test-only)
            ds = DatasetDict({"test": load_dataset(hf_id, split="test")})
        dump_local(name, ds, root/name)

if __name__ == "__main__":
    main()
