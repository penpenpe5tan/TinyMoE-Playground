#!/usr/bin/env python3
import os, sys, glob, math
import pandas as pd

def summarize_dir(run_dir):
    m = os.path.join(run_dir, "metrics.csv")
    row = {"run": os.path.basename(run_dir), "final_val_ppl": float("nan"),
           "best_val_ppl": float("nan"), "best_step": float("nan"),
           "last_train_loss": float("nan"), "max_tok_s": float("nan"),
           "peak_mem_mb": float("nan")}
    if not os.path.exists(m):
        return row
    df = pd.read_csv(m)
    if "train_loss" in df:
        row["last_train_loss"] = float(pd.to_numeric(df["train_loss"], errors="coerce").iloc[-1])
    if "val_ppl" in df:
        dff = df[pd.to_numeric(df["val_ppl"], errors="coerce").apply(math.isfinite)]
        if not dff.empty:
            row["final_val_ppl"] = float(dff["val_ppl"].iloc[-1])
            # best
            idx = dff["val_ppl"].idxmin()
            row["best_val_ppl"] = float(dff.loc[idx, "val_ppl"])
            row["best_step"] = float(df.loc[idx, "step"])
    if "tokens_per_sec" in df:
        row["max_tok_s"] = float(pd.to_numeric(df["tokens_per_sec"], errors="coerce").max())
    if "gpu_mem_mb" in df:
        row["peak_mem_mb"] = float(pd.to_numeric(df["gpu_mem_mb"], errors="coerce").max())
    return row

def main(root="runs"):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d):
            rows.append(summarize_dir(d))
    if not rows:
        print("No runs found under", root); return
    df = pd.DataFrame(rows)
    # baseline 比改善率
    try:
        base = df[df["run"].str.contains("baseline", na=False)]
        if not base.empty and math.isfinite(float(base["final_val_ppl"].iloc[-1])):
            base_ppl = float(base["final_val_ppl"].iloc[-1])
            df["ppl_improve_vs_base_%"] = (base_ppl - df["final_val_ppl"]) / base_ppl * 100.0
    except Exception:
        pass
    print(df.to_string(index=False))
    out = os.path.join(root, "summary.csv")
    df.to_csv(out, index=False)
    print("\nWrote", out)

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    main(root)
