#!/usr/bin/env python3
import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt

def load_curve(csv_path):
    df = pd.read_csv(csv_path)
    if "step" not in df.columns or "val_ppl" not in df.columns:
        return None, None
    df = df.dropna(subset=["val_ppl"])
    return df["step"].values, df["val_ppl"].values

def main(root="runs", outfile="ppl_curves.png", prefix=None):
    run_dirs = sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])
    if prefix:
        run_dirs = [d for d in run_dirs if os.path.basename(d).startswith(prefix)]
    found = 0
    plt.figure()
    for rd in run_dirs:
        csv = os.path.join(rd, "metrics.csv")
        if not os.path.exists(csv): 
            continue
        step, ppl = load_curve(csv)
        if step is None: 
            continue
        label = os.path.basename(rd)
        plt.plot(step, ppl, label=label); found += 1
    if found == 0:
        print("No curves found under", root, "with prefix", prefix or "<any>")
        return
    plt.xlabel("step"); plt.ylabel("validation perplexity (lower is better)")
    plt.title("Validation PPL Curves")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, outfile)
    plt.savefig(path, dpi=150)
    print("Saved plot to", path)

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "ppl_curves.png"
    prefix = sys.argv[3] if len(sys.argv) > 3 else None
    main(root, outfile, prefix)
