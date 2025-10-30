# #!/usr/bin/env python3
# import os, re, glob, argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# PAT = re.compile(r".*/moe_e(?P<E>\d+)_rj(?P<RJ>[\d\.]+)_cf(?P<CF>[\d\.]+)_s(?P<S>\d+)/?$")

# def best_from_metrics(csv_path: str):
#     df = pd.read_csv(csv_path)
#     if "val_ppl" not in df.columns:
#         return np.nan, -1
#     valid = df.dropna(subset=["val_ppl"])
#     if len(valid)==0:
#         return np.nan, -1
#     idx = valid["val_ppl"].idxmin()
#     return float(valid.loc[idx, "val_ppl"]), int(valid.loc[idx, "step"])

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("root", type=str, help="root dir e.g., runs/2025-10-29_16-35-55")
#     ap.add_argument("--e", type=int, default=8, help="num_experts to filter (default: 8)")
#     ap.add_argument("--outfile", type=str, default="router_grid.png")
#     args = ap.parse_args()

#     rows = []
#     for path in sorted(glob.glob(os.path.join(args.root, "moe_e*_rj*_cf*_s*"))):
#         m = PAT.match(path)
#         if not m: 
#             continue
#         if int(m.group("E")) != args.e:
#             continue
#         rj = float(m.group("RJ")); cf = float(m.group("CF"))
#         csv = os.path.join(path, "metrics.csv")
#         if not os.path.exists(csv):
#             continue
#         best_ppl, best_step = best_from_metrics(csv)
#         rows.append({"RJ": rj, "CF": cf, "best_ppl": best_ppl, "best_step": best_step, "run": os.path.basename(path)})

#     if not rows:
#         print("No runs found for E=", args.e, "under", args.root)
#         return

#     df = pd.DataFrame(rows)
#     df_pivot = df.pivot(index="RJ", columns="CF", values="best_ppl").sort_index().sort_index(axis=1)
#     print("\n=== Best PPL grid (E=%d) ===" % args.e)
#     print(df_pivot.to_string(float_format=lambda x: f"{x:.2f}"))

#     out_csv = os.path.join(args.root, f"router_grid_e{args.e}.csv")
#     df_pivot.to_csv(out_csv)
#     print("Wrote", out_csv)

#     plt.figure(figsize=(6,4))
#     im = plt.imshow(df_pivot.values, aspect="auto", origin="lower")
#     plt.colorbar(im, label="best PPL (lower is better)")
#     plt.xticks(range(df_pivot.shape[1]), [str(c) for c in df_pivot.columns])
#     plt.yticks(range(df_pivot.shape[0]), [str(r) for r in df_pivot.index])
#     plt.xlabel("capacity_factor"); plt.ylabel("router_jitter")
#     plt.title(f"Best PPL Heatmap (E={args.e})")
#     for i in range(df_pivot.shape[0]):
#         for j in range(df_pivot.shape[1]):
#             v = df_pivot.values[i, j]
#             if np.isfinite(v):
#                 plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
#     plt.tight_layout()
#     out_png = os.path.join(args.root, args.outfile)
#     plt.savefig(out_png, dpi=150)
#     print("Wrote", out_png)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# all_moe_ と moe_ の両方を許容
PAT = re.compile(r".*/(?:all_)?moe_e(?P<E>\d+)_rj(?P<RJ>[\d\.]+)_cf(?P<CF>[\d\.]+)_s(?P<S>\d+)/?$")

def best_from_metrics(csv_path: str):
    df = pd.read_csv(csv_path)
    if "val_ppl" not in df.columns:
        return np.nan, -1
    valid = df.dropna(subset=["val_ppl"])
    if len(valid) == 0:
        return np.nan, -1
    idx = valid["val_ppl"].idxmin()
    return float(valid.loc[idx, "val_ppl"]), int(valid.loc[idx, "step"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="root dir e.g., runs/2025-10-29_16-35-55")
    ap.add_argument("--e", type=int, default=8, help="num_experts to filter (default: 8)")
    ap.add_argument("--outfile", type=str, default="router_grid.png")
    args = ap.parse_args()

    rows = []
    # all_moe_ と moe_ の両方を探索
    for pattern in ["all_moe_e*_rj*_cf*_s*", "moe_e*_rj*_cf*_s*"]:
        for path in sorted(glob.glob(os.path.join(args.root, pattern))):
            m = PAT.match(path)
            if not m:
                continue
            if int(m.group("E")) != args.e:
                continue
            rj = float(m.group("RJ"))
            cf = float(m.group("CF"))
            csv = os.path.join(path, "metrics.csv")
            if not os.path.exists(csv):
                continue
            best_ppl, best_step = best_from_metrics(csv)
            rows.append({
                "RJ": rj,
                "CF": cf,
                "best_ppl": best_ppl,
                "best_step": best_step,
                "run": os.path.basename(path)
            })

    if not rows:
        print("No runs found for E=", args.e, "under", args.root)
        return

    df = pd.DataFrame(rows)
    df_pivot = df.pivot(index="RJ", columns="CF", values="best_ppl").sort_index().sort_index(axis=1)
    print("\n=== Best PPL grid (E=%d) ===" % args.e)
    print(df_pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    out_csv = os.path.join(args.root, f"router_grid_e{args.e}.csv")
    df_pivot.to_csv(out_csv)
    print("Wrote", out_csv)

    plt.figure(figsize=(6, 4))
    im = plt.imshow(df_pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im, label="best PPL (lower is better)")
    plt.xticks(range(df_pivot.shape[1]), [str(c) for c in df_pivot.columns])
    plt.yticks(range(df_pivot.shape[0]), [str(r) for r in df_pivot.index])
    plt.xlabel("capacity_factor")
    plt.ylabel("router_jitter")
    plt.title(f"Best PPL Heatmap (E={args.e})")

    for i in range(df_pivot.shape[0]):
        for j in range(df_pivot.shape[1]):
            v = df_pivot.values[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(args.root, args.outfile)
    plt.savefig(out_png, dpi=150)
    print("Wrote", out_png)

if __name__ == "__main__":
    main()
