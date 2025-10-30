#!/usr/bin/env bash
set -euo pipefail

# ================================
# Baseline vs All-layers MoE + Router Grid
# - 実行するだけで一式まわせる統合スクリプト
# - QUICK=1 で高速サニティラン
# - 出力: runs/<timestamp>/...
# ================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON=${PYTHON:-python}

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

# ============ 実行モード ============
# QUICK=1 を環境変数に設定すると短時間ラン（steps縮小・グリッド縮小）
QUICK="${QUICK:-0}"

# 既存フォルダを引数に渡すと集計のみ（学習スキップ）
if [ $# -ge 1 ]; then
  ROOT="$1"; NEW_RUN=0
else
  ROOT="runs/$(timestamp)"; mkdir -p "${ROOT}"; NEW_RUN=1
fi

echo "[ROOT] ${ROOT}"
echo "[GPU ]"; nvidia-smi || true
echo "[TIME] $(date)"

# ============ 共通ハイパラ ============
D_MODEL=512
N_LAYER=6
N_HEAD=8
SEQ_LEN=256
BATCH=16
STEPS=10000
EVAL_EVERY=200
VAL_BATCHES=50
BF16="--bf16"   # bfloat16で実行（不要なら空に）
SEED=42

# 早期停止
ES_PATIENCE=5
ES_DELTA=0.1
ES_WARMUP=400
MAX_TIME_MIN=

# Sweep（通常モード）
EXPERTS=("4" "8" "16")
RJ_LIST=("0.0" "0.01" "0.05")
CF_LIST=("1.0" "1.25" "1.5")

# QUICKモード上書き
if [ "${QUICK}" -eq 1 ]; then
  echo "[MODE] QUICK=1 (short sanity run)"
  STEPS=1500
  EVAL_EVERY=200
  VAL_BATCHES=20
  EXPERTS=("8")
  RJ_LIST=("0.0")
  CF_LIST=("1.5")
fi

run_job () {
  local OUT="$1"; shift
  local CMD="$*"

  echo "------------------------------------------------------------"
  echo ">> RUN: ${OUT}"
  echo "------------------------------------------------------------"
  mkdir -p "${OUT}"
  ${PYTHON} "${REPO_ROOT}/train_tinystories.py" ${CMD} \
    --out_dir "${OUT}" \
    --eval_every ${EVAL_EVERY} --val_batches ${VAL_BATCHES} \
    --early_stop_patience ${ES_PATIENCE} --early_stop_min_delta ${ES_DELTA} \
    --early_stop_warmup ${ES_WARMUP} \
    --seed ${SEED} \
    ${MAX_TIME_MIN:+--max_time_min ${MAX_TIME_MIN}}
}

if [ ${NEW_RUN} -eq 1 ]; then
  echo "[MODE] train + summarize"

  # ===== Baseline（MoEなし）
  BASE_OUT="${ROOT}/baseline_s${STEPS}"
  run_job "${BASE_OUT}" \
    --no_moe \
    --d_model ${D_MODEL} --n_layer ${N_LAYER} --n_head ${N_HEAD} --seq_len ${SEQ_LEN} \
    --batch_size ${BATCH} --steps ${STEPS} ${BF16}

  # ===== 全層MoE：Experts スイープ
  for E in "${EXPERTS[@]}"; do
    OUT="${ROOT}/all_moe_e${E}_s${STEPS}"
    run_job "${OUT}" \
      --moe_all_layers \
      --moe_num_experts ${E} \
      --d_model ${D_MODEL} --n_layer ${N_LAYER} --n_head ${N_HEAD} --seq_len ${SEQ_LEN} \
      --batch_size ${BATCH} --steps ${STEPS} ${BF16}
  done

  # ===== 全層MoE：Router グリッド（E=8 固定）
  E=8
  for RJ in "${RJ_LIST[@]}"; do
    for CF in "${CF_LIST[@]}"; do
      OUT="${ROOT}/all_moe_e${E}_rj${RJ}_cf${CF}_s${STEPS}"
      run_job "${OUT}" \
        --moe_all_layers \
        --moe_num_experts ${E} \
        --moe_router_jitter ${RJ} --moe_capacity_factor ${CF} \
        --d_model ${D_MODEL} --n_layer ${N_LAYER} --n_head ${N_HEAD} --seq_len ${SEQ_LEN} \
        --batch_size ${BATCH} --steps ${STEPS} ${BF16}
    done
  done
else
  echo "[MODE] summarize-only (no training)"
fi

# ===== 集計 & 可視化 =====
echo "[Summarize -> ${ROOT}/summary.csv]"
${PYTHON} "${REPO_ROOT}/tools/summarize.py" "${ROOT}" || true

echo "[Plot PPL curves -> ${ROOT}/ppl_curves_all.png]"
${PYTHON} "${REPO_ROOT}/tools/plot_ppl.py" "${ROOT}" "ppl_curves_all.png" || true

echo "[Heatmap (E=8, all-layers) -> ${ROOT}/router_grid.png]"
${PYTHON} "${REPO_ROOT}/tools/summarize_grid.py" "${ROOT}" --e 8 || true

echo "[DONE] $(date)"
echo "[RESULTS] ${ROOT}"
