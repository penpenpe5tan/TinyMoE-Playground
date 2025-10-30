# ──────────────── train_tinystories.py ────────────────
import argparse
from dataclasses import dataclass, asdict
import os, json, time, math
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tiktoken

from moe import GPTMoE

@dataclass
class TrainConfig:
    # 既存の項目はそのまま
    vocab_name: str = 'gpt2'
    d_model: int = 512
    n_layer: int = 6
    n_head: int = 8
    seq_len: int = 256
    d_ff: int | None = None
    pdrop: float = 0.0

    # 既存
    moe_layer_index: int | None = 3
    moe_num_experts: int = 8
    moe_capacity_factor: float | None = None
    moe_router_jitter: float = 0.0

    # ★ここを追加
    moe_all_layers: bool = False         # 全層をMoE化するフラグ
    moe_layers: str | None = None        # "0,2,5" みたいなカンマ区切り指定

    batch_size: int = 16
    steps: int = 1500
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = False
    grad_ckpt: bool = False
    device: str = 'cuda'
    dataset_name: str = 'roneneldan/TinyStories'
    split: str = 'train'
    out_dir: str = 'checkpoints/tinystories'
    save_every: int = 0
    max_samples: int | None = 200_000
    eval_every: int = 200
    val_batches: int = 50
    log_csv: bool = True
    no_moe: bool = False
    seed: int = 42
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    early_stop_warmup: int = 400
    target_val_ppl: float | None = None
    max_time_min: int | None = None

class TinyStoriesTokens(Dataset):
    def __init__(self, enc: tiktoken.Encoding, seq_len: int,
                 split: str = 'train', max_samples: int | None = None):
        ds = load_dataset('roneneldan/TinyStories', split=split)
        texts = (ex['text'] for ex in ds)
        if max_samples is not None:
            texts = (ds[i]['text'] for i in range(min(max_samples, len(ds))))
        ids = []
        eos_id = enc.eot_token if hasattr(enc, 'eot_token') and enc.eot_token is not None else enc.encode('\n')[0]
        for t in texts:
            ids.extend(enc.encode(t))
            ids.append(eos_id)
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.ids) // self.seq_len - 1
    def __getitem__(self, i):
        s = i * self.seq_len
        x = self.ids[s:s+self.seq_len]
        y = self.ids[s+1:s+1+self.seq_len]
        return x, y


def build_model(enc, cfg):
    vocab_size = enc.n_vocab

    # no_moe のときは一切MoEを使わない
    if cfg.no_moe or cfg.moe_num_experts == 0:
        use_all = False
        layers_set = None
        layer_index = None
        num_experts = 0
    else:
        use_all = cfg.moe_all_layers
        # "0,2,5" → {0,2,5}
        layers_set = None
        if cfg.moe_layers:
            layers_set = set(int(s) for s in cfg.moe_layers.split(',') if s.strip() != '')
        layer_index = cfg.moe_layer_index
        num_experts = cfg.moe_num_experts

    return GPTMoE(
        vocab_size=vocab_size,
        d_model=cfg.d_model, n_layer=cfg.n_layer, n_head=cfg.n_head,
        seq_len=cfg.seq_len, d_ff=cfg.d_ff,
        moe_layer_index=layer_index,
        moe_layers=layers_set,
        moe_all_layers=use_all,
        moe_num_experts=num_experts,
        moe_capacity_factor=cfg.moe_capacity_factor,
        moe_router_jitter=cfg.moe_router_jitter,
        pdrop=cfg.pdrop
    )

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--d_ff', type=int, default=None)
    p.add_argument('--pdrop', type=float, default=0.0)

    p.add_argument('--moe_layers', type=str, default='', 
                help='MoE化する層のインデックスをカンマ区切りで指定。例: "0,1,2"。空文字なら単層/旧指定にフォールバック')
    p.add_argument('--moe_all_layers', action='store_true',
                help='全ブロック（0..n_layer-1）をMoE化する')

    # 既存の --moe_layer_index は使わなくなるが、互換のため残してもOK
    p.add_argument('--moe_layer_index', type=int, default=3,
                help='後方互換：単層MoEのときのブロック番号（0始まり）')




    p.add_argument('--moe_num_experts', type=int, default=8)
    p.add_argument('--moe_capacity_factor', type=float, default=None)
    p.add_argument('--moe_router_jitter', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--grad_ckpt', action='store_true')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--max_samples', type=int, default=200_000)
    p.add_argument('--out_dir', type=str, default='checkpoints/tinystories')
    p.add_argument('--save_every', type=int, default=0)
    p.add_argument('--eval_every', type=int, default=200)
    p.add_argument('--val_batches', type=int, default=50)
    p.add_argument('--no_moe', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--early_stop_patience', type=int, default=5, help='何回の評価で改善が無ければ停止')
    p.add_argument('--early_stop_min_delta', type=float, default=0.0, help='改善とみなす最小差分（PPL）')
    p.add_argument('--early_stop_warmup', type=int, default=400, help='このステップまでは停止判定しない')
    p.add_argument('--target_val_ppl', type=float, default=None, help='到達したら即停止する目標PPL')
    p.add_argument('--max_time_min', type=int, default=None, help='分での時間上限（超えたら停止）')
    
    args = p.parse_args()
    return TrainConfig(**vars(args))


def maybe_grad_ckpt_block(module: nn.Module, enable: bool):
    if not enable:
        return module
    def _checkpoint_forward(*inputs):
        def run_fn(*x):
            return module(*x)
        return torch.utils.checkpoint.checkpoint(run_fn, *inputs, use_reentrant=False)
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = module
        def forward(self, *x):
            return _checkpoint_forward(*x)
    return Wrapper()


def generate(model: GPTMoE, enc: tiktoken.Encoding, prompt: str, max_new_tokens=100, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _, _ = model(ids[:, -model.seq_len:])
            logits = logits[:, -1, :]
            logits = logits / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return enc.decode(ids[0].tolist())


def save_checkpoint(out_dir: str, model: GPTMoE, enc_name: str, cfg: TrainConfig, step: int):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'pytorch_model.bin')
    torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
    meta = {
        'enc_name': enc_name,
        'arch': {
            'd_model': cfg.d_model, 'n_layer': cfg.n_layer, 'n_head': cfg.n_head,
            'seq_len': cfg.seq_len, 'd_ff': cfg.d_ff,
            'moe_layer_index': cfg.moe_layer_index, 'moe_num_experts': cfg.moe_num_experts,
            'moe_capacity_factor': cfg.moe_capacity_factor, 'moe_router_jitter': cfg.moe_router_jitter,
            'pdrop': cfg.pdrop
        },
        'step': step
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"[saved] step={step} -> {ckpt_path}")


def main():
    cfg = parse_args()
    print("[config]", cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    enc = tiktoken.get_encoding(cfg.vocab_name)
    model = build_model(enc, cfg).to(device)

    if cfg.grad_ckpt:
        for i, blk in enumerate(model.blocks):
            model.blocks[i] = maybe_grad_ckpt_block(blk, True)

    torch.manual_seed(cfg.seed)

    # ===== datasets / loaders =====
    train_ds = TinyStoriesTokens(enc, cfg.seq_len, split='train',      max_samples=cfg.max_samples)
    val_cap  = 20_000 if cfg.max_samples is None else min(20_000, cfg.max_samples)
    val_ds   = TinyStoriesTokens(enc, cfg.seq_len, split='validation', max_samples=val_cap)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=True,  num_workers=2)

    # ===== optim / amp =====
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.bf16 is False)
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    # ===== logging & csv =====
    csv_f = None
    if cfg.log_csv:
        os.makedirs(cfg.out_dir, exist_ok=True)
        csv_path = os.path.join(cfg.out_dir, 'metrics.csv')
        csv_f = open(csv_path, 'w')
        csv_f.write('step,train_loss,aux,val_loss,val_ppl,tokens_per_sec,gpu_mem_mb\n')

    # ===== helpers =====
    def run_eval(max_batches: int):
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i, (xe, ye) in enumerate(val_dl):
                if i >= max_batches:
                    break
                xe, ye = xe.to(device), ye.to(device)
                with torch.amp.autocast('cuda' if device.type=='cuda' else 'cpu', dtype=amp_dtype):
                    _, ce, _ = model(xe, ye)
                total += float(ce.item()); n += 1
        model.train()
        if n == 0:
            return float('nan'), float('nan')
        val_loss = total / n
        val_ppl = math.exp(val_loss) if math.isfinite(val_loss) and val_loss < 20 else float('inf')
        return val_loss, val_ppl

    def save_best_snapshot(step):
        # ベスト専用スナップショット
        path = os.path.join(cfg.out_dir, 'pytorch_model.best.bin')
        torch.save({'model_state_dict': model.state_dict(), 'step': step}, path)
        with open(os.path.join(cfg.out_dir, 'best_step.txt'), 'w') as f:
            f.write(str(step))
        print(f"[best_saved] step={step} -> {path}")

    # ===== training loop =====
    tokens_per_step = cfg.batch_size * cfg.seq_len
    step = 0
    model.train()
    t0 = time.time()
    t_start = time.time()

    # early stopping state（改善停止ベース）
    best_val = float('inf')
    best_step = -1
    no_improve = 0
    stop_training = False
    last_val_loss = float('nan')
    last_val_ppl  = float('nan')

    # ベスト重みを保持（メモリが許せば）
    best_state = None

    while step < cfg.steps and not stop_training:
        for x, y in train_dl:
            x = x.to(device); y = y.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if device.type=='cuda' else 'cpu', dtype=amp_dtype):
                _, ce, aux = model(x, y)
                loss = ce + 0.01 * aux

            if cfg.bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            if cfg.grad_clip is not None:
                if cfg.bf16:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                else:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if cfg.bf16:
                opt.step()
            else:
                scaler.step(opt); scaler.update()

            step += 1

            # throughput & memory
            dt = time.time() - t0
            toks_sec = (step * tokens_per_step) / max(1e-6, dt)
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

            # periodic eval（改善停止で早停）
            if cfg.eval_every and (step % cfg.eval_every == 0):
                val_loss, val_ppl = run_eval(cfg.val_batches)
                last_val_loss, last_val_ppl = val_loss, val_ppl

                # 改善判定（min_deltaぶん良くなったらOK）
                if val_ppl + cfg.early_stop_min_delta < best_val:
                    best_val = val_ppl
                    best_step = step
                    no_improve = 0
                    # ベスト状態を保持＆保存
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    save_best_snapshot(step)
                    print(f"[best] step={step} val_ppl={val_ppl:.3f}")
                else:
                    no_improve += 1
                    print(f"[plateau] no_improve={no_improve}/{cfg.early_stop_patience} (best={best_val:.3f} @ {best_step})")
                    if step >= cfg.early_stop_warmup and no_improve >= cfg.early_stop_patience:
                        print(f"[early_stop] plateau reached at step {step} (best {best_val:.3f} @ {best_step})")
                        stop_training = True

                # 時間上限
                if (cfg.max_time_min is not None) and ((time.time() - t_start) / 60.0 >= cfg.max_time_min):
                    print(f"[early_stop] reached time limit {cfg.max_time_min} min at step {step}")
                    stop_training = True
            else:
                val_loss, val_ppl = last_val_loss, last_val_ppl

            # logging
            if step % 20 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f} | aux {aux.item():.4f} | "
                      f"val {val_loss:.4f} ppl {val_ppl:.2f} | {toks_sec:.0f} tok/s | mem {gpu_mem_mb:.0f}MB")

            if csv_f is not None:
                csv_f.write(f"{step},{loss.item():.6f},{aux.item():.6f},{val_loss:.6f},{val_ppl:.6f},{toks_sec:.2f},{gpu_mem_mb:.1f}\n")
                csv_f.flush()

            if cfg.save_every and step % cfg.save_every == 0:
                save_checkpoint(cfg.out_dir, model, cfg.vocab_name, cfg, step)

            if stop_training or step >= cfg.steps:
                break

    if csv_f is not None:
        csv_f.close()

    # ベスト重みに戻してから final save
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore_best] step={best_step} val_ppl={best_val:.3f}")

    save_checkpoint(cfg.out_dir, model, cfg.vocab_name, cfg, step)

    print("=== SAMPLE ===")
    print(generate(model, enc, prompt="Once upon a time", max_new_tokens=80))

if __name__ == '__main__':
    main()

