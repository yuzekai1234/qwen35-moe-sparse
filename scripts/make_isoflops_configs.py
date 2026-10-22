#!/usr/bin/env python3
"""Prepare IsoFLOPs configuration files and validation partitions; never train."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tokenizers import Tokenizer
from q35.presets import MODEL_DEFAULTS, ROOT, budget_tokens, model_config
from q35.flops import count

POOL = Path('/home/shared/qwen35_moe_sparse/cache/isoflops-gzy10k-16b-v1')
VOCAB = 10112
DEST = ROOT / 'configs/isoflops'
BUDGETS = {'1e16': 10**16, '3e16': 3*10**16, '1e17': 10**17, '3e17': 3*10**17}
HIDDENS = range(128, 1601, 64)
TOKENS_PER_ACTIVE = (4, 400)  # D/Na window of the sweep; every on-rule shape inside it is trained
SEQ = 8192
MIN_ACTIVE_FRACTION = 0.10    # competition rule: non-vocab active / total parameters


def shape_for(hidden):
    """The declared scaling rule: hidden alone fixes the model.  Depth max(4, round(hidden / 51.2))
    keeps the original width/depth ratio; within-layer proportions follow original Qwen3.5-35B-A3B
    ratios where they scale (attention q 2h / kv h/4, GDN k h / v 2h), head dims fixed at 64,
    64 experts top-4 with expert and shared widths h/2.  Sweep and submitted models share it, so the
    fitted N_opt lies on the same one-parameter family the submissions are taken from."""
    q, kv = hidden // 32, max(1, hidden // 256)
    while q % kv:
        kv -= 1
    layers = max(4, round(hidden / 51.2))
    return {**MODEL_DEFAULTS, 'layers': layers, 'hidden': hidden, 'heads': q, 'kv_heads': kv,
            'head_dim': 64, 'lin_k_heads': hidden // 64, 'lin_v_heads': hidden // 32, 'lin_head_dim': 64,
            'experts': 64, 'topk': 4, 'moe_ffn': hidden // 2, 'shared_ffn': hidden // 2, 'experts_impl': 'eager'}


def candidates():
    for hidden in HIDDENS:
        shape = shape_for(hidden)
        fc = count(model_config(shape, VOCAB, SEQ).to_dict(), SEQ)
        ratio = fc['active_nonvocab_params'] / fc['total_nonvocab_params']
        if ratio < MIN_ACTIVE_FRACTION:
            raise ValueError(f'h{hidden}: active fraction {ratio:.3f} violates the 10% rule')
        yield shape, fc, ratio


def select_shapes(compute, batch):
    """Every on-rule shape whose D/Na falls inside TOKENS_PER_ACTIVE (4-6 per budget, ~2x apart in Na)."""
    chosen = []
    for shape, fc, ratio in candidates():
        tokens = budget_tokens(fc['train_flops_per_token'], compute, batch)
        if tokens and TOKENS_PER_ACTIVE[0] <= tokens / fc['active_nonvocab_params'] <= TOKENS_PER_ACTIVE[1]:
            chosen.append((shape, fc, ratio, tokens))
    return chosen


def lr_for(compute):
    """DeepSeek LLM (2024) compute-optimal peak learning rate; shared by every shape of a budget."""
    return round(0.3118 * compute ** -0.125, 6)


def batch_for(compute):
    """DeepSeek LLM compute-optimal batch, rounded to a power-of-two number of sequences (at least 8)."""
    sequences = 0.2920 * compute ** 0.3271 / SEQ
    return SEQ * max(8, 2 ** round(np.log2(sequences)))


def write_json(path, value, force=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != text and not force:
        raise ValueError(f'Refusing to overwrite edited config: {path}; use --force explicitly')
    path.write_text(text)


def validation_partitions(force=False):
    meta = json.loads((POOL / 'meta.json').read_text())
    target = DEST / 'validation.json'
    if target.exists():
        result = json.loads(target.read_text())
        if result['tokenizer_sha256'] != meta['tokenizer_sha256'] or result['validation_sha256'] != meta['valid_sha256']:
            raise ValueError('Frozen validation partition identity changed')
        return result
    reference = json.loads(Path(meta['validation']['manifest']).read_text())
    source = Path(meta['validation']['manifest']).parent / 'valid.jsonl'
    tok = Tokenizer.from_file(str(POOL / 'tokenizer/tokenizer.json'))
    lengths, batch = [], []
    encoded_hash, raw_hash = hashlib.sha256(), hashlib.sha256()
    def flush():
        if not batch:
            return
        for encoding in tok.encode_batch(batch, add_special_tokens=False):
            ids = np.asarray(encoding.ids + [53], dtype='<u4')
            lengths.append(len(ids)); encoded_hash.update(ids.tobytes())
        batch.clear()
    with source.open('rb') as f:
        for raw in f:
            raw_hash.update(raw)
            batch.append(json.loads(raw)['text'])
            if len(batch) >= 128:
                flush()
        flush()
    if raw_hash.hexdigest() != reference['jsonl_sha256'] or encoded_hash.hexdigest() != meta['valid_sha256']:
        raise ValueError('Validation re-tokenization does not reproduce the frozen binary')
    ends = np.cumsum(lengths, dtype=np.int64)
    if int(ends[-1]) != meta['valid_tokens']:
        raise ValueError('Validation document lengths do not match its binary')
    dev_last = int(np.searchsorted(ends, 2_000_000))
    dev_end = int(ends[dev_last])
    curve_last = int(np.searchsorted(ends, dev_end + 16_000_000))
    curve_end = int(ends[curve_last])
    result = dict(version=1, tokenizer_sha256=meta['tokenizer_sha256'], validation_sha256=meta['valid_sha256'],
                  source_jsonl=str(source), document_count=len(lengths), total_tokens=int(ends[-1]),
                  boundary_rule='complete documents, end-exclusive token ranges; no shared documents',
                  dev=dict(offset=0, end=dev_end, targets=dev_end-1, documents=dev_last+1),
                  curve=dict(offset=dev_end, end=curve_end, targets=curve_end-dev_end-1,
                             documents=curve_last-dev_last),
                  report=dict(offset=curve_end, end=int(ends[-1]), targets=int(ends[-1])-curve_end-1,
                              documents=len(lengths)-curve_last-1))
    write_json(target, result, force)
    print(f'Frozen validation partitions: {result}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Replace existing generated JSON configurations')
    args = parser.parse_args()
    if not (POOL / 'READY').is_file():
        raise ValueError('16B training pool is not READY')
    DEST.mkdir(parents=True, exist_ok=True)
    val = validation_partitions(args.force)
    rows = []
    for budget_name, compute in BUDGETS.items():
        lr, batch = lr_for(compute), batch_for(compute)
        for shape, fc, ratio, tokens in select_shapes(compute, batch):
            hidden, layers = shape['hidden'], shape['layers']
            name = f'h{hidden}_L{layers}'
            directory = DEST / budget_name / name
            steps = tokens // batch
            run_dir = f'runs/{budget_name}/{name}'
            train = dict(run_dir=run_dir, data=str(POOL/'train.bin'), val_data=str(POOL/'valid.bin'),
                         tokenizer=str(POOL/'tokenizer'), seq=SEQ, tokens=0, batch_tokens=batch,
                         micro_bs=8, auto_micro_bs=1, lr=lr, min_lr_ratio=0.1, warmup=0, warmup_ratio=0.05,
                         warmup_min=10, wd=0.1, clip=1.0, seed=0, data_seed=0,
                         data_sampling='shuffled_blocks', require_ready_pool=1,
                         eval_sampling='sequential', eval_offset=val['dev']['offset'],
                         eval_tokens=val['dev']['targets'], final_eval_offset=val['curve']['offset'],
                         final_eval_tokens=val['curve']['targets'], log_every=10, eval_every=100,
                         ckpt_every=100, resume=1, max_steps=0, tensorboard=1)
            accounting = dict(total_params=fc['total_nonvocab_params']+fc['embed_params'],
                              total_nonvocab_params=fc['total_nonvocab_params'],
                              active_nonvocab_params=fc['active_nonvocab_params'],
                              active_fraction=ratio,
                              active_params_with_vocab=fc['active_nonvocab_params']+fc['embed_params'],
                              train_flops_per_token=fc['train_flops_per_token'], train_tokens=tokens,
                              tokens_per_active_param=tokens/fc['active_nonvocab_params'],
                              steps=steps, warmup_steps=max(10,round(steps*0.05)),
                              actual_train_flops=tokens*fc['train_flops_per_token'],
                              compute_shortfall_fraction=1-tokens*fc['train_flops_per_token']/compute)
            config = dict(schema_version=1, compute=compute, model=shape, train=train, accounting=accounting,
                          experiment=dict(budget=budget_name, model=name, phase='main',
                                          scaling_rule='scripts/make_isoflops_configs.py:shape_for (hidden only; layers = max(4, round(hidden/51.2)))',
                                          lr_rule='0.3118*C^-0.125 (DeepSeek LLM), shared within a budget',
                                          batch_rule='0.2920*C^0.3271 tokens, rounded to 2^k sequences',
                                          validation_manifest='configs/isoflops/validation.json'))
            write_json(directory/'config.json',config,args.force)
            (ROOT/run_dir).mkdir(parents=True,exist_ok=True)
            command = f'GPUS=0 bash scripts/train_isoflops.sh {budget_name} {name}'
            text = (f'# {budget_name} / {name}\n\n'
                    f'模型：{layers} 层 × hidden {hidden}（宽深比 {hidden/layers:.0f}）；全注意力 q/kv 头 {shape["heads"]}/{shape["kv_heads"]}，'
                    f'GDN k/v 头 {shape["lin_k_heads"]}/{shape["lin_v_heads"]}（头维均 64）；64 专家 top-4，专家/共享专家宽 {hidden//2}；'
                    f'GDN/full attention=3:1；词表 gzy 10K（{VOCAB}）。层内比例由 shape_for 统一生成。\n\n'
                    f'预算 {compute:,} FLOPs；训练 {tokens:,} token，{steps:,} 步；'
                    f'每 token {fc["train_flops_per_token"]:,} FLOPs。\n\n'
                    f'非词表总参数 {fc["total_nonvocab_params"]:,}，激活非词表参数 {fc["active_nonvocab_params"]:,}'
                    f'（Na/N={ratio:.1%}）；D/Na={tokens/fc["active_nonvocab_params"]:.0f}。\n\n'
                    f'全局 batch={batch} token（{batch//SEQ} 条序列），seq={SEQ}；micro_bs 从 8 起自动探测（auto_micro_bs=1），'
                    f'梯度累积 = {batch//SEQ}/(卡数×micro_bs)；warmup={accounting["warmup_steps"]} 步；cosine 降到峰值的 10%。\n\n'
                    f'峰值学习率 {lr}（0.3118·C^-0.125，同一预算内统一）；model seed=0，data seed=0。'
                    '所有预算/模型共用同一无放回打乱块序列。验证分区按完整文档冻结，见 ../../validation.json。\n\n'
                    f'输出：`{run_dir}/`（日志、checkpoint、TensorBoard、final_eval.json、HF 导出）。\n\n'
                    f'仅检查（无 GPU 分配，不训练）：\n\n```bash\n{command} --dry-run\n```\n\n'
                    f'以后启动：\n\n```bash\n{command}\n```\n\n'
                    'CLI 参数覆盖 JSON；修改预算、学习率或 seed 后须使用新 run_dir。\n')
            (directory/'README.md').write_text(text)
            rows.append(dict(budget=budget_name,model=name,lr=lr,batch_tokens=batch,
                             config=str((directory/'config.json').relative_to(ROOT)),run_dir=run_dir,**accounting))
    write_json(DEST/'manifest.json',dict(version=1,pool=str(POOL),runs=rows),args.force)
    for r in rows:
        print(f"{r['budget']} {r['model']:>6} lr={r['lr']} batch={r['batch_tokens']} Na/N={r['active_fraction']:.1%} "
              f"D/Na={r['tokens_per_active_param']:.0f} steps={r['steps']}", flush=True)
    print(f'Prepared {len(rows)} configs. No training was started.', flush=True)


if __name__ == '__main__':
    main()
