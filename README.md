# Finite-Precision Continuous-CoT Experiment Harness


## Implemented experiments

### Hybrid frontier retrieval

The encoder sees a uniformly random bit frontier `S`, creates a hard finite
state and/or hard categorical transcript, and then the prefix is sealed. The
decoder receives only the retained summary and an independent query `V`.
By default training evaluates all legal membership queries for each encoded
frontier (the prefix is encoded once); evaluation still uses an independent
uniform query. Set `train_all_queries: false` for single-query supervision.

```bash
python experiments/run_finite_cot.py args/finite_cot/frontier_capacity.yaml
python experiments/run_sweep.py args/finite_cot/frontier_sweep.yaml \
  --output results/finite_cot/frontier_sweep.jsonl
python experiments/run_finite_cot.py args/finite_cot/frontier_learned.yaml
```

`frontier_capacity.yaml` and the sweep use `fixed_hybrid`, an explicit packing
construction that cleanly checks the predicted total-bit threshold. The
`frontier_learned.yaml` run asks whether optimization discovers a useful code;
do not merge these evidence types. For equal retained budgets, compare
latent-only, transcript-only, and hybrid settings. Run `model: unquantized` and
`model: prefix_reread` only as controls; their ledgers explicitly mark them as
outside the finite-state bound.

### Pointer chasing

`LearnedPointerMachine` carries a hard `d`-coordinate, `p`-bit state and may
make exactly one hard query to `f` per update. Training may use the cached path
for teacher-forced query selection and intermediate-state supervision;
evaluation uses the model's hard decoded query.

```bash
python experiments/run_finite_cot.py args/finite_cot/pointer_depth.yaml
```

Sweep capacity with `updates >= depth`, then sweep updates with enough state.
`BinaryPointerMachine` is the imposed-code construction check; it is not learned
evidence.

### Rare witness

The full hypercube aggregator reads every marker and reports `K*d` scalar work.
The sampled comparator can observe only explicitly inspected branches. The
normalized aggregator separately sweeps fixed-point fractional resolution.

```bash
python experiments/run_finite_cot.py args/finite_cot/rare_witness.yaml
python experiments/run_mechanism_checks.py \
  --output results/finite_cot/mechanism_checks.json
```

## Strict ProsQA adapter

The original `coconut.py` retains the prefix KV cache and every earlier latent
position. That behavior is useful as the original baseline but is not a strict
`CCoT(d,p,T)` state machine. Set `finite_state.enabled: true` to select
`StrictFiniteStateCoconut`:

```bash
torchrun --nnodes 1 --nproc_per_node 2 run.py \
  args/prosqa_finite_state_2l_8h_768d.yaml
```

`readonly_input` recomputes each update from the immutable prefix and current
hard state. `sealed_prefix` uses the prefix once to initialize the state and
then applies a state-only transition. Neither mode uses `past_key_values` or
retains the latent history. This reference implementation loops over batch
items for an easily audited boundary, so it is slower than the original model.

The original curriculum can still include a zero-latent stage. Do not report
that stage as a finite-state evaluation; evaluation examples must contain at
least one latent slot for the bottleneck to apply.

Set `pretrained_model_id` to initialize the causal LM from Hugging Face before
adding the finite-state CCoT bottleneck. This bypasses `model_id`, resizes the
language-model vocabulary to `STokenizer`, and writes the resolved Hugging Face
configuration to `<save_path>/<name>/model_config.json`. Set it to `null` to
construct random weights from the local `model_id` configuration instead.

Set `tokenizer: stokenizer` to use the built-in symbolic tokenizer. Any other
value is treated as a Hugging Face tokenizer ID. The trainer then registers
`<|start-latent|>`, `<|end-latent|>`, and `<|latent|>` as additional special
tokens, configures right padding, resizes the model embeddings, and saves the
resolved tokenizer under `<save_path>/<name>/tokenizer/`.

### ProsQA QAT grid search

The strict finite-state quantizer uses hard quantize/dequantize operations at
`model_bits` in the forward pass and an identity straight-through gradient in
the configured floating `training_dtype`. Sweep state dimension and model
precision while holding the training dtype fixed:

```bash
python experiments/run_prosqa_grid.py args/prosqa_finite_state_grid.yaml --dry-run
python experiments/run_prosqa_grid.py args/prosqa_finite_state_grid.yaml
```

Runs execute sequentially and get independent checkpoint directories named
with `d`, model bits (`mb`), and the training dtype. Generated configs and a
JSONL manifest are written under `results/prosqa_grid/`. Existing checkpoints
are resumed by `run.py`.

Grid names are generated from every varied field, not just the original two.
For example:

```yaml
grid:
  state_dim: [32, 64]
  model_bits: [4, 8]
  batch_size_training: [256, 512]
  training_dtype: [bfloat16]
```

This produces names such as
`prosqa-finite-state-readonly-qat-d32-mb4-bs256-dtypebfloat16` and saves its
checkpoints under
`ckpts/prosqa-finite-state-readonly-qat-d32-mb4-bs256-dtypebfloat16/`.
Dotted nested keys such as `finite_state.access_mode` are also supported. Each
manifest entry includes the exact `grid_values`, generated config path,
`checkpoint_dir`, and launch command.

## Verification

```bash
python -m unittest discover -s tests -v
python experiments/run_mechanism_checks.py
```

## Two-B200 high-throughput training

Use the BF16/DDP preset for the original Coconut model:

```bash
cd /workspace/LCoT
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  run.py args/prosqa_coconut_2xb200.yaml
```

The preset starts at `batch_size_training: 1024` per GPU (global batch 2048),
uses BF16, fused AdamW, SDPA, tensor-core-friendly padding, pinned asynchronous
copies, batched validation, and DDP. For this two-layer model, DDP is faster than
FSDP and easily fits in B200 memory. Raise the per-GPU batch until peak allocated
VRAM is around 85--90%; lower it if using a materially larger pretrained model.
Set `distributed_strategy: fsdp` only when the model itself no longer fits
comfortably on one GPU. Gradient checkpointing is available for the base and
strict finite-state paths, but not for original Coconut because its recurrent
training pass depends on a live KV cache.

Blackwell requires a PyTorch wheel built with CUDA 12.8 or newer. Verify before
launching:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda); print([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])'
```

The CUDA backend should report 12.8 or newer and both devices should be visible.

The tests verify finite alphabets and hard codes, prefix sealing, one-query
oracle enforcement, Boolean BFS, pointer capacity/depth controls, hypercube
decoding, and the no-history ProsQA adapter.

## Interpretation rules

- Deterministic BFS, binary pointer, and hypercube checks validate the
  constructions, not learnability.
- The frontier phase diagram is learned evidence for total retained-summary
  capacity only under the sealed-prefix interface.
- The learned pointer run uses intermediate path supervision; report this
  explicitly and keep answer-only training as a later, harder control.
- The hypercube result is an access-model separation. It compresses persistent
  identity but still reads all `K=2^d` markers.
- Always retain the emitted resource ledger and hard code examples with each
  result JSON.
