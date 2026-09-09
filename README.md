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
Pretrained models are adapted with rank-16 LoRA. By default the original
transformer is frozen while LoRA weights, the resized token input/output
interface, and the finite-state bottleneck remain trainable. Large-vocabulary
models can set `train_task_interfaces: false` to keep the full embedding and LM
head frozen; this is the recommended strict finite-state Llama configuration.
Resumed pretrained runs must use a checkpoint created with the same LoRA
topology; older full-finetuning checkpoints are rejected with a compatibility
error.

Both vocabulary and latent inputs preserve the transformer's hidden width.
Vocabulary IDs map directly to vectors of width `H`; a latent hidden vector is
projected from `H` to the finite state width `d`, hard-quantized, and expanded
from `d` back to `H`. The expanded vector occupies one sequence position next
to vocabulary embeddings. Its width is continuous, but its retained
information is still limited to the discrete `d * model_bits` code because the
expansion is a deterministic function of that code.

Set `tokenizer: stokenizer` to use the built-in symbolic tokenizer. Any other
value is treated as a Hugging Face tokenizer ID. The trainer then registers
`<|start-latent|>`, `<|end-latent|>`, and `<|latent|>` as additional special
tokens, configures right padding, resizes the model embeddings, and saves the
resolved tokenizer under `<save_path>/<name>/tokenizer/`.

### Qwen3-0.6B LoRA run

Install the pinned dependencies from `requirements.txt`, then save the
following as `args/prosqa_finite_state_qwen3_0.6b.yaml`. Qwen3 requires
Transformers 4.51 or newer; the requirements file pins a compatible release.
The native [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
tokenizer is retained and the three latent markers are added automatically.

```yaml
project: finite-cot
save_path: ckpts
name: prosqa-qwen3-0.6b-finite-state-d64-p2-readonly

only_eval: false
coconut: true
cot: false
no_thoughts: false
no_cot: false

c_thought: 1
epochs_per_stage: 25
max_latent_stage: 4
pad_latent_to_max: true

finite_state:
  enabled: true
  state_dim: 64
  model_bits: 2
  bits_per_coordinate: 2
  clip_value: 1.0
  learnable_clip: false
  access_mode: readonly_input

save_only_improve: true
save_best_only: true
uniform_prob: 0.1
model_id: configs/symbol-2layer-8head-768dim.json # bypassed for pretrained runs
pretrained_model_id: Qwen/Qwen3-0.6B
tokenizer: Qwen/Qwen3-0.6B
load_model_path: None

seed: 17
resume: 0
bf16: true
training_dtype: bfloat16
train_path: data/prosqa_train_graph_4_coconut.json
val_path: data/prosqa_valid_graph_4_coconut.json
reset_optimizer: false
batch_size_training: 2
debug: false
gradient_accumulation_steps: 64
num_epochs: 300
lr: 1.0e-4
weight_decay: 0.01
```

Launch on one GPU from the repository root:

```bash
python -m pip install -r requirements.txt
torchrun --standalone --nnodes=1 --nproc_per_node=1 run.py \
  args/prosqa_finite_state_qwen3_0.6b.yaml
```

With `save_best_only: true`, each global validation-accuracy improvement
overwrites `best_model.pt`. The best epoch within each curriculum stage is also
retained as `best_stage_<stage>.pt` and overwritten only when that stage's score
improves. For LoRA runs, these files contain only trainable weights: the LoRA
adapters, token embeddings/LM head, and finite-state modules; the frozen Qwen
base is not duplicated. The same best-only behavior can be enabled for any
config with the `--save-best-only` command-line flag.

The effective global batch size is `1 GPU * 2 examples * 64 accumulation =
128`. Reduce `batch_size_training` if memory is tight and increase
`gradient_accumulation_steps` proportionally to retain that effective batch.

### Llama 3.2-3B LoRA run

`args/prosqa_finite_state_llama3.2_3b.yaml` provides a conservative single-GPU
starting point for `meta-llama/Llama-3.2-3B`. It loads the checkpoint directly
in BF16, uses SDPA, enables non-reentrant gradient checkpointing, and freezes
the large token embedding/LM-head matrices while training LoRA plus the finite
state bottleneck. To keep the first 3B run small and preserve the existing Qwen
LoRA topology, Llama initially adapts only the attention projections
(`q/k/v/o_proj`).

Before launching a full ProsQA curriculum, run the one-step smoke test:

```bash
python scripts/smoke_llama32_3b.py
```

The smoke test loads the real 3B checkpoint, adds the latent tokens, performs a
finite-state forward/backward/optimizer step, checks that LoRA and bottleneck
gradients are non-zero, and prints peak CUDA allocated/reserved memory. The
model repository may require accepting Meta's license and authenticating with
Hugging Face first.

Then launch training with:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 run.py \
  args/prosqa_finite_state_llama3.2_3b.yaml
```

The Llama config starts at `batch_size_training: 1` and
`gradient_accumulation_steps: 32`. Increase the micro-batch only after the
smoke test and an initial training step establish available VRAM headroom.

### Full fine-tuning on one B200

The LoRA path above is a memory-saving option, not a Coconut requirement. To
follow the original Coconut-style full fine-tuning setup on a 1-GPU B200, use:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 run.py \
  args/prosqa_finite_state_llama3.2_3b_full_b200.yaml
```

This config keeps the Llama 3.2-3B **Base** checkpoint, sets `use_lora: false`,
trains the full pretrained model plus the finite-state modules, and uses
BF16 with `batch_size_training: 16` and `gradient_accumulation_steps: 8`.
If memory remains comfortably below the limit, try batch 32 and accumulation 4.
The `peft` package is only needed for the LoRA configuration; full fine-tuning
does not call PEFT.

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

## Verification

```bash
python -m unittest discover -s tests -v
python experiments/run_mechanism_checks.py
```

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
