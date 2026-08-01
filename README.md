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
