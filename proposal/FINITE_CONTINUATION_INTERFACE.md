# Shared Finite Continuation Interface

The strict finite-state adapter is dataset-independent. A dataset loader divides
each example into four strings/counts and passes them through
`build_finite_continuation_prompt`:

```text
evidence:       information consumed exactly once
continuation:   later query/control text that may be reread
latent_tokens:  number T of recurrent hard-state updates
tail:           answer-control text such as [A]
```

The formatter emits:

```text
evidence
<|start-latent|>
continuation
<|latent|> * T
<|end-latent|>
tail
```

`<|start-latent|>` is the access boundary, not a learned state slot. The
adapter computes `z_0` from tokens before it, deletes those tokens and their KV,
and performs one update per `<|latent|>` from the continuation plus current
hard code. Only `z_t` transports evidence-dependent information between
updates. The final decoder receives the continuation, final state, and tail.

Every new dataset can use the strict interface by:

1. defining which fields form `evidence` and which form the legal
   post-boundary `continuation`;
2. constructing latent examples with the shared formatter;
3. setting `continuation_interface: strict_continuation` and
   `finite_state.access_mode: strict_read_once`;
4. documenting why the continuation may legally be reread.

## ProsQA mapping

ProsQA assigns:

```text
evidence     = shuffled graph edges
continuation = [Q] target neg_target [R] root
tail         = [A] only at final-answer/no-answer stages
```

Its existing curriculum is preserved:

| Curriculum stage | Hard updates | Supervised output |
|---:|---:|---|
| 0 | 0 | sampled one-hop node |
| 1 | 1 | sampled two-hop node |
| 2 | 2 | sampled three-hop node |
| final depth D | D | `[A]` target |

With probability `uniform_prob`, training rehearses an earlier legal stage.
Only the selected node or final target is supervised. The graph prefix is
sealed before `[Q]` at every stage, including stage 0; therefore `z_0` exists
even when no `<|latent|>` update marker is present.

Run the fully fine-tuned GPT-2 configuration with:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 run.py \
  args/finite_cot/prosqa_gpt2_strict_continuation.yaml
```

The configuration uses no LoRA and applies explicit four-bit quantization only
to the 64-coordinate persistent state. GPT-2 weights and ordinary temporary
activations remain floating-point and fully trainable.

## MuSiQue mapping

MuSiQue assigns cropped evidence passages to `evidence`, the natural question
prefixed by `[Q]` to `continuation`, and `[A]` to `tail`. Its complete data and
access record is in [MUSIQUE_FINITE_COT.md](MUSIQUE_FINITE_COT.md).

## Question-conditioned alternative

The same formatter supports `interface="question_conditioned"`, which moves
the continuation before the boundary. In that mode the first latent marker
initializes `z_0` from evidence plus question, and subsequent markers are
state-only updates. This is an ecological finite-state baseline, not a strict
continuation experiment.

