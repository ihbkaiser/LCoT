# MuSiQue Finite-State CoT: Data and Access Note

This note records exactly how the current MuSiQue experiments use the official
data, where the finite boundary lies, and which claims each access interface can
support. It is part of the experimental contract; the two runs below must not
be pooled under one access label.

The marker layout is produced by the shared
`build_finite_continuation_prompt` formatter also used by strict ProsQA. The
finite adapter itself is dataset-agnostic: loaders decide which text is
evidence and which text is the legal continuation.

## Data used

- Training reads `data/musique_ans_v1.0_train.jsonl`, the official answerable
  training split.
- Validation and answer-accuracy reporting read
  `data/musique_ans_v1.0_dev.jsonl`, the public labeled development split.
- The public test JSONL is not used for answer accuracy because it has no
  released answers or decompositions.
- An example is retained only when it is answerable, has a nonempty final
  answer and decomposition, and every decomposition answer occurs in its
  labeled supporting paragraph. Failed examples are filtered; this runtime
  loader does not yet freeze or publish a rejection manifest.

For each retained example, all labeled supporting paragraphs come first. The
loader then takes the configured number of non-supporting paragraphs supplied
with that MuSiQue instance (`distractors: 8` by default). These are
dataset-supplied distractors; the current loader does not independently verify
that they are topic-matched.

Each paragraph is capped at 112 GPT-2 BPE tokens. For a supporting paragraph,
the crop starts with the sentence containing its labeled decomposition answer
and admits nearby sentences while they fit. Distractors are cropped from their
first sentence and nearby context. The evidence prefix is capped at 896 tokens
because GPT-2 has only 1,024 positional embeddings. If the complete example is
too long, the loader reserves tokens for the full question, boundary markers,
latent markers, `[A]`, and supervised answer before shortening the evidence.
Gold passages precede distractors, so right-side shortening removes distractor
material first.

The fast GPT-2 byte-level BPE tokenizer is mandatory. Preprocessing uses eight
host threads, prints progress every 1,000 source examples, and is cached in
memory for all epochs of one run.

## Shared model and training policy

Both experiments load pretrained GPT-2 and fully fine-tune every transformer,
embedding, language-model head, and finite-state parameter. `use_lora: false`
is explicit. No LoRA adapters are created.

Only the persistent state is passed through the explicit hard quantizer. With
`state_dim: 64` and `bits_per_coordinate: 4`, each state has

```text
B = d p = 64 * 4 = 256 bits.
```

GPT-2 weights and ordinary temporary activations use the configured floating
training dtype (BF16 by default); there is no weight QAT, low-bit weight loader,
ordinary-activation quantizer, or quantized KV cache. BF16 arithmetic is a
separate implementation cost and is not the declared four-bit state alphabet.
At evaluation, each of the 64 state coordinates is one of exactly 16 hard code
levels. Straight-through gradients are used only to train through this state
rounding.

Prompt tokens are masked from the loss. Training supervision is only the final
MuSiQue answer followed by EOS. Published decomposition answers are used for
grounding and evidence cropping, not as intermediate latent-state targets.

## Question-conditioned finite baseline

Launch:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 run.py \
  args/finite_cot/musique_gpt2_question_conditioned.yaml
```

Visible layout:

```text
[EVIDENCE] passages [Q] question
<|start-latent|>
<|latent| z_0> <|latent| z_1> ... <|latent| z_T>
<|end-latent|> [A] answer
```

The first latent marker initializes the state from evidence and question; the
remaining `latent_steps = T` markers perform state-only recurrent updates:

```text
z_0 = Q_p(E(evidence, question))
z_{t+1} = Q_p(U(z_t))
answer = V(z_T)
```

After initialization, prefix tokens and prefix KV are absent, and no latent
history is retained. This run tests whether a 256-bit query-conditioned state
can solve natural MuSiQue QA. Because the question is known while evidence is
compressed, the encoder may retain only question-relevant evidence. This run
is therefore an ecological finite-state baseline, not evidence for the strict
continuation bound.

## Strict read-once continuation run

Launch:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 run.py \
  args/finite_cot/musique_gpt2_strict_continuation.yaml
```

Visible layout:

```text
[EVIDENCE] passages
<|start-latent|>                 # read-once boundary
[Q] question
<|latent| z_1> ... <|latent| z_T>
<|end-latent|> [A] answer
```

The adapter computes and hard-quantizes `z_0` from evidence alone, then drops
the evidence tokens and their KV. Each of exactly `T` updates recomputes from
the shared question plus only the current hard state:

```text
z_0 = Q_p(E(evidence))
delete evidence and evidence KV
z_{t+1} = Q_p(U(question, z_t))
delete the previous state expansion and temporary update activations
answer = V(question, z_T)
```

The question may be reread because it is the legal post-boundary continuation;
it contains no information copied from the consumed evidence. Only `z_t`
carries evidence-dependent information between updates. The trace contains
`T + 1` hard codes (`z_0` through `z_T`). With `T=0`, the decoder receives the
question and evidence-only `z_0` without a recurrent update. This is the run
aligned with the continuation interface.

## What is not implemented yet

This loader is not the complete frozen EvidenceGraph-QA transformation in the
Experiment 3 proposal. It currently lacks opaque per-instance entity aliases,
a grouped continuation bank, counterfactual twins, a released preprocessing
manifest, chain-versus-branching labels, inverse-hop-frequency sampling, the
manual 200-example audit, sealed-paragraph selection, binary/hybrid retained
transcripts, and 2WikiMultiHopQA. Results must state these omissions rather
than calling the current data a completed evidence-locked benchmark.
