# COTCE Extended Experimental Program

## A theorem-facing, causal-mechanism evaluation for ICLR/NeurIPS

This document specifies the experimental extension for **Finite-Precision Continuous Chain-of-Thought: Serialization, Frontier Capacity, and Sampling Lower Bounds**. It is intentionally narrower than a benchmark suite. Each main experiment is designed to falsify one resource claim, locate the corresponding empirical phase boundary, and test whether a learned LLM uses the predicted internal mechanism.

The recommended main paper has **four experiments**:

1. a synthetic read-once continuation task testing the joint retained-information bound;
2. a synthetic local-oracle task testing the separate state and update-depth thresholds;
3. an evidence-locked natural-language study on **MuSiQue** and **2WikiMultiHopQA** testing whether the controlled state mechanism transfers to real multihop text;
4. a rare-witness study on the real-news **MultiHop-RAG** corpus testing sampled coverage, continuous aggregation, numerical resolution, and total work.

Thus the program contains two exact theorem stress tests and two real-data mechanism studies, using three published real datasets. The existing GraphQA, CLUTRR, bAbI, fixed BFS, and structured PointerChase results remain useful as appendix pilots and construction checks; they should not carry the new headline empirical claim.

No finite experiment guarantees acceptance at ICLR or NeurIPS. This program clears the appropriate *design bar*: theorem-specific predictions, enforced interfaces, causal interventions rather than probes alone, current latent-reasoning baselines, real-data transfer, preregistered statistics, and explicit compute accounting.

---

## 1. Claims and experiments

| Paper claim | Necessary empirical interface | Main experiment | Preregistered primary outcome | What would falsify the empirical mechanism claim? |
|---|---|---|---|---|
| Approximate continuation retrieval obeys $dp+L\ge n[1-h_2(\varepsilon)]$ for a fixed binary retained channel | Prefix is consumed once; only the hard state and exactly $L$ retained bits survive; query is revealed after the boundary | E1: FrontierRate | Bit error versus $B/n$, $B=dp+L$, overlaid with the information-theoretic lower envelope | Reproducible below-envelope performance after every side channel has been excluded |
| Local-oracle PointerChase needs $T\ge D$ and $dp\ge\lceil\log_2n\rceil$ | At most one hard oracle-row access per update; no full table, old response, or KV cache persists | E2: LocalOracle-PC | Accuracy surface over $T-D$ and $dp-\lceil\log_2n\rceil$, plus counterfactual state-swap accuracy | Success on paired completions at $T<D$, or endpoint accuracy without a causally used current-vertex state |
| Learned recurrent state can implement the controlled mechanism on natural multihop text | Evidence is before a read-once boundary; question is after it; entity aliases block parametric recall | E3: EvidenceGraph-QA | Hop-conditioned accuracy and donor-consistent answer changes under latent-state interchange | Accuracy gains without evidence sensitivity or without predicted state-patch effects |
| Independent sampled inspection obeys an inverse-mass law; full aggregation changes access but incurs full work | One decisive support branch is uniformly hidden among $K$ exchangeable branches; certified success identifies that branch | E4: RareWitness-RAG | Coverage versus $1-(1-1/K)^r$, certified accuracy, and accuracy-versus-branch-work Pareto front | Above-law sampled coverage under the uniform interface, or an aggregation advantage that vanishes without being disclosed after work matching |

The first two experiments instantiate the paper's formal assumptions. Experiment 3 is partly theorem-aligned on its constructed continuation banks and otherwise an ecological mechanism test. Experiment 4 applies the exact sampling theorem to a controlled real-corpus interface and treats the full multi-document setting as a separately labeled extension.

---

## 2. The instrumented computation model

### 2.1 Quantized Read-Once Latent Loop

Use one common instrumented architecture, called the **Quantized Read-Once Latent Loop (QROLL)** in experiment logs. This is an evaluation harness, not a claimed new reasoning method.

For every example, the only instance-dependent object allowed to persist after the read-once boundary is

\[
(z_t,\tau_t)\in Q_p^d\times\{\langle b0\rangle,\langle b1\rangle\}^{L},
\qquad B=dp+L.
\]

The controlled transcript has exactly $L$ binary symbols, including leading zeros, so it contains exactly $L$ bits. Natural-language CoT is evaluated separately and is never assigned an exact $L$-bit cost.

Let $H$ be the backbone hidden width. A single recurrent step is

\[
z_0=Q_p(W_{\downarrow}h_{\mathrm{prefix}}),\qquad
e_t=W_{\uparrow}z_t,
\]

\[
z_{t+1}=Q_p\!\left(W_{\downarrow}
h_\theta(e_t,\tau_t,c,o_t)\right),
\]

where $c$ is the legal continuation, such as the post-boundary question, and $o_t$ is the legal observation, such as one oracle response. The same transition parameters are tied across every step. Use one latent state token in all primary experiments. A four-token state is allowed only as an ablation and is charged as $4dp$.

The exact forward quantizer is

\[
Q_p(u)=
-1+2\frac{\operatorname{round}\!\left((2^p-1)
(\operatorname{clip}(u,-1,1)+1)/2\right)}{2^p-1}.
\]

The clipping range is global and frozen before test evaluation. There is no per-example scale, offset, norm, temperature, or unquantized residual. Straight-through gradients may be used during training; evaluation uses the exact hard codebook. Log saturation rates and the empirical frequency of every code level.

### 2.2 Mandatory deletion semantics

After computing $z_{t+1}$, destroy all of the following:

- the prefix tokens and their KV cache;
- the latent input embedding $e_t$, the unquantized hidden state, and all temporary activations;
- the KV cache of every previous latent token;
- the previous oracle response or inspected branch;
- soft transcript logits, attention maps, retrieval scores, and any input-dependent quantizer metadata.

Only the integer code for $z_{t+1}$, the $L$ hard binary token IDs, the shared continuation, and the next legally supplied observation may be used. Standard Coconut-style accumulation of all prior latent-token KVs is therefore **not** the primary QROLL condition: it introduces an uncharged $O(TH)$ persistent channel. Include that implementation only as an explicitly unrestricted control.

### 2.3 Executable interface contract

Every evaluation batch should assert the following invariants:

```text
encode prefix -> quantize z0 -> delete prefix and prefix KV
for t = 0, ..., T-1:
    expose only continuation, hard z_t, hard transcript, legal observation
    run one tied transition
    hard-quantize z_{t+1}
    delete every temporary tensor and prior latent KV
decode only from final hard state, hard transcript, and continuation
```

The test harness must fail closed if a tensor reachable from the prefix remains live. Store a per-example hash of the exact integer state and transcript after each step. Run a red-team leakage test in which prefix tensors are overwritten by random bytes after the boundary; predictions must be bit-identical.

### 2.4 Supernet discovery versus fixed-budget confirmation

A full Cartesian grid over $d,p,T,L$, backbone, method, dataset, and seed is neither necessary nor credible. Use a budget-conditioned supernet only for discovery:

- sample $T$, an active coordinate prefix $d$, bit width $p$, and transcript length $L$ per batch;
- mask inactive coordinates before quantization;
- use hard quantization and hard transcript truncation in the forward pass;
- infer the candidate phase boundary on development data.

The headline result must then retrain independently at five preregistered cells: two below the predicted boundary, the boundary cell, and two above it. This separates a true resource transition from interference caused by multi-budget training.

---

## 3. Models, training, and comparison methods

### 3.1 Backbones

Use pretrained-only checkpoints for the primary mechanism study. This avoids treating instruction tuning or a model's native verbal reasoning policy as part of the controlled state mechanism.

| Role | Checkpoint | Use |
|---|---|---|
| Discovery and debugging | [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | Full supernet sweeps, leakage audit, and failed-design diagnosis |
| Primary paper model | [`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base) | Five-seed fixed-budget confirmation for all four experiments |
| Within-family replication | [`Qwen/Qwen3-8B-Base`](https://huggingface.co/Qwen/Qwen3-8B-Base) | Three-seed boundary replication |
| Dense large-model confirmation | [`Qwen/Qwen3-14B-Base`](https://huggingface.co/Qwen/Qwen3-14B-Base) | Three selected configurations per experiment after the scale gate |
| Cross-family cap, below 30B | [`google/gemma-3-27b-pt`](https://huggingface.co/google/gemma-3-27b-pt) | Selected real-data configurations only; text tower, three adapter seeds |

Do not fit a neural scaling exponent across Qwen and Gemma. The 27B result is a cross-family robustness check, not another point on a controlled dense scaling curve. An instruction-tuned checkpoint may be included in an appendix zero-shot study but should not replace the base-checkpoint primary analysis.

All implementation should use Hugging Face Transformers, PEFT, Accelerate, and PyTorch SDPA or FlashAttention-2. Pin `transformers>=4.51.0` for Qwen3 support and record the exact package commit in the artifact manifest. vLLM is not needed.

### 3.2 Optimization

- LoRA targets: `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
- LoRA rank $16$, alpha $32$, dropout $0.05$.
- Train $W_{\downarrow}$, $W_{\uparrow}$, the hard transcript writer, task-token rows, task readout, and LoRA parameters. Freeze all other backbone parameters.
- Learning rate: (10^{-4}) for LoRA; (3\times10^{-4}) for new modules.
- Optimizer: AdamW or paged AdamW 8-bit, $\beta=(0.9,0.95)$, weight decay $0.01$.
- Schedule: 5% linear warm-up, cosine decay, gradient norm clipped to $1.0$.
- Compute dtype: BF16. For 8B, 14B, and 27B, load frozen backbone weights in NF4 with double quantization. Run a BF16-versus-NF4 key-cell check at 4B so weight quantization is not confused with the experimental state precision $p$.
- Use gradient checkpointing and dynamic-length batches. Accumulate to 65,536 non-padding tokens per update for 0.6B/4B/8B and 32,768 for 14B/27B.
- Train synthetic tasks for two passes through the generated manifest and public data for three epochs. Choose the checkpoint by a preregistered development metric; do not tune on final test cells.
- Primary seeds: $\{17,42,137,314,2718\}$ for 0.6B and 4B. Replication seeds: $\{17,42,137\}$ for 8B, 14B, and 27B.

Headline synthetic models use answer-only supervision. Canonical intermediate states are reserved for an oracle-supervision upper bound. On real data, report both a controlled answer-only tier and a method-faithful tier, because Coconut and CoT2 were designed with intermediate-reasoning supervision. Do not compare methods trained with different privileged labels without marking the supervision tier.

### 3.3 Competitive baselines and controls

Use four competitive baseline families, then a small set of diagnostic controls.

| Family | Exact configuration | Fairness rule |
|---|---|---|
| Explicit CoT and self-consistency | Greedy budgets $32,64,128,256$ tokens; self-consistency $r\in\{4,8,16\}$, temperature $0.7$, nucleus threshold $0.95$ | Report generated tokens and total backbone forwards; never count a natural token as one exact bit |
| Coconut | Same backbone and data; author-style curriculum; $T\in\{1,2,4,8,16\}$ | Primary reproduction may retain its native latent KVs, but label its memory as unrestricted; also port the transition to the QROLL deletion interface |
| Soft Thinking | Mixture support $k\in\{1,5,10,15,30\}$; $T\in\{1,2,4,8\}$ | Disable adaptive stopping in the fixed-$T$ mechanism sweep; enable it only for the efficiency appendix |
| CoT2 | Parallel tracks $K_{\parallel}\in\{1,2,4,8,16\}$; continuous steps $T\in\{1,2,4,8\}$ | Use the same canonical step labels in the supervised tier; place policy optimization in an optional appendix |

Mandatory noncompetitive controls are direct answer ($T=L=0$), trained pause/dot tokens matched by backbone forwards, an unrestricted full-context transformer, and an unrestricted latent-KV loop. A Reasoning-by-Superposition-style full aggregation is a construction ceiling in E4, not a fifth general baseline family.

Report separately for every method:

- answer accuracy and evidence certification;
- generated tokens;
- recurrent backbone calls;
- oracle rows or branches read;
- non-padding tokens processed;
- measured wall time, peak VRAM, and trainable parameters;
- profiler-estimated and, where supported, hardware-measured FLOPs.

There is no single scalar notion of “matched compute.” Present answer-versus-forward, answer-versus-branch-read, answer-versus-FLOP, and answer-versus-latency views.

---

## 4. Experiment 1 — FrontierRate: the retained-information phase transition

### 4.1 Task

For each prefix, sample

\[
S=(S_1,\ldots,S_n)\sim\operatorname{Unif}\{0,1\}^n.
\]

Render all $n$ bits as shuffled natural-language clauses using eight balanced templates, for example “vertex 017 is active” and “vertex 017 is inactive.” Consume the prefix, quantize $z_0$, write the fixed binary transcript, and delete the prefix. Only then reveal $V\sim\operatorname{Unif}[n]$ and ask for $S_V$. All queries derived from one $S$ belong to the same split.

This exactly instantiates Corollary 2.2 under the fixed binary channel. With $B=dp+L$, the error prediction is

\[
\varepsilon\ge
h_2^{-1}\!\left(\max\{0,1-B/n\}\right),
\]

where the inverse is taken on $[0,1/2]$. The theorem supplies a lower envelope, not an achievability guarantee.

### 4.2 Data manifest

| Split | $n$ | Prefixes per $n$ | Queries per prefix |
|---|---:|---:|---:|
| Train | 32, 64, 128 | 20,000 | 4 |
| Development | 32, 64, 128 | 2,000 | 8 |
| IID test | 32, 64, 128 | 2,000 | 16 |
| Size-OOD test | 192, 256 | 1,000 | 16 |
| All-continuations audit | 32, 64, 128, 256 | 500 | all $n$ queries |

Use a counter-based generator and publish the generator seed, prefix ID, bit-vector hash, template IDs, permutation, and query indices. No rendered prefix may repeat across splits.

### 4.3 Resource sweep

- $p\in\{1,2,4,8\}$.
- Candidate $d\in\{4,8,16,32,64,128,256\}$.
- $B/n\in\{0.25,0.50,0.75,1.00,1.25,1.50\}$.
- For each $B$, evaluate state/transcript allocations $(dp,L)/B\in\{(1,0),(0.75,0.25),(0.5,0.5),(0.25,0.75),(0,1)\}$, rounded while reporting the actual bit count.
- At matched $dp=32$, compare $(d,p)\in\{(32,1),(16,2),(8,4),(4,8)\}$; repeat at $dp=64$ and $128$.
- Use $T=1$ for the primary theorem test and $T\in\{0,1,2,4\}$ as a nuisance sweep. Extra updates cannot create information that was not retained.

The 0.6B supernet discovers the transition. The 4B fixed models are retrained at $B/n\in\{0.75,0.90,1.00,1.10,1.25\}$, with the closest realizable integer budgets, for five seeds. The 8B and 14B models repeat $0.75,1.00,1.25$ for three seeds.

### 4.4 Mechanistic interventions

1. **Exact collision audit.** Hash $(z_0,\tau_0)$. For every state shared by different frontiers, measure whether the corresponding labels conflict over the all-continuations bank.
2. **Channel ablation.** Zero, independently resample, or permute $z_0$ and $\tau$ separately.
3. **Interchange intervention.** Pair two prefixes queried at the same $V$ with opposite $S_V$. Patch the donor latent state while holding the recipient transcript fixed, then patch the transcript while holding the state fixed. Record donor-consistent output flips.
4. **Single-bit counterfactual.** Change only $S_V$ before the boundary while holding clause order, templates, continuation, and every other bit fixed.
5. **Inference-time precision shock.** Requantize a trained sufficient-state model from $p\in\{8,4,2,1\}$ without retraining and compare with matched-budget retraining.

Linear probes for $S_i$ may be reported descriptively. A representation claim requires the intervention effects above.

### 4.5 Outcomes and decision rule

Primary: mean queried-bit error versus $B/n$, with a hierarchical 95% confidence interval and the theoretical envelope.

Secondary: exact recovery over all continuations, all-frontier reconstruction from a separate frozen decoder, negative log-likelihood, collision-conflict rate, saturation rate, and donor-consistent flip rate.

The strongest positive result is not merely “accuracy rises with dimension.” It is:

- the transition is governed by total retained bits $dp+L$, after controlling shape;
- performance never reproducibly crosses below the lower envelope;
- above the boundary, exact-state collisions disappear and causal patches move outputs through the channel that received the bits.

Above-boundary failure is an optimization or decoding failure, not a contradiction of sufficiency. Below-envelope success triggers a leakage audit before any scientific interpretation.

---

## 5. Experiment 2 — LocalOracle-PC: state and update depth are separate

### 5.1 Task and access rule

Sample a function $f:[n]\to[n]$, a source $s$, and a requested depth $D$. The target is $f^D(s)$. Encode the source into $z_0$ before the boundary, then delete its token and KV; the source cannot be reread. The requested depth is the shared post-boundary continuation. The full table is never placed in the recurrent model's context.

At update $t$, a hard query head chooses one row index or `STOP`. If it chooses $v$, the environment returns only $f(v)$; this response is folded into $z_{t+1}$ and then deleted. The head uses straight-through Gumbel top-1 during training, annealing temperature linearly from $1.0$ to $0.1$ over the first 40% of updates. Evaluation uses exact argmax and one row read. `STOP` makes the state unchanged and consumes no row read.

The requested depth is part of the shared continuation and may be reread. No part of $f$, apart from the current response, is accessible. This interface tests $T\ge D$ and $dp\ge\lceil\log_2n\rceil$, not a claim about a transformer that can see the whole table.

### 5.2 Data manifest

- Train: 120,000 functions, $n\in\{16,32,64,128\}$, $D\in\{1,2,3,4\}$, balanced by cell.
- Development: 12,000 fresh functions with the same support.
- Test: 10,000 fresh examples for each $n\in\{64,128,256,512\}$ and $D\in\{1,2,4,6,8,12,16\}$.
- Joint OOD headline: $n=512,D=16$.
- Paired impossibility test: for each frozen deterministic model, generate 5,000 lazy-oracle transcripts at $n=512$ for every $D\in\{4,8,12,16\}$ and $T<D$. Answer all of the model's first $T$ adaptive queries identically in two partial functions, then complete the unassigned rows so the two $D$-step endpoints differ. Publish the completion algorithm and seeds.

Generate a simple path through step $D$ and fill nonpath rows independently. Group every graph, all of its sources, and all paired completions into one split.

### 5.3 Resource sweep

- $T\in\{0,1,2,4,6,8,12,16\}$, interpreted as a maximum number of updates.
- Let $b_n=\lceil\log_2n\rceil$. Evaluate state budgets $dp\in\{\lfloor b_n/2\rfloor,b_n-1,b_n,b_n+2,2b_n\}$.
- For each realizable budget, compare matched-bit $(d,p)$ shapes with $p\in\{1,2,4,8\}$.
- Primary test has $L=0$. A fixed-width binary pointer trace of $L=T b_n$ is a separate theorem check, not extra free memory.

The 4B fixed-budget confirmation uses the Cartesian cross

\[
T-D\in\{-2,-1,0,+1,+2\},\qquad
dp-b_n\in\{-2,-1,0,+2\},
\]

where the nonnegative parameterization is realizable, plus the joint OOD cell. The 8B and 14B replications use the four corners and the predicted boundary. Do not train every grid point at every size.

### 5.4 Controls

- a trained pause-token model with the same number of backbone calls but no oracle rows;
- a model that sees the whole function table, labeled **outside the local-oracle model**;
- a natural textual pointer trace;
- the fixed binary pointer construction from the current draft as an achievability ceiling;
- teacher-forced current-row access and intermediate vertex supervision as separate oracle upper bounds.

### 5.5 Causal mechanism tests

1. **Current-vertex decode.** At every step, decode $v_t=f^t(s)$ from the hard state. Use held-out node relabelings.
2. **Formal state interchange.** Run two sources under the same $f$, replace the recipient state at step $t$ by the donor state, and score the counterfactual endpoint
   \[
   f^{D-t}(v_t^{\mathrm{donor}}).
   \]
3. **Causal light cone.** Change the oracle response at hop $j$. State hashes through $j$ must be identical; later decoded vertices must follow the modified chain.
4. **No-op insertion.** Insert a legal update with no response. A genuine hop executor should not advance the represented vertex.
5. **Forced stopping.** Compare stopping at $D-1,D,D+1$.
6. **Paired completion test.** At $T<D$, the lazy-oracle completion pair has the same entire observed query-response transcript, not merely the same local path prefix. The retained computation must therefore be identical, so at least one endpoint answer is wrong. Fix all decoding randomness before constructing the pair.

### 5.6 Outcomes and decision rule

Primary: a two-dimensional accuracy surface over $T-D$ and $dp-b_n$, estimated transition locations, and state-interchange accuracy.

Secondary: row reads, decoded-current-vertex accuracy by step, STOP calibration, no-op advance rate, and graph-size/depth OOD accuracy.

A convincing mechanism requires all three:

- increasing $d$ or $p$ cannot compensate for $T<D$;
- when $dp\ge b_n$, the transition occurs near $T=D$;
- state interchange produces the theorem-predicted counterfactual endpoint.

A high-accuracy probe with a weak interchange effect is epiphenomenal encoding, not evidence that the state executes the pointer chain.

---

## 6. Experiment 3 — EvidenceGraph-QA on MuSiQue and 2WikiMultiHopQA

### 6.1 Why these datasets

Use [MuSiQue](https://aclanthology.org/2022.tacl-1.31/) (TACL 2022) because it supplies 2–4-hop composed questions and reasoning graph structures. Use [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/) (COLING 2020) because its evidence triples and reasoning types provide concrete labels for bridge-state analysis. These are real Wikipedia-based multihop QA datasets, not synthetic graph renderings.

### 6.2 Evidence-locked transformation

Run the untouched official format as an ecological reference, but base all mechanism claims on an evidence-locked transformation:

1. retain only examples whose answer and every labeled bridge are grounded in the supplied supporting evidence;
2. consistently replace named entities, bridge entities, and answer spans with per-instance opaque aliases such as `ENT_4Q7M`; aliases are freshly sampled for every example and never shared across splits;
3. place evidence passages **before** the read-once boundary and reveal the question only after the prefix and its KV are deleted;
4. preserve all gold passages and add $K\in\{0,4,8,12\}$ topic-matched distractors;
5. cap each passage at 112 tokenizer tokens by retaining the labeled evidence sentence and its nearest context; cap the complete prefix at 2,048 tokens and remove distractors before any gold text;
6. construct a continuation bank from the published decomposition questions or evidence triples: the final question, hop subquestions, bridge-identification queries, and a matched counterfactual final query all share one evidence prefix;
7. create a counterfactual twin by changing one supported relation object to a same-type unused alias and recomputing the downstream answer from the provided decomposition/evidence graph;
8. automatically check entity consistency and answer derivability, then manually audit 200 examples per dataset before freezing the manifest.

All continuations from one evidence prefix and both counterfactual twins stay in the same split. This grouping is essential; otherwise the model can memorize the prefix across train and test queries.

### 6.3 Splits and training sets

- **MuSiQue:** use the official answerable training split; stratify development and test reporting by 2, 3, and 4 hops and by chain versus branching graph. Use inverse-frequency sampling so each hop stratum contributes equally during training.
- **2WikiMultiHopQA:** use the official train, development, and test partitions; stratify by comparison, inference, compositional, and bridge-comparison type. Subsample at most 40,000 grouped training prefixes, balanced by type, so compute remains comparable with MuSiQue.
- Reserve 10% of the official training groups as an internal development set only when a public test label is unavailable. Freeze this choice before model training.
- Evaluate at least 2,000 grouped prefixes per dataset on the final labeled split, or every available group if fewer survive filtering.

Release a preprocessing manifest with original example ID, support IDs, hop/type label, aliases, distractor IDs, counterfactual edit, token count, and split. Report the exact realized counts after filtering; do not silently replace failed examples.

### 6.4 Resource and access conditions

Evaluate three access modes:

1. **Strict read-once:** all evidence is consumed before the boundary and cannot be reread.
2. **Sealed paragraph:** at each recurrent step, a hard selector reads at most one paragraph, which is immediately discarded. This is an empirical access model; do not call it the pointer theorem unless the same oracle restrictions hold.
3. **Unrestricted full context:** every step may attend to the entire evidence set. This is an ecological ceiling outside the continuation bound.

Use:

- $T\in\{0,1,2,4,6,8\}$;
- $p\in\{1,2,4,8\}$;
- retained budgets $B\in\{64,128,256,512,1024,2048\}$ bits;
- latent-only, binary-only, and 50/50 hybrid allocations at every realizable $B$;
- matched-shape tests at $dp=128,256,512$.

Select the five fixed 4B configurations from the transition seen on internal development data, then lock them. The 8B replication uses below/boundary/above settings. The 14B and 27B models run only the best sufficient QROLL setting, its matched pause control, and the best competitive latent baseline.

### 6.5 Training tiers

**Tier A: answer-only mechanism learning.** Direct, pause, QROLL latent, binary transcript, and hybrid models receive only final answer and evidence-ID losses. This is the headline test of whether the state mechanism is learned without privileged hop labels.

**Tier B: shared canonical-step supervision.** Explicit CoT, Coconut, CoT2, and a QROLL oracle-state variant receive the same ordered support IDs, bridge aliases, and final answer. This supports a fair architecture comparison when the baseline requires a reasoning curriculum.

**Tier C: method-faithful reproduction.** Run each published method with its prescribed training schedule and label the supervision and persistent memory it uses. This is the practical-performance table, not the controlled causal comparison.

### 6.6 Mechanistic measurements

At each step, train held-out probes for current support-passage index, current bridge alias among the instance candidates, next support index, and final answer. Probes are fit on training states and evaluated on unseen prefixes and aliases.

The causal tests are primary:

1. **Support deletion:** remove a gold paragraph before encoding. Accuracy and the corresponding bridge state should change. Deleting the original tensor after the strict boundary must have no effect and is an interface audit.
2. **Counterfactual state interchange:** run an original/counterfactual twin, patch $z_t$ from one into the other, and measure whether the final answer follows the donor evidence history.
3. **Matched random patch:** patch a random state from an example of the same hop count and norm as a negative control.
4. **State erasure by hop:** zero the state after hop $t$ while retaining the question and measure which downstream answers fail.
5. **Bridge-direction removal:** estimate a bridge subspace on training examples, project it out at test time, and compare the causal effect with a random subspace of the same rank. This is supporting evidence only; state interchange remains the decisive test.
6. **Evidence timing:** remove a support passage before versus after it has been compressed. The asymmetry checks that the model actually stores rather than rereads it.

### 6.7 Metrics and hypothesis

- Answer exact match and token F1.
- Supporting-evidence ID F1.
- Accuracy by hop count, reasoning graph/type, and distractor count.
- Original-to-aliased and original-to-counterfactual retention.
- Donor-consistent answer rate under state interchange.
- Probe selectivity and intervention effect size.
- Accuracy–FLOP, accuracy–latency, and accuracy–generated-token Pareto fronts.

Fit a mixed-effects logistic model for exact answer correctness:

\[
\operatorname{logit}P(Y=1)=
\beta_0+\beta_1T+\beta_2H+\beta_3(T\times H)
+\beta_4B+\beta_5A+\beta_6M+u_{\mathrm{seed}}+u_{\mathrm{prefix}},
\]

where $H$ is hop count, $A$ is access mode, and $M$ is method. The preregistered empirical mechanism claim requires:

- a positive hop-conditioned recurrent-compute effect under strict or sealed access;
- robustness on aliased/counterfactual evidence;
- a donor-consistent state-interchange effect beyond the matched-random patch control.

Raw improvement on the original questions is insufficient: it can arise from parametric recall, lexical retrieval, or a stronger decoder.

---

## 7. Experiment 4 — RareWitness-RAG: sampling, aggregation, precision, and work

### 7.1 Corpus and controlled real-data task

Use [MultiHop-RAG](https://openreview.net/forum?id=t4eB3zYWBK), published at COLM 2024. It contains multihop queries over a real-news article collection; its questions were constructed with model assistance and validated, so describe it as a **real-news corpus**, not as wholly human-authored QA.

For every query requiring $m\in\{2,3,4\}$ supporting documents, place $m-1$ supports in a common visible prefix and withhold one decisive support. Put the withheld support uniformly at random among $K-1$ topic- and date-matched distractor branches. Alias the answer entity and require the output to contain both the answer alias and the decisive branch ID. A deterministic verifier checks both. Parametric guessing without the branch cannot receive certified credit.

If the released data have no document-disjoint official split, build a document co-occurrence graph, assign connected components greedily to 70/15/15 train/development/test targets, and drop any query whose support documents cross partitions. Publish both the component assignment and the exact realized counts.

### 7.2 Part A: exact sampled-inspection law on real text

Use:

- $K\in\{8,16,32,64\}$;
- $r\in\{1,2,4,8,16,32,64,128\}$ independent samples with replacement;
- 1,000 query/placement instances per $(K,r)$ where the split permits, otherwise every available held-out query with enough independently randomized placements to reach 1,000;
- branch snippets of at most 112 tokens, always retaining the answer-bearing sentence;
- fixed decoding for the within-branch solver.

Under the explicitly uniform sampler, decisive-branch coverage is exactly

\[
P_{\mathrm{hit}}(K,r)=1-(1-1/K)^r.
\]

The law predicts *coverage*, not unconditional QA accuracy. Decompose certified success as

\[
P_{\mathrm{cert}}
=P_{\mathrm{hit}}P(\mathrm{cert}\mid\mathrm{hit})
+(1-P_{\mathrm{hit}})P(\mathrm{cert}\mid\mathrm{miss}).
\]

Aliasing and evidence certification should make the second conditional term approximately zero. Report both terms rather than attributing solver errors to the sampling theorem.

For the secondary full task in which all $m$ supports are hidden among $K$ exchangeable branches, compare uniform-sampler coverage with

\[
P_{\mathrm{cover}}(K,m,r)
=\sum_{j=0}^{m}(-1)^j{m\choose j}(1-j/K)^r.
\]

This inclusion–exclusion formula is an explicitly labeled multi-support extension. Do not apply either analytic law to a learned nonuniform or adaptive retriever. For learned retrieval, log the realized support masses and report empirical coverage.

### 7.3 Part B: continuous aggregation and work matching

Compare:

1. independent uniform sampled CoT;
2. self-consistency with learned sampling;
3. serial exhaustive scan;
4. QROLL latent aggregation over every branch;
5. a Reasoning-by-Superposition-style full aggregate;
6. full attention with all branches;
7. branch-read- and FLOP-matched sampled controls.

Full aggregation is allowed to remove the $r=\Omega(K)$ sampled-inspection penalty because it reads every branch. Charge $K$ branch encodings, all attention/aggregation FLOPs, memory traffic, and latency. The scientific claim is an **access separation**, not free exponential computation.

For 27B, cap $K\le32$. Larger-$K$ tests run at 0.6B/4B/8B, and the fixed construction runs without an LLM.

### 7.4 Part C: precision calibration of the hypercube construction

Keep a synthetic calibration inside this experiment so the numerical-resolution statement is tested without semantic confounds.

- $K=2^k$, $k\in\{4,6,8,10,12\}$, for the fixed construction.
- Learned LLM aggregation uses $K\in\{16,32,64,128,256\}$.
- Fractional precision $q\in\{2,4,6,8,10,12,16\}$.
- Compare normalized aggregation, in which the marked signal is $1/K$, with an unnormalized sum.
- Use 10,000 fixed-construction trials and 2,000 learned trials per cell.

The predicted normalized transition is near $q\approx\log_2K$: below that scale, the decisive $1/K$ contribution can round to zero. The unnormalized construction avoids this particular resolution failure but still pays $\Theta(K\log K)$ branch-code work. Keep the paper's $p$-bit state-cardinality parameter distinct from this fixed-point fractional resolution $q$.

### 7.5 Causal interventions and outcomes

- Remove the decisive support after selecting branch placements: certified success should collapse.
- Move the decisive support from branch $i$ to branch $j$: the returned evidence ID should follow the move.
- Patch a post-aggregation state from a counterfactual twin: the answer and evidence ID should follow the donor.
- On the hypercube calibration, flip one signed code coordinate and verify the corresponding decoded branch bit.
- Compare a learned retrieval policy with a uniform sampler only after plotting its actual support probability; favorable nonuniform access is not a violation of the law.

Primary outcomes are maximum absolute and mean absolute calibration error for $P_{\mathrm{hit}}$, certified accuracy, conditional solver accuracy, branch reads, and the accuracy-versus-work Pareto front. Secondary outcomes are aggregation resolution versus $q-\log_2K$, latency, peak memory, and evidence-ID causal effects.

---

## 8. Shared statistical protocol

Preregister one primary hypothesis for each experiment before the final runs:

- **H1:** FrontierRate error respects the Fano envelope, and the estimated transition in total retained bits lies near $B/n=1$.
- **H2:** LocalOracle-PC has separable transitions near $T=D$ and $dp=\lceil\log_2n\rceil$, with correct counterfactual state interchange.
- **H3:** On evidence-locked MuSiQue and 2Wiki, recurrent compute interacts positively with hop count and the latent state has a causal donor-consistent effect.
- **H4:** Uniform sampled coverage follows the exact hit law, while full aggregation changes the access/work frontier rather than obtaining uncharged parallelism.

Use the following analysis throughout:

- Cluster all resampling by the underlying prefix, graph, or question family, never by individual paraphrase, query, branch placement, or counterfactual twin.
- Report seed means, seed standard deviations, and hierarchical 95% confidence intervals from 10,000 bootstrap replicates that resample seeds and then grouped instances.
- Use paired bootstrap differences because methods share examples and seeds.
- Apply Holm correction across H1–H4. Treat secondary analyses as estimation, not another search for significance.
- Fit monotone logistic change-point models for $B/n$, $T-D$, and $dp-\log_2n$. Report transition estimates and confidence intervals.
- Use an equivalence margin of ±1.5 percentage points when claiming a plateau or absence of an effect.
- For real-data EM/F1, use the grouped hierarchical bootstrap; for exact answer accuracy, also report the mixed-effects model in E3.
- For E4, report binomial intervals and calibration residuals, not only a fitted curve.
- Treat 14B and 27B as replications of a directional effect, not enough points to infer a scaling law.

Freeze preprocessing, resource cells, prompt templates, checkpoint selection, intervention definitions, and the four primary tests before opening final test results.

---

## 9. Compute plan and scale gate

### 9.1 Hardware settings

The full program is feasible with an RTX 6000 Ada 48 GB plus optional cloud replicas, but it should be staged.

| Backbone | Weight loading | Typical microbatch | Expected peak VRAM | Maximum primary sequence |
|---|---|---:|---:|---:|
| Qwen3-0.6B | BF16 | 16 | 10–14 GB | 2,048 |
| Qwen3-4B | BF16 | 2 | 34–42 GB | 2,048 |
| Qwen3-8B | NF4 double quantization | 1–2 | 29–38 GB | 2,048 |
| Qwen3-14B | NF4 double quantization | 1 | 36–45 GB | 2,048 |
| Gemma-3-27B-pt | NF4, text tower, attention-only LoRA | 1 | 43–48 GB | 2,048 |

Profile each model for 100 training steps before committing the run matrix. If 27B exceeds 48 GB, use two 48 GB GPUs; do not use CPU offload in timed comparisons.

### 9.2 Stages

| Stage | Runs | Gate | Planning envelope |
|---|---|---|---:|
| 0: implementation audit | 0.6B supernets, exact-state logging, leak tests, 100-step profiles | All deletion and analytic construction tests pass | 30–50 Ada-hours |
| 1: primary | 4B fixed-budget models, five seeds, all four experiments | At least three of four primary effects pass their directional gate | 380–460 Ada-hours |
| 2: within-family replication | 8B, three seeds, below/boundary/above cells | Direction and causal intervention replicate | 140–190 Ada-hours |
| 3: large confirmation | 14B selected cells; 27B real-data cells only | Run only if the 4B mechanism gate passes | 180–260 Ada-hours |
| Final causal/statistical sweep | locked interventions, profiling, final bootstrap | No interface violation | 40–60 Ada-hours |

Expected full envelope: roughly **800–1,000 RTX 6000 Ada GPU-hours**, including failed-run reserve. A lean submission omitting 27B and most 14B cells is roughly **430–500 GPU-hours**. These are planning estimates; replace them with measured profiler extrapolations and actual compute in the paper.

### 9.3 Large-model launch gate

Do not launch 14B or 27B merely to enlarge a null result. Proceed only if the locked 4B analysis shows:

- at least a 20-point accuracy change across the FrontierRate boundary;
- at least a 20-point pointer gain from $T=D-1$ to $T=D$ at sufficient state budget;
- QROLL exceeds the matched pause control by at least five points in the pooled evidence-locked real-data analysis, with a paired 95% interval excluding zero and the same sign on both datasets;
- a state-swap, state-erasure, or precision intervention changes the answer in the predicted direction;
- no prefix/KV leakage is detected.

If this gate fails, the correct paper is a smaller, honest negative result about optimization failing to realize the constructions—not a larger benchmark sweep.

---

## 10. Required main-paper figures and tables

### Figures

1. **Retained-information frontier:** queried-bit error versus $B/n$, with the exact lower envelope, iso-budget latent/transcript allocations, and a lower panel for collision-conflict rate.
2. **Pointer phase diagram:** accuracy over $T-D$ and $dp-\lceil\log_2n\rceil$, alongside current-vertex probe accuracy and causal interchange accuracy.
3. **Natural evidence mechanism:** MuSiQue and 2Wiki accuracy by hop count under strict, sealed, and full-context access; a paired panel for donor-consistent state-swap effects.
4. **Rare witness and work:** measured coverage versus $r/K$ with the exact law, plus certified accuracy versus branch reads/FLOPs for sampled and aggregate methods.
5. **Precision appendix figure:** normalized and unnormalized aggregation versus $q-\log_2K$.

### Tables

1. Interface audit and charged resources for every method.
2. Primary 4B and replication 8B results with confidence intervals and causal effects.
3. MuSiQue/2Wiki original, aliased, and counterfactual results.
4. MultiHop-RAG coverage, conditional solver accuracy, certified success, branch reads, FLOPs, latency, and VRAM.
5. Training supervision, trainable parameters, and actual GPU-hours.

The main text should emphasize the phase diagrams and interventions. Put the full grid, prompts, example transformations, probes, and per-seed tables in the appendix.

---

## 11. Interpretation and claim gates

### Claims supported if all relevant tests pass

- Learned read-once LLMs exhibit a joint retained-information transition governed by $dp+L$, rather than dimension alone.
- Under enforced local access, a recurrent finite state causally tracks pointer progress one hop per update.
- The same controlled state can carry causally relevant bridge information on aliased natural multihop evidence, although the formal pointer lower bound is not claimed for unrestricted QA.
- Sampled explicit exploration and continuous full aggregation occupy different access regimes; the aggregate avoids sampling failure by processing all branches and must be charged for that work.
- Numerical resolution can be a separate bottleneck from the number of representable states.

### Claims not supported by this program

- Continuous states are inherently more expressive than an unrestricted discrete scratchpad.
- Dimension alone determines capacity.
- $T\ge D$ for a full-context transformer or any model that can preprocess the complete pointer table.
- Probe decodability alone reveals the reasoning mechanism.
- Latent aggregation performs exponential search for free.
- Better accuracy on unmodified real questions proves use of the supplied evidence.

### Failure meanings

| Observation | Correct interpretation |
|---|---|
| Below-envelope FrontierRate success | Uncounted memory or leakage until proven otherwise |
| Above-boundary FrontierRate failure | Optimization/decoder limitation; not a theorem failure |
| Pointer success for $T<D$ only with full-table access | Scope demonstration, not contradiction |
| High pointer probe, weak state interchange | Decodable but noncausal representation |
| Natural-QA gains disappear under aliasing | Parametric/lexical shortcut |
| Natural-QA gains survive aliasing but not state patching | Useful recurrence without the claimed bridge-state mechanism |
| Sampled coverage above the uniform law | Nonuniform access, dependence, or a side channel |
| Full aggregate wins in accuracy but loses after work matching | Access advantage without an efficiency advantage |

---

## 12. Integration with the current draft

1. Keep the current deterministic BFS, fixed-code PointerChase, sampling simulation, and capacity calculator as **construction-verification appendix material**.
2. Keep the learned structured PointerChase result as a pilot motivating LocalOracle-PC, but do not mix its learned identifier embeddings with the new hard finite-state capacity claim.
3. Move GraphQA, CLUTRR, and bAbI learned controls to an appendix titled “Preliminary structural diagnostics.” Their flat or weak $T$ curves motivate the stronger read-once and causal design; they are not evidence for the new mechanism claim.
4. Replace the present main experimental narrative with E1–E4. The theory section does not need to change except for a short forward reference to the exact experimental interfaces.
5. Use one terminology table consistently: state dimension $d$, bits per coordinate $p$, persistent state bits $dp$, recurrent updates $T$, fixed binary transcript bits $L$, sampled traces/branch reads $r$, branch count $K$, witness mass $\mu$, and fractional resolution $q$.
6. State in every figure caption whether the result is a theorem instantiation, a learned mechanism test, an ecological transfer, or an unrestricted control.

---

## 13. Validated primary sources for implementation and positioning

### Datasets

- Trivedi et al., [“MuSiQue: Multihop Questions via Single-hop Question Composition,”](https://aclanthology.org/2022.tacl-1.31/) *Transactions of the Association for Computational Linguistics*, 2022.
- Ho et al., [“Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps,”](https://aclanthology.org/2020.coling-main.580/) *COLING*, 2020.
- Tang and Yang, [“MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries,”](https://openreview.net/forum?id=t4eB3zYWBK) *COLM*, 2024.

### Current reasoning comparisons

- Wei et al., [“Chain-of-Thought Prompting Elicits Reasoning in Large Language Models,”](https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) *NeurIPS*, 2022.
- Wang et al., [“Self-Consistency Improves Chain of Thought Reasoning in Language Models,”](https://openreview.net/forum?id=1PL1NIMMrw) *ICLR*, 2023.
- Goyal et al., [“Think Before You Speak: Training Language Models With Pause Tokens,”](https://proceedings.iclr.cc/paper_files/paper/2024/file/76917808731dae9e6d62c2a7a6afb542-Paper-Conference.pdf) *ICLR*, 2024.
- Hao et al., [“Training Large Language Models to Reason in a Continuous Latent Space,”](https://openreview.net/forum?id=Itxz7S4Ip3) *COLM*, 2025.
- Zhu et al., [“Reasoning by Superposition,”](https://proceedings.neurips.cc/paper_files/paper/2025/hash/72c363c2a573ca2128bd176d3317696b-Abstract-Conference.html) *NeurIPS*, 2025.
- Zhang et al., [“Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space,”](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f7396d1c54d51416958d63e285377103-Abstract-Conference.html) *NeurIPS*, 2025.
- Gozeten et al., [“Continuous Chain of Thought Enables Parallel Exploration and Reasoning,”](https://openreview.net/forum?id=sTPKDKn5ig) *ICLR*, 2026.

These are publication or official model/dataset records, not secondary web summaries. Reconcile their BibTeX keys with `references.bib` when this plan is incorporated into the manuscript.

---

## 14. Minimal submission version

If compute or time becomes binding, preserve the causal and theorem-facing core:

- E1 and E2 at 0.6B and 4B, with five 4B seeds;
- E3 on both MuSiQue and 2Wiki at 4B, with 8B replication only at below/boundary/above settings;
- E4 exact sampling calibration and 4B aggregate/work comparison on MultiHop-RAG;
- explicit CoT, Coconut, CoT2, and pause controls; Soft Thinking may be evaluation-only;
- all interface audits and causal patches;
- no 27B run unless the primary effects have already passed.

That lean version is scientifically stronger than a wide but underpowered sweep across many models and benchmarks.
