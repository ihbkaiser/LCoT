---
bibliography: references.bib
link-citations: true
---

# Finite-Precision Continuous Chain-of-Thought: Serialization, Frontier Capacity, and Sampling Lower Bounds

## Abstract

Continuous latent chain-of-thought (CoT) breaks the usual identification of reasoning effort with generated-token count. We study a finite-precision model with a $d$-coordinate persistent state, at most $p$ bits per coordinate, and $T$ recurrent updates. Three qualifications are essential: finite latent states can be serialized by an unrestricted discrete scratchpad; state lower bounds require a specified continuation or access model; and a continuous superposition is not classical nondeterminism. Within this scope, we prove (i) a finite-grid serialization bound, (ii) a joint latent-state/retained-transcript continuation bound, including the randomized approximate frontier tradeoff $dp+\log_2 M_\Sigma(L)\ge n[1-h_2(\varepsilon)]$, where $M_\Sigma(L)$ counts transcripts of length at most $L$, (iii) tight state, update, and fixed-width trace requirements for local-oracle pointer chasing, and (iv) an exact inverse-mass law for independent sampled traces. A $d$-coordinate hypercube code supplies $2^d$ robust labels and therefore an $\Omega(2^d)$ lower bound for sampled inspection, while explicitly exposing the construction's $\Theta(d2^d)$ aggregation work. Fixed-construction sweeps realize the predicted capacity, update-depth, and trace-coverage transitions. Structured diagnostics on GraphQA, CLUTRR, and bAbI, together with learned controls, test whether the same accounting remains informative on language-rendered tasks; the learned evidence is mixed. The results support reporting latent state capacity, recurrent depth, and sampling budget alongside answer accuracy and verbal-CoT length.

## 1. Introduction

Explicit CoT prompting established the utility of generated intermediate reasoning [@wei2022cot], and self-consistency improved answer selection by sampling multiple such traces [@wang2023selfconsistency]. Continuous latent reasoning instead replaces some generated tokens by recurrent hidden-state computation: Coconut feeds hidden states back as input embeddings [@hao2025coconut], while looped transformers formalize repeated latent computation [@saunshi2025latent]. These developments make generated-token count an incomplete proxy for inference effort: a model can emit no rationale while using many recurrent updates, or sample many explicit traces while rarely visiting the decisive branch.

The corresponding theory requires care. A continuous vector is not automatically nondeterministic. Under finite precision it is a finite state; under exact real arithmetic it may carry unbounded analog information; and under superposition semantics it is a structured parallel aggregate rather than a set of freely accessible branches. We therefore ask what can be concluded from finite state capacity alone and introduce stronger access restrictions only where a step or sampling lower bound needs them.

### 1.1 Related Work

**Explicit, hidden, and compressed computation.** Dot-by-dot computation shows that extra token positions can provide hidden computation without a semantic rationale [@pfau2024dot], while PENCIL demonstrates that explicit scratchpads can erase obsolete intermediate state and therefore need not retain a full fixed-width history [@yang2025pencil]. Formal analyses characterize the expressivity of transformers with serial CoT [@merrill2024cot; @li2024serial] and the learnability of autoregressive CoT [@joshi2025theory]. Accordingly, our trace lower bound is deliberately restricted to a context-free, fixed-width state code at every step; it is not a lower bound for arbitrary erasable scratchpads.

**Continuous latent reasoning.** Coconut introduced recurrent continuous thoughts [@hao2025coconut], and looped transformers give an explicit recurrent-computation model [@saunshi2025latent]. Reasoning by Superposition represents a reachability frontier as a latent aggregate [@zhu2025superposition], while CoT2 analyzes parallel exploration and dimension-dependent continuous-thought resources [@gozeten2026cot2]. These are the closest comparisons to our setting. Our contribution is narrower than a new latent-reasoning method: we identify which finite-state, continuation, and sampled-inspection claims remain valid after the access model and uncharged work are made explicit.

**Computational assumptions and graph primitives.** Transformer lower bounds and parallelism tradeoffs depend on precision and architectural assumptions [@hahn2020limitations; @merrill2023parallelism]; a recent survey systematizes the roles of precision, positional encoding, masking, attention type, and uniformity [@strobl2024survey]. Hard-attention Turing completeness and analog neural computation further show why exact-real assumptions cannot be imported silently into a finite-grid model [@perez2021attention; @siegelmann1994analog]. For graph reasoning, Sanford et al. analyze transformer depth, width, and extra-token resources [@sanford2024graph]. Classic pointer chasing also has communication-round lower bounds [@nisan1993rounds], but our theorem is not a communication-complexity result: it charges one local oracle query per persistent update.

### 1.2 Research Questions

- **RQ1:** Which comparisons between continuous and discrete reasoning follow from finite persistent-state capacity alone?
- **RQ2:** How do exact and approximate frontier reasoning trade persistent latent capacity against a retained explicit transcript?
- **RQ3:** Under a local-access model, which state, update, and explicit-trace budgets are necessary for pointer chasing?
- **RQ4:** When does sampling explicit branches fail to recover a low-mass signal represented by a parallel aggregate?

### 1.3 Contributions

1. **Finite-state model and serialization.** We define $\mathsf{CCoT}(d,p,T)$ and prove a state-serialization upper bound of $(T+1)\lceil dp/\log_2|\Sigma|\rceil+O(T)$ tokens. This rules out an unconditional representational separation from unrestricted discrete scratchpads.
2. **Hybrid continuation tradeoff.** We formulate a continuation principle for a $dp$-bit latent state together with an at-most-$L$-token retained transcript. If $N$ prefixes are pairwise distinguishable by shared continuations, then $dp+\log_2 M_\Sigma(L)\ge\log_2N$, where $M_\Sigma(L)=\sum_{\ell=0}^L|\Sigma|^\ell$. For a uniformly random $n$-bit frontier queried at a uniformly random vertex, average error $\varepsilon$ requires $dp+\log_2M_\Sigma(L)\ge n[1-h_2(\varepsilon)]$. The exact state-only bound $dp\ge n$ is the special case $L=0$ and $\varepsilon=0$.
3. **A tight local pointer-chasing law.** In a one-query-per-update oracle model, exact $D$-step pointer chasing for $D\le n$ has worst-case update complexity $D$ and state capacity at least $\log_2 n$ bits, matched by a Boolean binary-code construction. Requiring a fixed-width decodable state at every explicit step additionally costs $D\lceil\log_{|\Sigma|}n\rceil$ tokens.
4. **Exact sampled-inspection law.** Independent traces that hit a decisive branch with mass $\mu$ succeed with probability at most $1-(1-\mu)^r$. A hypercube family supplies $2^d$ robust labels and an $\Omega(2^d)$ sampled-inspection lower bound, with the construction's exponential aggregation work stated explicitly.
5. **Theory-aligned diagnostics.** Controlled sweeps isolate the predicted $dp$, $T$, fixed-width $L$, and $r$ axes. Structured GraphQA, CLUTRR, and bAbI checks and learned text controls assess how far this accounting transfers to language-rendered inputs without treating deterministic formula checks as evidence of learned reasoning.

### 1.4 Claim Scope

The serialization result concerns persistent finite-state capacity, while the continuation result concerns the total retained post-prefix summary. Update-depth lower bounds use the explicitly defined local-oracle model; per-step trace lower bounds use fixed-width, context-free decodability; and sampling lower bounds use sampled inspection with no side channel to the decisive branch. We do not claim that continuous CoT is classical nondeterminism, that it outperforms unrestricted coordinate-serializing scratchpads, that the hypercube construction saves total work, or that current LLMs learn the analyzed update rules from answer-only supervision.

## 2. Finite-Precision Latent CoT

For input $x$ of length $n$, define a finite-precision continuous-CoT computation by

$$
z_0=E_n(x),\qquad
z_{t+1}=Q_p(U_n(x,z_t)),\qquad
\hat y=V_n(x,z_T),
$$

where $z_t\in Q_p^{d(n)}$ and $|Q_p|\le 2^{p(n)}$. We write this model as $\mathsf{CCoT}(d,p,T)$. The resources are:

- $d$: latent dimension;
- $p$: stored bits per coordinate, defined by coordinate-alphabet cardinality;
- $T$: number of latent update steps.

This model makes the persistent bandwidth of one latent thought at most $dp$ bits. The update rule may be implemented by a transformer block, threshold circuit, or another specified finite computation. State capacity does not by itself bound the size, depth, or arithmetic precision of that update; any theorem about those resources must state an additional access or implementation model.

**Standing assumption: finite-precision latent computation.** Throughout the theoretical sections, a $\mathsf{CCoT}(d,p,T)$ computation satisfies the following assumptions.

1. **Rounded latent state.** For each input length $n$, the persistent latent state after every thought step lies in the finite grid $Q_{p(n)}^{d(n)}$, with $|Q_{p(n)}|\le 2^{p(n)}$. Any real-valued intermediate quantity produced inside $U_n$ is not a persistent latent state unless it is rounded back into this grid.
2. **Fixed finite update family.** The maps $E_n$, $U_n$, and $V_n$ are fixed finite procedures for length $n$. The next persistent state is determined by $(x,z_t)$ through $Q_p(U_n(x,z_t))$; there is no hidden side channel carrying additional real-valued memory across steps.
3. **Resource accounting.** The stored latent bandwidth is $dp$ bits per step and the computation has $T$ persistent latent updates. If an implementation uses additional internal precision, normalized aggregation, randomness, or auxiliary memory, those resources must be specified separately and are not included for free in $\mathsf{CCoT}(d,p,T)$.
4. **Exactness and comparison models.** Unless a theorem states otherwise, correctness is exact over the stated task family under the rounded update model. Results about explicit traces, sampled traces, or arbitrary discrete scratchpads use the additional restrictions stated in those theorems; in particular, Theorem 1 allows the discrete simulator to evaluate and serialize the same rounded update rule.

**State-accounting scope.** With unrestricted access to $x$ and no complexity charge on $E_n$, $U_n$, or $V_n$, the base $\mathsf{CCoT}(d,p,T)$ definition is a persistent-state model, not a time-, query-, or circuit-complexity model. In particular, $T$ alone has no lower-bound meaning in the base model: an unrestricted encoder or verifier could compute the answer directly. Every statement about update necessity, prefix indistinguishability, or sampling therefore uses the theorem-specific access interface stated below; no such lower bound is attributed to the unrestricted base model.

**Continuation model.** A prefix $a$ and continuation $c$ determine an answer $H(a,c)$. At the prefix boundary, the consumed prefix becomes inaccessible; subsequent updates receive only a retained summary and the shared continuation. The summary contains a latent state $z\in Q_p^d$ and, when explicitly allowed, a transcript $\tau\in\Sigma^{\le L}$. Define $a\equiv_Ha'$ when $H(a,c)=H(a',c)$ for every admissible shared continuation $c$. The number of equivalence classes, denoted $N_H$, is the task's continuation index. This is a Myhill--Nerode-style interface, specialized to the task and legal continuations. Correctness for all continuations requires the retained summary to distinguish these classes. Write

$$
M_\Sigma(L)=|\Sigma^{\le L}|=\sum_{\ell=0}^L|\Sigma|^\ell.
$$

The state-only model is the special case $L=0$; a fixed-length transcript has exactly $|\Sigma|^L$ possible values.

**Local-oracle pointer model.** For pointer chasing, the input is an oracle $f:[n]\to[n]$ and a source $s$. After initialization, the algorithm has no direct side channel to $(f,s)$: each persistent update may make at most one oracle query, receives the response, and updates a state in $Q_p^d$; the final answer is decoded from the persistent state. This restriction is used only for Theorem 4 and matches the controlled pointer-update experiments.

**Fixed-point resolution.** The cardinality condition $|Q_p|\le2^p$ supports state-counting claims but says nothing about the numerical spacing of grid points. Whenever we discuss whether a magnitude $1/K$ survives normalization, we separately assume a fixed-point quantum $\Delta_q=2^{-q}$ that maps magnitudes below $\Delta_q/2$ to zero. The fractional-resolution parameter $q$ is not inferred from state cardinality alone.

The access assumptions and charged resources are summarized below.

| Result | Input/access model | Charged resources | Explicitly uncharged |
|--------|--------------------|-------------------|----------------------|
| Serialization | common rounded transition oracle | recorded state tokens | cost of evaluating $U_n$ |
| Continuation index | read-once prefix, shared continuation | post-prefix state bits and retained transcript tokens | work within a continuation update |
| BFS construction | read-only adjacency, specified one-hop recurrence | persistent bits, recurrence steps, threshold-circuit size | training cost |
| Pointer chasing | one adaptive query to $f$ per update | state bits, oracle updates, fixed-width trace tokens | full-table preprocessing, which is disallowed |
| Sampled inspection | independent branch inspections with witness certification | number of inspections | uninspected branch contents |
| Hypercube code | full access to all $K$ markers | output state and $\Theta(Kd)$ work | none of the $K$ marker accesses |

## 3. Theoretical Results

### Theorem 1: Serialization Upper Bound

For any finite alphabet $\Sigma$ with $|\Sigma|\ge 2$, every $\mathsf{CCoT}(d,p,T)$ latent trace can be serialized using at most

$$
(T+1)\left\lceil\frac{dp}{\log_2|\Sigma|}\right\rceil+O(T)
$$

tokens including delimiters, assuming the discrete simulator can evaluate the same rounded update rule.

**Proof sketch.** Each coordinate has at most $2^p$ values, so a latent vector has at most $2^{dp}$ states and can be encoded in $dp$ bits. A $\Sigma$-token carries $\log_2|\Sigma|$ bits of code capacity. Serializing each $z_t$ and inducting over the deterministic rounded update reproduces the latent trace exactly.

**Implication.** Finite precision alone cannot yield a representation-level separation from an unrestricted discrete scratchpad. The theorem accounts for serialized persistent state, not the simulator's model-dependent cost of evaluating $U_n$.

### Theorem 2: Hybrid Continuation-Index Tradeoff

Let $H$ be a prefix--continuation task with continuation index $N_H$. Any deterministic finite-precision computation whose retained post-prefix summary belongs to $Q_p^d\times\Sigma^{\le L}$ and that is exactly correct for every admissible continuation satisfies

$$
2^{dp}M_\Sigma(L)\ge N_H,
\qquad
dp+\log_2M_\Sigma(L)\ge\log_2N_H.
$$

For a fixed-length $L$-token transcript, this becomes $dp+L\log_2|\Sigma|\ge\log_2N_H$. In particular, if a family $\mathcal F$ of frontier states must be exactly distinguished, then $N_H\ge|\mathcal F|$. With no retained transcript, arbitrary frontiers over $[n]$ give $dp\ge n$, while frontiers of size at most $k$ give

$$
dp\ge \log_2\sum_{i=0}^k {n\choose i}.
$$

**Proof sketch.** If two continuation-inequivalent prefixes share a summary $(z,\tau)$, a continuation that distinguishes them is processed from the same retained information and forces the same answer, contradicting exactness. Thus the $N_H$ equivalence classes inject into $Q_p^d\times\Sigma^{\le L}$, which has at most $2^{dp}M_\Sigma(L)$ elements.

**Scope.** The bound is on total retained information, not dimension alone, and it requires continuation distinguishability. It does not lower-bound a graph-reachability algorithm that can reread the full graph after the boundary.

### Corollary 2.1: Layered Frontier-State Lower Bound

Consider a one-pass layered reachability model in which a deterministic finite-precision latent state processes a graph prefix, then must be sufficient for all possible future layer continuations. If the prefix can induce any frontier $S\subseteq[n]$, then exact correctness for all continuations requires at least $2^n$ latent states, hence $dp\ge n$.

**Proof sketch.** For every $S\subseteq[n]$, construct a prefix whose reachable frontier is exactly $S$. If $S\neq S'$, choose $v\in S\triangle S'$ and append the same continuation that connects only $v$ to the target. Hence all $2^n$ frontiers are pairwise continuation-inequivalent, and Theorem 2 applies.

### Corollary 2.2: Randomized Approximate Hybrid Frontier Retrieval

Let $S$ be uniform on $\{0,1\}^n$, let $V$ be an independent uniform index in $[n]$, and let a possibly randomized one-pass encoder retain $(Z,\tau)\in Q_p^d\times\Sigma^{\le L}$. A decoder receives the retained summary and $V$ and predicts $S_V$. If its average error is at most $\varepsilon\in[0,1/2)$, then

$$
dp+\log_2M_\Sigma(L)
\ge n\bigl[1-h_2(\varepsilon)\bigr],
$$

where $h_2(u)=-u\log_2u-(1-u)\log_2(1-u)$ is binary entropy. Thus a latent state and a retained explicit transcript substitute bit for bit in the exact case, while any constant error below $1/2$ still requires $\Omega(n)$ total retained information.

**Proof sketch.** Condition on any public randomness $R$. If $\varepsilon_i$ is the error when $V=i$, binary Fano bounds give $H(S_i\mid Z,\tau,R)\le h_2(\varepsilon_i)$. Subadditivity and concavity yield $H(S\mid Z,\tau,R)\le n h_2(\varepsilon)$. Hence $I(S;Z,\tau\mid R)\ge n[1-h_2(\varepsilon)]$, whereas the summary alphabet gives $I(S;Z,\tau\mid R)\le dp+\log_2M_\Sigma(L)$.

### Theorem 3: Continuous BFS Upper Bound

For directed graph adjacency matrix $A\in\{0,1\}^{n\times n}$, source $s$, and latent state $z_r\in\{0,1\}^n$, initialize $z_0=e_s$ and update

$$
z_{r+1}[v]=\mathbf 1\left[z_r[v]+\sum_{u=1}^n z_r[u]A_{uv}\ge 1\right].
$$

After $D$ latent steps, $z_D[t]=1$ iff $t$ is reachable from $s$ by a path of length at most $D$. The persistent state has $d=n$ and Boolean precision $p_{\mathrm{state}}=1$. One update is implementable by a depth-two threshold circuit with $O(n^2)$ gates and read-only access to $A$. For this one-hop recurrence, the update bound is tight: on a directed path whose target is at distance $D$, $T<D$ cannot activate the target. If normalized aggregation produces a smallest nonzero message $1/n$, retaining it under the fixed-point model requires $q=\Omega(\log n)$ fractional bits.

**Proof sketch.** Induct on $r$. The update preserves all already reached vertices and activates every outgoing neighbor of a reached vertex. Threshold gates compute edge activations and OR them by destination.

### Theorem 4: Tight Local-Oracle Resources for Pointer Chasing

Fix $1\le D\le n$. In the local-oracle model, computing $f^D(s)$ exactly is achievable with $\lceil\log_2 n\rceil$ Boolean latent coordinates and $T=D$ updates. Conversely, every deterministic exact solver requires, in the worst case,

$$
T\ge D,\qquad dp\ge\log_2n.
$$

If an explicit trace contains $D$ fixed-width segments of $\ell$ tokens over alphabet $\Sigma$, and each segment must decode the current vertex for every input, then

$$
\ell\ge\left\lceil\log_{|\Sigma|}n\right\rceil,
\qquad
L=D\ell\ge D\left\lceil\log_{|\Sigma|}n\right\rceil.
$$

**Proof sketch.** The upper bound stores $\mathrm{bin}(v_t)$ and queries $f(v_t)$ once per update. For the update lower bound, an adversary answers queries consistently while extending the pointer path only when its current endpoint is queried. After fewer than $D$ queries, that endpoint remains unqueried and admits two oracle completions with different $D$-step endpoints. For state capacity, choose the identity oracle: the $n$ sources require $n$ different final outputs and hence $n$ distinguishable final states. Finally, $|\Sigma|^\ell\ge n$ is necessary for a fixed-width segment to decode any current vertex.

**Scope.** The $T$ lower bound is a sequential-oracle statement, not a claim about an unrestricted update that can read the full table and compute $f^D$ internally. The $L$ bound is for fixed-width per-step state logging; arbitrary scratchpads remain covered by Theorem 1.

### Theorem 5: Exact Rare-Branch Sampling Law

Suppose a sampled-trace simulator has no side channel identifying the decisive branch and can succeed only if at least one of $r$ independent traces inspects a decisive branch of mass $\mu\in(0,1)$. Then success probability is at most

$$
1-(1-\mu)^r \le r\mu.
$$

Thus success probability at least $1-\delta$ requires

$$
r\ge
\left\lceil\frac{\log(1/\delta)}{-\log(1-\mu)}\right\rceil
=\Omega\!\left(\frac{\log(1/\delta)}{\mu}\right)
$$

uniformly for $0<\mu\le1/2$, and the union bound gives the assumption-light necessity $r\ge(1-\delta)/\mu$. If the decisive branch is uniform among $K$ branches, $r=\Omega(K)$ for constant success.

**Proof sketch.** The probability of missing the branch in all independent samples is $(1-\mu)^r$. Rearranging $(1-\mu)^r\le\delta$ gives the exact threshold; the union bound gives the adaptive extension whenever every conditional hit probability is at most $\mu$.

### Theorem 6: Hypercube Rare-Witness Search Code

For every $d\ge 1$, let $K=2^d$ and assign each branch a distinct code $c_i\in\{-1,+1\}^d$. The input contains markers $b_i\in\{0,1\}$ under the promise $\sum_i b_i\in\{0,1\}$. The task is to output NULL if no marker is present and otherwise output the marked branch code.

An unnormalized parallel aggregator computes

$$
z=\sum_{i=1}^K b_i c_i.
$$

Then $z=0$ if no branch is marked, and $z=c_j$ if branch $j$ is marked. The decoder returns NULL when $\|y\|_\infty<1/2$ and otherwise returns the coordinate-wise signs of $y$. It recovers both cases under every $\ell_\infty$ perturbation of radius less than $1/2$. Direct aggregation uses $\Theta(d2^d)$ scalar additions or equivalent fan-in; the construction compresses persistent branch identity, not total work.

**Proof sketch.** Under the at-most-one-witness promise, the sum is either zero or one hypercube codeword. Noise below $1/2$ keeps a NULL vector inside the open $\ell_\infty$ ball of radius $1/2$, while every coordinate of a marked code retains magnitude above $1/2$ and its original sign.

**Exponential sampling corollary.** If the marked branch is uniform over $K=2^d$ branches, any independent sampled-trace simulator satisfying Theorem 5 and targeting success at least $2/3$ needs

$$
r\ge
\left\lceil\frac{\log 3}{-\log(1-2^{-d})}\right\rceil
=\Omega(2^d)
$$

samples. If the aggregator averages by $1/K$ instead of summing, the nonzero signal has magnitude $1/K$. Under the fixed-point resolution model, preventing this signal from rounding to zero requires $q\ge\log_2K-O(1)=\Omega(d)$ fractional bits; this conclusion does not follow from grid cardinality alone.

## 4. Mechanism Checks and Learned Diagnostics

We separate three kinds of evidence. First, deterministic checks verify that the stated recurrences and fixed binary codes implement the corresponding constructions. Second, metadata diagnostics apply the resource formulas to dataset instances. Third, learned controls test whether analogous behavior emerges under optimization. Only the third category is evidence about learnability; the first two are implementation and coverage checks rather than independent confirmation of the theorems.

| Mechanism | Evidence type | Result |
|-----------|---------------|--------|
| Exact frontier capacity | analytic capacity calculation | $n=4096$: 144 proxy bits versus 4096 arbitrary-frontier bits |
| BFS latent update | deterministic unit check | 81/81 cases; full-frontier agreement 1.000 |
| PointerChase width-one state | fixed-code construction check | exact once the imposed $d$ and $T$ budgets cover the instance |
| Explicit state traces | deterministic metadata coverage | per-instance $L_{\rm req}=D\lceil\log_2 n\rceil$ |
| Rare-branch sampling | Monte Carlo formula check | maximum absolute error 0.01033 from $1-(1-1/K)^r$ |
| Public text-rendered tasks | structured-solver diagnostics plus learned controls | structural axes are present; learned effects are mixed |

### 4.1 Controlled Resource Checks

For BFS, directed graphs use $n\in\{32,64,128\}$, edge probabilities $\{0.02,0.05,0.10\}$, depths $\{4,8,16\}$, and seeds $\{17,42,137\}$. The Boolean recurrence in Theorem 3 matches classical limited-depth BFS in all 81 cases, including exact agreement on the full frontier vector. This is an implementation check of the construction.

The capacity diagnostic instantiates the finite-state inequality in Theorem 2. Exact arbitrary-frontier tracking must distinguish $2^n$ subsets. The illustrative choice $d=p=\lceil\log_2 n\rceil$ provides only $\lceil\log_2 n\rceil^2$ stored bits: 49 at $n=128$ versus the required 128, and 144 at $n=4096$ versus 4096. This calculation concerns total capacity $dp$; it does not imply a dimension-only lower bound.

The sampling check varies $K=2^d$ for $d\in\{4,6,8,10,12\}$ and $r/K\in\{1/16,1/8,1/4,1/2,1,3/2\}$. Across 30 configurations, the mean absolute error from $1-(1-1/K)^r$ is 0.00418 and the maximum error is 0.01033. At $r=K$, success approaches $1-e^{-1}$ as $K$ grows. The simulation checks the implementation of the Bernoulli experiment; the inverse-mass law itself is analytic.

### 4.2 PointerChase: Fixed Construction and Learned Diagnostic

PointerChase separates frontier-scale reasoning from width-one state tracking. Arbitrary BFS frontiers require capacity proportional to the frontier family, while pointer chasing stores only the current vertex identity. Theorem 4 therefore predicts success once the latent identity has $d\ge\lceil\log_2 n\rceil$ Boolean coordinates and the update budget satisfies $T\ge D$.

![Figure 1: Binary-code PointerChase sweeps.](figures/pointer_binary_sweeps.png)

**Figure 1. Fixed-code PointerChase instantiates the construction's capacity and update thresholds.** Left: with $d=8$, train-range examples are solved once $T$ reaches the train maximum $D=4$, while depth-OOD examples require $T=12$. Right: on the graph-size split with up to 64 nodes, $d=6$ reaches exact accuracy when the step budget is sufficient. The code and hard pointer lookup are imposed, so this is a construction check rather than evidence that a neural model discovers the representation.

The following table reports the measured outcomes. For the binary construction, the $T$-sweep isolates local-oracle update depth and the graph-size split isolates identity capacity. The learned structured sweep is a separate trainability diagnostic and does not control numerical precision $p$.

| PointerChase result | Value |
|---------------------|-------|
| Binary $d=8$ T-sweep, IID at $T=4$ | 1.000 |
| Binary $d=8$ T-sweep, depth-OOD at $T=12$ | 1.000 |
| Binary graph-size split, $d=6,T=12,n\le64$ | 1.000 |
| Learned structured, $d=8,T=4$, IID | 1.000$\pm$0.000 |
| Learned structured, $d=8,T=12$, depth-OOD | 1.000$\pm$0.000 |
| Per-instance binary trace requirement | $L_{\rm req}=D\lceil\log_2 n\rceil$ |

![Figure 2: Learned structured PointerChase sweeps.](figures/pointer_learned_sweeps.png)

**Figure 2. Structured learned PointerChase exhibits depth dependence but not graph-size extrapolation.** Points show means over seeds $\{17,42,137\}$, with variability indicated where nonzero. IID accuracy reaches 1.000 at $T=4$, and depth-OOD accuracy rises from 0.754 at $T=4$ to 1.000 at $T=12$. In contrast, graph-size-OOD accuracy remains approximately 0.50--0.53 as $T$ or $d$ increases because learned node embeddings do not extrapolate to unseen identifiers.

The fixed-code sweep realizes the $\lceil\log_2n\rceil$-bit construction. The learned sweep shows that the structured recurrence can be optimized on IID and depth-OOD splits with a fixed identifier set, while its graph-size failure isolates compositional identity encoding as an additional requirement. It does not test the $dp$ lower bound because floating-point precision is uncontrolled.

![Figure 3: Fixed-width binary trace-budget coverage.](figures/pointer_trace_budget.png)

**Figure 3. Fixed-width explicit-state trace coverage is deterministic metadata accounting.** The per-instance requirement is $L_{\rm req}=D\lceil\log_2n\rceil$. On the evaluated budget grid, all dev examples are covered by $L=16$, all depth-OOD examples by $L=48$, and all graph-size-OOD examples by $L=48$; the exact maximum graph-size requirement is 36, so 48 is simply the next tested budget. These curves are coverage, not trained discrete-CoT accuracy.

### 4.3 Public-Dataset Structural Diagnostics and Learned Controls

We apply deterministic parsers and task-specific limited-step solvers to GraphQA [@sanford2024graph], CLUTRR [@sinha2019clutrr], and bAbI [@weston2016babi]. These runs ask whether instances span the same structural resource axes; they do not show that a generic neural model discovers the algorithms from text. We separately train positional latent text encoders over three seeds.

![Figure 4: GraphQA structural resource axes.](figures/hf_graphqa_resource_axes.png)

**Figure 4. GraphQA structured diagnostics expose step, identity-capacity, and trace-budget axes.** On `zero_shot_test`, parser accuracy is 1.000. The limited-depth solver reaches 0.632 at $T=1$ and 0.994 at $T=4$. Since the split has up to 19 nodes, binary identity coverage rises from 0.784 at $d=4$ to 1.000 at $d=5$; deterministic trace coverage rises from 0.838 at $L=16$ to 0.972 at $L=64$. Only the first curve is task-solver accuracy.

| Diagnostic/control | Value | Interpretation |
|--------------------|-------|----------------|
| GraphQA limited BFS, $T=1\to4$ | 0.632 to 0.994 | task-specific solver accuracy |
| GraphQA identity coverage, $d=4\to5$ | 0.784 to 1.000 | deterministic metadata coverage |
| GraphQA trace coverage, $L=16\to64$ | 0.838 to 0.972 | deterministic metadata coverage |
| CLUTRR structured solver, $T=12$ | 1.000 | task-specific relation-chain recurrence |
| bAbI structured solver, $T=2$ | 1.000 | task-specific route verification |
| GraphQA lexical baseline | 0.888 | high answer-only baseline |
| GraphQA learned text encoder | dev-selected $T=4$: 0.931$\pm$0.003$ | above lexical baseline, but flat in $T$ and matched by zero-CoT |
| CLUTRR learned text encoder | dev-selected $T=8$: 0.178$\pm$0.032$ | weak length-OOD control |
| bAbI learned text encoder | dev-selected $T=2$: 0.475$\pm$0.003$ | near-chance negative control |

![Figure 5: Learned GraphQA latent-step control.](figures/hf_graphqa_learned_control.png)

**Figure 5. The learned GraphQA control is accurate but does not exhibit a latent-step threshold.** Test accuracy is essentially flat over $T\in\{0,1,2,4\}$ and is comparable to the zero-CoT control. Thus it establishes task learnability for the encoder, not use of additional latent updates.

The structured solvers expose depth, identity-code, and transcript-length requirements in the public instances. The learned controls do not reproduce the solver-level thresholds: GraphQA is learnable but insensitive to $T$, while CLUTRR and bAbI remain weak or near chance. These results support ecological coverage of the resource regimes, not a claim that generic learned latent-reasoning models exploit them.

### 4.4 Alignment and Non-Alignment with the Theory

The deterministic checks instantiate the constructions directly. The Boolean BFS recurrence agrees with iterative BFS, the fixed binary PointerChase model reaches exact accuracy when its imposed identity and update budgets cover the instance, and the sampling simulation follows the Bernoulli hit probability. These are implementation checks rather than independent empirical tests.

The learned results are narrower. Structured PointerChase with intermediate-state supervision exhibits the predicted update-depth dependence on a fixed identifier set, but learned embeddings fail to extrapolate to unseen node identifiers and do not test the $dp$ capacity bound because numerical precision is uncontrolled. On public text-rendered tasks, deterministic parsers and structured solvers expose analogous instance-level depth and representation requirements, whereas the generic learned text controls show no consistent benefit from additional latent steps. Thus the experiments support the accounting framework as a diagnostic language; they do not establish that current language models learn the theorem constructions or that latent CoT is empirically superior to discrete CoT.

## 5. Limitations and Future Work

The lower bounds are conditional on explicit comparison models. The serialization theorem permits an unrestricted scratchpad to encode each rounded state and evaluate the same update. The pointer update bound uses sequential oracle access, its transcript bound uses fixed-width per-step decodability, and the sampling result assumes that success requires hitting a decisive branch with no identifying side channel. None is an unconditional lower bound for discrete CoT.

The resource accounting charges persistent-state capacity and recurrent updates but not update-circuit size, parameter count, full-input rereading, attention fan-in, internal arithmetic precision, training compute, or energy. This matters for the hypercube construction, whose $d$-coordinate output aggregates $2^d$ markers. Its corollary is a separation between access models, not an end-to-end runtime advantage.

Most reported curves are construction or metadata checks. Exact BFS agreement, binary-code thresholds, trace coverage, and Bernoulli sampling agreement follow from the implemented algorithms or formulas and should not be interpreted as independent evidence of learnability.

The learned evidence is mixed. Structured PointerChase uses function-table inputs and intermediate-state supervision; it generalizes in depth but not to unseen node identifiers. The learned GraphQA control is largely insensitive to latent-step count and comparable to its zero-CoT control, while the learned CLUTRR and bAbI controls are weak or near chance. No experiment evaluates a frontier latent-CoT LLM trained from answer-only natural-language supervision.

Finally, the exact and average-error frontier bounds concern a one-pass membership interface in which the prefix is inaccessible after the summary boundary. Other approximate reachability criteria, task-specific sufficient statistics, nonuniform encodings, and architectures with auxiliary memory can change the required resources. Extending the fixed-width pointer transcript result to broader trace models would require a communication- or streaming-complexity formulation consistent with Theorem 1.

## 6. Conclusion

Finite-precision latent CoT is best analyzed through explicit resource accounting, not by analogy to nondeterminism. An unrestricted discrete scratchpad can serialize any rounded latent trace, so meaningful comparisons require a specified representation or access restriction. Under such restrictions, exact frontier tracking, width-one pointer chasing, and rare-branch sampling occupy distinct capacity and update regimes. The fixed constructions and structural diagnostics illustrate these regimes, while the mixed learned controls show that realizing them through optimization is a separate question. The immediate evaluation recommendation is therefore modest: report persistent-state capacity, numerical resolution, recurrent updates, and sampling budget together with generated-token length, and state which computation and access costs remain uncharged.

## References

::: {#refs}
:::

## Appendix A: Full Proofs

### A.1 Serialization Upper Bound

**Claim.** Fix input length $n$, latent dimension $d=d(n)$, precision $p=p(n)$, and latent step count $T=T(n)$. Let $Q_p$ be a quantization grid with $|Q_p|\le 2^p$. For any finite alphabet $\Sigma$ with $|\Sigma|\ge 2$, any latent trace

$$
z_0,z_1,\ldots,z_T \in Q_p^d
$$

can be serialized using at most

$$
(T+1)\left\lceil \frac{dp}{\log_2 |\Sigma|}\right\rceil
$$

state tokens, plus $O(T)$ delimiter tokens. A discrete simulator that can evaluate the same deterministic rounded update rule

$$
z_{t+1}=Q_p(U_n(x,z_t))
$$

exactly reproduces the continuous-CoT output.

**Assumptions.** $Q_p$ is finite with $|Q_p|\le 2^p$; the latent update is deterministic after rounding; the discrete simulator can encode and decode arbitrary bit strings over $\Sigma$; and the simulator has access to the same rounded map $z\mapsto Q_p(U_n(x,z))$.

**Proof.** Let $b=\log_2|\Sigma|$. Since each coordinate belongs to a set of size at most $2^p$, there exists an injective binary code $\mathrm{code}_1:Q_p\to\{0,1\}^p$. For $z=(z[1],\ldots,z[d])\in Q_p^d$, define

$$
\mathrm{code}(z)=\mathrm{code}_1(z[1])\Vert \cdots \Vert \mathrm{code}_1(z[d]).
$$

This is an injective string of length $dp$. A length-$m$ token string over $\Sigma$ distinguishes $|\Sigma|^m$ values. Choose

$$
m=\left\lceil \frac{dp}{\log_2|\Sigma|}\right\rceil.
$$

Then $|\Sigma|^m\ge 2^{dp}$, so all possible latent-vector codes can be injected into $\Sigma^m$. Delimiters add only $O(T)$ tokens over the trace.

The simulator writes the serialization of $z_0$. Suppose it has decoded the exact state $z_t$ after step $t$. It computes $z'_{t+1}=Q_p(U_n(x,z_t))$. The continuous model defines the same next state $z_{t+1}=Q_p(U_n(x,z_t))$, so $z'_{t+1}=z_{t+1}$. Induction over $t=0,\ldots,T$ shows that the simulator's decoded state equals the latent model's state at every step. Applying the same output map $V_n(x,z_T)$ gives the same final answer.

The case $T=0$ reduces to serializing $z_0$ once. The theorem excludes $|\Sigma|=1$, where nontrivial bit strings cannot be represented. If a weaker discrete model must expose arithmetic substeps as tokens, that computation overhead is model-dependent and is separate from state serialization. Thus every finite-precision latent trace is serializable using $O(Tdp/\log|\Sigma|)$ tokens up to delimiters. $\square$

### A.2 Hybrid Continuation-Index Tradeoff

**Claim.** Let $H:\mathcal A\times\mathcal C\to\mathcal Y$ be a deterministic prefix--continuation task. After a prefix $a\in\mathcal A$ is consumed, it becomes inaccessible and only a summary

$$
q(a)=(z(a),\tau(a))\in Q_p^d\times\Sigma^{\le L}
$$

remains. For $a,a'\in\mathcal A$, write $a\equiv_Ha'$ when $H(a,c)=H(a',c)$ for every shared continuation $c\in\mathcal C$, and let $N_H$ be the number of equivalence classes. Any machine that is exactly correct for every $(a,c)$ satisfies

$$
2^{dp}M_\Sigma(L)\ge N_H,
\qquad
dp+\log_2M_\Sigma(L)\ge\log_2N_H,
$$

where $M_\Sigma(L)=\sum_{\ell=0}^L|\Sigma|^\ell$. If every transcript has fixed length $L$, the second term becomes $L\log_2|\Sigma|$. If a family $\mathcal F$ of explicit frontier states is pairwise continuation-distinguishable, then

$$
dp+\log_2M_\Sigma(L)\ge\log_2|\mathcal F|.
$$

For the state-only case $L=0$, all frontiers $\mathcal F=2^{[n]}$ require $dp\ge n$; frontiers of size at most $k$ require

$$
dp\ge\log_2\sum_{i=0}^k{n\choose i}.
$$

**Assumptions.** The prefix is unavailable after the boundary; future computation sees the same continuation and depends on the prefix only through $(z(a),\tau(a))$; the machine is deterministic after rounding; $|Q_p|\le2^p$; and the retained transcript has length at most $L$.

**Proof.** Choose one representative from each $\equiv_H$ class. Suppose two representatives $a\not\equiv_Ha'$ satisfy $q(a)=q(a')$. By definition of inequivalence, some continuation $c$ has $H(a,c)\ne H(a',c)$. Both runs begin this same continuation with the same latent state and retained transcript. Determinism and the absence of prefix rereading imply identical future states and identical outputs, contradicting exact correctness. Hence the class-to-summary map is injective. The number of possible summaries is at most

$$
|Q_p|^d\sum_{\ell=0}^L|\Sigma|^\ell
\le 2^{dp}M_\Sigma(L),
$$

which gives the claim after taking base-two logarithms. If all retained transcripts have exactly $L$ tokens, their number is $|\Sigma|^L$, yielding $dp+L\log_2|\Sigma|\ge\log_2N_H$.

For an explicitly recoverable or pairwise distinguishable frontier family, each frontier belongs to a different class, so $N_H\ge|\mathcal F|$. In the state-only case, the arbitrary and sparse bounds follow from $|2^{[n]}|=2^n$ and $|\{S:|S|\le k\}|=\sum_{i=0}^k{n\choose i}$. When $1\le k\le n/2$, ${n\choose k}\ge(n/k)^k$ further gives $dp\ge k\log_2(n/k)$. The result is a retained-information lower bound for the stated interface, not a lower bound for an algorithm that may reread the entire prefix. $\square$

### A.3 Layered Streaming Frontier Corollary

**Claim.** Let the layers be $L_0=\{s\}$, $L_1=[n]$, and $L_2=\{\tau\}$. For each $S\subseteq[n]$, a fixed-length prefix $P_S$ encodes the $n$ indicators of edges $(s,v)$, with $(s,v)$ present iff $v\in S$. A deterministic one-pass machine consumes $P_S$, after which the prefix is inaccessible and only a state in $Q_p^d$ remains. The machine must answer $s$--$\tau$ reachability for every continuation $C_v$ containing only the edge $(v,\tau)$. Then $dp\ge n$.

**Proof.** Take distinct $S,S'\subseteq[n]$ and choose $v\in S\triangle S'$. Under the shared continuation $C_v$, exactly one of $P_SC_v$ and $P_{S'}C_v$ contains the path $s\to v\to\tau$. Hence all $2^n$ prefixes are pairwise continuation-inequivalent. Theorem 2 gives

$$
2^n\le|Q_p|^d\le2^{dp},
$$

so $dp\ge n$. The read-once boundary is essential: an unrestricted verifier that can reread $P_S$ is outside this corollary. $\square$

**Randomized approximate extension.** Let $S=(S_1,\ldots,S_n)$ be uniform on $\{0,1\}^n$, and let the continuation contain an independent uniform query $V\in[n]$. A randomized one-pass encoder retains $Y=(Z,\tau)\in Q_p^d\times\Sigma^{\le L}$, and a decoder predicts $S_V$ from $(Y,V)$. If its average error is at most $\varepsilon<1/2$, then

$$
dp+\log_2M_\Sigma(L)\ge n[1-h_2(\varepsilon)].
$$

To prove the claim, reveal all random coins $R$ to the decoder; this can only reduce its error. For each coordinate $i$, let $e_i$ be the Bayes error for predicting $S_i$ from $(Y,R)$. Then $e_i\le1/2$ and $n^{-1}\sum_i e_i\le\varepsilon$. Binary Fano's inequality, subadditivity of conditional entropy, and concavity of $h_2$ give

$$
\begin{aligned}
H(S\mid Y,R)
&\le \sum_{i=1}^n H(S_i\mid Y,R)\\
&\le \sum_{i=1}^n h_2(e_i)\\
&\le n h_2\!\left(\frac1n\sum_{i=1}^n e_i\right)\\
&\le n h_2(\varepsilon).
\end{aligned}
$$

Because $S$ is uniform and independent of $R$,

$$
I(S;Y\mid R)=n-H(S\mid Y,R)
\ge n[1-h_2(\varepsilon)].
$$

On the other hand, $Y$ takes at most $2^{dp}M_\Sigma(L)$ values for each $R$, so

$$
I(S;Y\mid R)\le H(Y\mid R)
\le dp+\log_2M_\Sigma(L).
$$

Combining the last two displays proves Corollary 2.2. The exact state-only frontier bound is recovered at $\varepsilon=0$ and $L=0$. $\square$

### A.4 Continuous BFS Upper Bound

**Claim.** For a directed graph with $A\in\{0,1\}^{n\times n}$ and source $s$, let $z_0=e_s$ and

$$
z_{t+1}[v]=\mathbf 1\left[z_t[v]+\sum_{u=1}^n z_t[u]A_{uv}\ge1\right].
$$

Then $z_D[v]=1$ iff $v$ is reachable from $s$ by a path of length at most $D$. One update is computed by a uniform depth-two unbounded-fan-in threshold circuit with $n^2$ AND gates and $n$ OR gates, hence size $O(n^2)$, and read-only access to $A$. The persistent state uses $d=n$ Boolean coordinates. For this recurrence, $D$ updates are worst-case necessary to reach a vertex at distance $D$.

**Proof.** Let $R_t$ be the vertices reachable from $s$ within $t$ edges. The base state $z_0=e_s$ is the indicator of $R_0$. If $z_t$ indicates $R_t$, then $\sum_u z_t[u]A_{uv}\ge1$ exactly when some $u\in R_t$ has an edge to $v$; the term $z_t[v]$ preserves vertices already reached. Thus $z_{t+1}$ indicates $R_{t+1}$, and induction proves the reachability claim.

For the circuit, first compute

$$
g_{uv}=\mathbf1[z_t[u]+A_{uv}\ge2]=z_t[u]\wedge A_{uv}
$$

for all $(u,v)$ using $n^2$ threshold gates. A second layer computes $z_{t+1}[v]=\mathbf1[z_t[v]+\sum_ug_{uv}\ge1]$ with one OR-threshold gate per $v$. This family is uniformly constructible from $n$. On the path $v_0=s\to v_1\to\cdots\to v_D$, induction also gives $z_T[v_D]=0$ for every $T<D$, proving tightness for the specified one-hop recurrence.

If integer counts are materialized, values in $\{0,\ldots,n\}$ need $\lceil\log_2(n+1)\rceil$ magnitude bits. If normalized aggregation replaces a nonzero message by $1/n$ and nearest-neighbor fixed-point rounding has quantum $2^{-q}$, avoiding forced rounding to zero requires $1/n\ge2^{-q-1}$, or $q\ge\log_2n-1$. This is a quantizer-specific resolution statement, not a consequence of $|Q_p|\le2^p$. $\square$

### A.5 Tight Local-Oracle Resources for Pointer Chasing

**Claim.** Let $f:[n]\to[n]$, source $s\in[n]$, and $1\le D\le n$. After initialization, an algorithm accesses $(f,s)$ only through a persistent state and at most one adaptive oracle query to $f$ per update; its endpoint is decoded only from the final state. Then exact computation of $f^D(s)$ is achievable with $T=D$ and $\lceil\log_2n\rceil$ Boolean state coordinates. Every deterministic exact algorithm needs $T\ge D$ oracle updates in the worst case and $dp\ge\log_2n$ persistent bits. If, in addition, a context-free injective fixed-width code $\operatorname{Enc}:[n]\to\Sigma^\ell$ is emitted for each of the $D$ visited states, then

$$
\ell\ge\left\lceil\log_{|\Sigma|}n\right\rceil,
\qquad
L=D\ell\ge D\left\lceil\log_{|\Sigma|}n\right\rceil.
$$

**Proof.** For the upper bound, choose an injective binary code $\operatorname{bin}:[n]\to\{0,1\}^{\lceil\log_2n\rceil}$. Store $z_0=\operatorname{bin}(s)$ and, at update $i$, decode $v_i$, query $f(v_i)$, and store $z_{i+1}=\operatorname{bin}(f(v_i))$. Induction gives $z_i=\operatorname{bin}(f^i(s))$.

For the query lower bound, consider a deterministic algorithm making $q<D$ queries. Answer queries online while maintaining a known pointer path $x_0=s,x_1,\ldots,x_j$. When the algorithm queries the current unassigned endpoint $x_j$, return a fresh vertex $x_{j+1}$ not queried previously; such a vertex exists before query $D$ because $D\le n$. Answer every other previously unassigned query by a self-loop. At termination, the current path endpoint remains unqueried. There are two completions consistent with the entire transcript: one fixes that endpoint, while another sends it to a distinct fixed vertex (or, when only the final pointer value remains, to any distinct vertex). These completions have different $D$-step endpoints. The algorithm has the same query transcript on both and therefore fails on one, so $q\ge D$ is necessary.

For state capacity, take the identity oracle $f(v)=v$. Across the $n$ possible sources the required endpoint is $s$. Because the final decoder receives no source side channel, the $n$ outputs require $n$ distinguishable final states. Hence $|Q_p|^d\ge n$ and $dp\ge\log_2n$.

Finally, injectivity of the fixed-width code gives $n\le|\Sigma|^\ell$; multiplying the common width by the required $D$ segments gives the trace bound. Variable-length or context-dependent transcripts are not covered, and an unrestricted update that can read the full table is outside the oracle lower bound. $\square$

### A.6 Exact Rare-Branch Sampling Law

**Claim.** Suppose each of $r$ independent inspections hits a decisive branch with probability $\mu\in(0,1)$, and a simulator may return a successful answer only together with a witness contained in an inspected branch. Then

$$
\Pr(\mathrm{success})\le1-(1-\mu)^r\le r\mu.
$$

Success probability at least $1-\delta$ requires

$$
r\ge\left\lceil\frac{\log(1/\delta)}{-\log(1-\mu)}\right\rceil.
$$

For $0<\mu\le1/2$, this is $\Omega(\log(1/\delta)/\mu)$; for $\mu=1/K$ and constant $\delta$, it is $\Omega(K)$.

**Proof.** Let $E_i$ be the event that inspection $i$ hits a decisive branch and $E=\bigcup_iE_i$. Witness certification implies $\{\mathrm{success}\}\subseteq E$. Independence gives

$$
\Pr(E)=1-\Pr\!\left(\bigcap_{i=1}^rE_i^c\right)=1-(1-\mu)^r.
$$

The union bound gives $\Pr(E)\le r\mu$. If success is at least $1-\delta$, then necessarily $(1-\mu)^r\le\delta$, and taking logarithms yields the displayed exact threshold. Since $-\log(1-\mu)\le\mu/(1-\mu)\le2\mu$ for $\mu\le1/2$, the asymptotic bound follows. If inspections are adaptive but every conditional hit probability is at most $\mu$, independence need not hold; the union-bound conclusion $r\ge(1-\delta)/\mu$ remains valid. $\square$

### A.7 Hypercube Rare-Witness State Code

**Claim.** Let $K=2^d$ and $C_d=\{-1,+1\}^d=\{c_1,\ldots,c_K\}$. Given markers $b_i\in\{0,1\}$ with $\sum_i b_i\in\{0,1\}$, the full-access aggregate

$$
z=\sum_{i=1}^Kb_ic_i
$$

is zero in the NULL case and equals the marked codeword otherwise. Define

$$
\operatorname{Dec}(y)=
\begin{cases}
\mathrm{NULL},&\|y\|_\infty<1/2,\\
(\operatorname{sign}y_1,\ldots,\operatorname{sign}y_d),&\text{otherwise}.
\end{cases}
$$

The decoder is correct in both cases under every perturbation $\eta$ with $\|\eta\|_\infty<1/2$. Direct computation uses $K$ marker accesses and $\Theta(Kd)=\Theta(d2^d)$ elementary scalar work, or the corresponding unbounded-fan-in wiring.

**Proof.** The hypercube contains $2^d=K$ codewords. If every marker is zero, $z=0$, so $\|z+\eta\|_\infty<1/2$ and the decoder returns NULL. If $b_j=1$, the promise gives $z=c_j$. Every coordinate of $c_j+\eta$ retains magnitude greater than $1/2$ and the sign of $c_j$, so the decoder returns $c_j$. Computing all $d$ sums directly reads every marker-code pair, giving the stated work and access count.

If the aggregate is normalized to $z/K$, a marked coordinate has magnitude $1/K$. Under nearest-neighbor rounding with fixed-point quantum $2^{-q}$, it rounds to zero whenever $1/K<2^{-q-1}$. Avoiding this forced loss requires $q\ge\log_2K-1=d-1$. This statement is specific to fixed-point scale and does not follow from state cardinality alone. Multiple marked branches are outside the promise because sums of codewords need not identify their summands uniquely. $\square$

### A.8 Hypercube Sampled-Inspection Corollary

**Claim.** In the hypercube family, suppose one marked branch is uniform over the $K=2^d$ branches and a sampled simulator must inspect the marked branch to return its witness. Independent uniform inspections achieve success at least $2/3$ only if

$$
r\ge\left\lceil\frac{\log3}{-\log(1-2^{-d})}\right\rceil
=(\log3+o(1))2^d.
$$

**Proof.** One inspection hits the marked branch with mass $\mu=2^{-d}$. Apply Appendix A.6 with $\delta=1/3$. The asymptotic equality follows from $-\log(1-x)=x+O(x^2)$ as $x\downarrow0$. Thus $r=\Omega(2^d)=\Omega(K)$. This is a query/access separation: the full aggregate in Appendix A.7 reads all $K$ branches, while the comparator is restricted to sampled inspections. It is not an end-to-end work separation from arbitrary discrete scratchpads. $\square$

## Appendix B: Full Experiment Details

### B.1 Synthetic Mechanism Checks

The synthetic mechanism checks were run by `experiments/run_synthetic_experiments.py` using only the Python standard library and seeds $\{17,42,137\}$. They test exact BFS frontier updates, pointer-chasing latent IDs, frontier-capacity formulas, and rare-branch sampling probabilities.

For BFS, directed graphs used $n\in\{32,64,128\}$, edge probabilities $\{0.02,0.05,0.10\}$, depths $\{4,8,16\}$, and seeds $\{17,42,137\}$. The baseline was classical iterative BFS; the model was the Boolean latent threshold update from Theorem 3. Across 81 cases, decision accuracy was 1.000 and full-frontier agreement was 1.000.

For pointer chasing, random functions $f:[n]\to[n]$ used $n\in\{64,256,1024,4096\}$, steps $D\in\{8,16,32\}$, and seeds $\{17,42,137\}$. The baseline was direct function iteration and the model was binary latent ID iteration. Across 36 cases, final vertex accuracy was 1.000. For a binary vocabulary, the explicit-state trace lower bound is $D\lceil\log_2 n\rceil$ tokens; the largest tested setting, $n=4096$ and $D=32$, has 12 latent bits per step and a 384-token binary explicit-state lower bound.

For rare-branch sampling, $K=2^d$ with $d\in\{4,6,8,10,12\}$ and $r/K\in\{1/16,1/8,1/4,1/2,1,3/2\}$. Each configuration used 2000 trials per seed. Across 30 configurations, mean absolute error against $1-(1-1/K)^r$ was 0.00418, maximum absolute error was 0.01033, and the tolerance was 0.04000.

For frontier capacity, the capacity calculator compared arbitrary frontier bits $n$ with the log-dimensional proxy $dp=\lceil\log_2 n\rceil^2$:

| $n$ | $\lceil\log_2 n\rceil^2$ | arbitrary-frontier bits | gap |
|-----|---------------------------|-------------------------|-----|
| 32 | 25 | 32 | 7 |
| 64 | 36 | 64 | 28 |
| 128 | 49 | 128 | 79 |
| 256 | 64 | 256 | 192 |
| 512 | 81 | 512 | 431 |
| 1024 | 100 | 1024 | 924 |
| 4096 | 144 | 4096 | 3952 |

### B.2 LatentReasoning-NL Data and GPU Setup

The EMNLP-oriented dataset cache is `LatentReasoning-NL v0.1`, generated by `experiments/build_latent_reasoning_nl.py`. The mini cache contains 2200 examples across PointerChase-NL, Reachability-NL, ConstraintPlan-NL, and RareWitness-NL, with cached tensors under `data/latent_reasoning_nl/v0.1/cache/`.

The cached GPU sweep used `experiments/run_cached_latent_gpu_sweep.py`. It trained a mean-pooling probe baseline and tiny latent cross-attention models for seeds $\{17,42,137\}$, latent dimensions $d\in\{16,32,64,128\}$, latent steps $T\in\{1,2,4,8\}$, 10 epochs, and batch size 32. All 51 planned CUDA runs completed. The best mean dev accuracy was 0.5317 with high cross-seed variation, so the run is treated as a training-harness and negative diagnostic rather than headline evidence.

A redesigned PointerChase-only pilot cache, `LatentReasoning-NL v0.2_pointer_pilot`, used the `emnlp_small` split sizes, the PointerChase-NL subset, max length 1024, and no truncation. The pilot added a positional Transformer text encoder before the latent loop because ordered edge pairs cannot be parsed by a bag-of-words encoder.

The full PointerChase cache, `LatentReasoning-NL v0.3_pointer_full`, used the `emnlp_full` preset and only the PointerChase-NL subset. It contains 60k train examples, 6k dev examples, 8k IID test examples, and 4k examples for each OOD split. Two CUDA sweeps were run on this cache: a deterministic binary-code pointer model that fixes node-id extrapolation, and a three-seed learned embedding lookup model that measures trainable structured-model stability.

### B.3 PointerChase Pilot and Full Sweeps

The corrected structured PointerChase pilot uses structured pointer cache fields (`function`, source id, target id, depth, and path ids), initializes the latent state from the source node, updates it through differentiable function-table lookup, halts after the requested depth, and adds intermediate path-state supervision. Its one-seed CUDA results were:

| Configuration | Dev acc. | IID test acc. | Depth OOD acc. |
|---------------|---------:|--------------:|---------------:|
| $d=8,T=4$ | 0.985 | 0.981 | 0.746 |
| $d=8,T=8$ | 0.981 | 0.983 | 0.850 |
| $d=8,T=12$ | 0.994 | 0.993 | 0.990 |
| $d=16,T=4$ | 1.000 | 1.000 | 0.755 |

The full binary-code PointerChase sweep used deterministic $d$-bit node codes and hard pointer lookup, directly testing $d\ge\lceil\log_2 n\rceil$ and $T\ge D$. It ran 49 CUDA configurations over $d\in\{2,3,4,5,6,8,16\}$ and $T\in\{0,1,2,4,6,8,12\}$ on the full v0.3 cache. All 49 runs completed.

| Configuration | Dev acc. | IID acc. | Depth OOD acc. | Graph-size OOD acc. |
|---------------|---------:|---------:|---------------:|--------------------:|
| $d=8,T=0$ | 0.562 | 0.549 | 0.551 | 0.515 |
| $d=8,T=1$ | 0.592 | 0.585 | 0.606 | 0.528 |
| $d=8,T=2$ | 0.759 | 0.757 | 0.695 | 0.644 |
| $d=8,T=4$ | 1.000 | 1.000 | 0.754 | 0.835 |
| $d=8,T=6$ | 1.000 | 1.000 | 0.795 | 1.000 |
| $d=8,T=12$ | 1.000 | 1.000 | 1.000 | 1.000 |
| $d=4,T=12$ | 1.000 | 1.000 | 1.000 | 0.498 |
| $d=6,T=12$ | 1.000 | 1.000 | 1.000 | 1.000 |

The train/dev/IID splits have $n\le16$, so $d=4$ suffices there. The graph-size split has $n\le64$, and $d=6$ reaches 1.000 once the latent step budget is sufficient. The depth split has $D\le12$, so $T=4$ and $T=6$ remain below 1.000 while $T=12$ solves it.

The full learned structured sweep used the same structured function-table fields as the fixed pilot, but learned node embeddings and differentiable lookup. It ran 39 CUDA configurations: seeds $\{17,42,137\}$ over a $d=8$ T-sweep and a $T=4$ d-sweep. The IID/depth subset was:

| Configuration | Dev acc. mean | Dev acc. std | IID acc. mean | Depth OOD acc. |
|---------------|--------------:|-------------:|--------------:|---------------:|
| $d=8,T=4$ | 1.000 | 0.000 | 1.000 | 0.754 |
| $d=8,T=8$ | 1.000 | 0.000 | 1.000 | 0.860 |
| $d=8,T=12$ | 1.000 | 0.000 | 1.000 | 1.000 |
| $d=6,T=4$ | 1.000 | 0.000 | 1.000 | 0.754 |
| $d=16,T=4$ | 1.000 | 0.000 | 1.000 | 0.754 |

The learned model is stable on IID and depth-controlled splits, but graph-size-OOD accuracy remains approximately 0.50--0.53 across both sweeps. Figure 2 reports this negative result together with the positive depth behavior. Learned node embeddings therefore do not provide compositional extrapolation to unseen node-id ranges.

### B.4 Explicit Trace Token Budget Diagnostic

The explicit-state trace diagnostic computes the binary token budget needed to write every visited vertex in a pointer-chasing trace:

$$
L_{\mathrm{required}}=D\lceil\log_2 n\rceil.
$$

The budget grid used $L\in\{0,4,8,12,16,24,32,48,64\}$ on dev, depth-OOD, and graph-size-OOD splits. Verification found that the dev split reaches full explicit-trace coverage at $L=16$, the depth-OOD split reaches full coverage at $L=48$, and the graph-size-OOD split reaches full coverage at $L=48$ with maximum required budget 36. This diagnostic is deterministic metadata accounting, not a trained discrete-CoT accuracy curve.

### B.5 Public-Dataset Structural Diagnostics

The structural diagnostics used three public Hugging Face datasets: GraphQA reachability/shortest path, CLUTRR relation-chain reasoning, and bAbI NLI path-finding. The deterministic run uses parsers plus structured limited-step solvers and answer-only lexical baselines. Its resource grids were $T\in\{0,1,2,4,8,12,16\}$, $d\in\{2,3,4,5,6,8,16,32,64,128\}$, $L\in\{0,4,8,16,32,64,128,256,512\}$, and seed 17 for lexical baseline sampling.

| Dataset | Parser acc. | Answer-only lexical acc. | Resource result |
|---------|------------:|--------------------------:|-----------------|
| GraphQA reachability | 1.000 | 0.888 | $T=1$: 0.632, $T=4$: 0.994 |
| CLUTRR | metadata | 0.113 | test $T=4$: 0.265, $T=8$: 0.782, $T=12$: 1.000 |
| bAbI path-finding | 1.000 | 0.524 | test $T=2$: 1.000 |

GraphQA also exposes identity-capacity and explicit-trace axes: on `zero_shot_test`, identity-code coverage rises from 0.784 at $d=4$ to 1.000 at $d=5$, and explicit trace coverage rises from 0.838 at $L=16$ to 0.972 at $L=64$. These are instance-coverage diagnostics, not theorem proofs or evidence that a learned model uses the corresponding representations.

### B.6 Learned Public-Dataset Neural Controls

As a separate learnability diagnostic, the learned control trained a small positional text encoder with a continuous latent loop. The three-seed aggregate used seeds $\{17,42,137\}$, GraphQA $T\in\{0,1,2,4\}$, CLUTRR $T\in\{0,2,4,8\}$, and bAbI $T\in\{0,1,2,4\}$.

| Task | Dev-selected $T$ | Dev acc. mean $\pm$ sd | Test acc. mean $\pm$ sd | Interpretation |
|------|---------:|------------------------:|-------------------------:|----------------|
| GraphQA reachability | 4 | 0.943$\pm$0.001 | 0.931$\pm$0.003 | task is learned, but accuracy is flat in $T$ and matches zero-CoT |
| CLUTRR | 8 | 0.644$\pm$0.034 | 0.178$\pm$0.032 | weak length-OOD control |
| bAbI path-finding | 2 | 0.535$\pm$0.003 | 0.475$\pm$0.003 | near-chance negative control |

GraphQA is the positive learned external control, but its T curve is not a theorem-level latent-step threshold. CLUTRR and bAbI are reported as weak or negative controls for the generic text model.

### B.7 Verification and Data-Leakage Checks

Experiment verification is stored in `experiment_workspace/verification_results.json`. The passed experiments are `exp_bfs_frontier`, `exp_pointer_chasing`, `exp_sampling_lower_bound`, `exp_capacity_table`, `exp_pointer_full_binary`, `exp_pointer_full_learned`, `exp_pointer_trace_budget`, and `exp_hf_external_validation`. Partial experiments are `exp_latent_nl_gpu_sweep`, `exp_pointer_theory_pilot`, `exp_pointer_theory_pilot_fixed`, `exp_hf_learned_baselines`, and `exp_hf_learned_baselines_multiseed`.

The verification checks found no data leakage in the synthetic experiments, formula tables, cached PointerChase full sweep, or public-dataset diagnostics. The v0.3 PointerChase cache contains 60k train examples with zero truncation. The binary-code sweep reports all 49 CUDA records complete. The learned structured sweep reports all 39 CUDA records complete and cross-seed stability for IID/depth claims. The public-dataset parser/solver run uses public test examples without GraphQA training labels in the structured solver; lexical baselines train only on official train splits and evaluate on validation/test splits.
