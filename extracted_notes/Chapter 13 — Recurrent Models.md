# Chapter 13 — Recurrent Models

[TOC]

**High-level summary**

- Recurrent models aim to overcome the quadratic complexity of transformers by using fixed-size **state** (memory) updated per token in constant time.
- **Linearized attention** rewrites attention with kernel feature maps, enabling recurrent (causal) updates with constant memory.
- A **generic recurrent layer** is formalized as a state transition plus a readout; this includes standard RNNs, implicit layers, and structured state-space models (SSMs).
- **Vanilla RNNs** suffer from inefficient parallelization and severe gradient issues (vanishing/exploding) via backpropagation through time (BPTT).
- **Gated RNNs** (Li-GRU, GRU, LSTM) mitigate these issues by using learned gates to control state updates.
- **Structured state space models (SSMs)** and their diagonal/diagonalizable variants (e.g., LRU, S4/S5-like layers) offer efficient sequence modeling, often interpretable as convolutions or parallel scans.
- New architectures (ATF, RWKV, Mamba) combine ideas from attention, recurrence, and SSMs, giving recurrent-like models that can compete with transformers at scale.
- The chapter closes by situating these models in the broader landscape of modern deep learning and pointing to future directions.

---

## 1. Motivation: Why Recurrent Models?

**Problem with transformers**

- Standard self-attention has **quadratic complexity** in sequence length $n$:
  - Compute all pairwise interactions between tokens.
  - Even with key–value caches, memory grows linearly with $n$.
- In contrast, a **recurrent** mechanism wants:
  - A **fixed-size memory/state tensor** summarizing all past tokens.
  - **Constant** time and memory per new token, independent of sequence length.

**Goal**

- Design layers where, for each new token:
  - We update a **state** of fixed size.
  - The per-step cost is constant.
- Use this idea to reinterpret attention, build classical RNNs, structured SSMs, and recent recurrent-like architectures.

---

## 2. Linearized Attention Models

### 2.1 Generalized attention with a generic similarity

Consider self-attention with a generic scalar similarity function $\alpha(\cdot,\cdot)$:

- Inputs:
  - Queries $q_i$, keys $k_j$, values $v_j$.
- **General self-attention form**:
  $$
  h_i
  = \frac{\sum_{j=1}^n \alpha(q_i, k_j)\, v_j}
         {\sum_{j=1}^n \alpha(q_i, k_j)}.
  $$
- Standard self-attention uses:
  $$
  \alpha(x,y) = \exp(x^\top y).
  $$

For **autoregressive** use, this formulation is inconvenient:
- Complexity is $\mathcal{O}(n^2)$ in sequence length.
- A KV cache still grows linearly with $n$ in memory.

We want a way to **compress** information from all past tokens into a **fixed-size state**.

---

### 2.2 Kernel viewpoint and feature maps

In machine learning, a **non-negative similarity function** $\alpha$ that is a valid kernel can typically be written as:

> **Kernel representation**
> $$
> \alpha(x,y) = \phi(x)^\top \phi(y),
> $$
> where $\phi : \mathbb{R}^c \to \mathbb{R}^e$ is a (possibly high-dimensional) feature map.

Examples:

- **Polynomial kernel**:
  $$
  \alpha(x,y) = (1 + x^\top y)^d,
  $$
  which corresponds to $\phi(\cdot)$ computing all monomials up to degree $d$.
- **Gaussian kernel**: corresponds to an infinite-dimensional $\phi$, but can be approximated with random Fourier features.

Using this, we can rewrite attention:

$$
h_i
= \frac
  {\sum_{j=1}^n \phi(q_i)^\top \phi(k_j)\, v_j}
  {\sum_{j=1}^n \phi(q_i)^\top \phi(k_j)}.
$$

Rearrange by pulling $\phi(q_i)$ outside the sum:

$$
h_i
= \frac
  {\phi(q_i)^\top \sum_{j=1}^n \phi(k_j) v_j^\top}
  {\phi(q_i)^\top \sum_{j=1}^n \phi(k_j)}.
$$

This is the **linearized attention** formulation.

- Complexity to compute for all tokens: roughly $\mathcal{O}(n(e^2 + e v))$.
- Linear in $n$, advantageous when $n < e^2$.
- $\phi$ can be chosen freely (e.g. quadratic features or simple $\phi(x) = \operatorname{ELU}(x) + 1$ for short sequences).

---

### 2.3 Causal / recurrent formulation

For **causal** (autoregressive) use, we restrict sums to past indices $j \le i$.

Define:

- **Attention memory**:
  $$
  S_i = \sum_{j=1}^i \phi(k_j) v_j^\top
  $$
- **Normalizer memory**:
  $$
  z_i = \sum_{j=1}^i \phi(k_j).
  $$

Then the output is:
$$
h_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}.
$$

These memories admit a **recurrent update**:

- Initialization:
  $$
  S_0 = 0,\quad z_0 = 0.
  $$
- Per-token update:
  $$
  S_i = S_{i-1} + \phi(k_i) v_i^\top,
  $$
  $$
  z_i = z_{i-1} + \phi(k_i).
  $$

- Output:
  $$
  h_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}.
  $$

**Key properties**

- At each step $i$, we only need $(S_{i-1}, z_{i-1})$ and the current $(q_i, k_i, v_i)$.
- **Per-token cost** (time and memory) is constant (independent of $i$ and total sequence length).
- Past states $S_{i-1}, z_{i-1}$ can be discarded after being updated.

**Practical pattern**

- Use a **vectorized** formulation for training (efficient on GPUs).
- Use the **recurrent** formulation for autoregressive inference.

This is our first explicit example of a **recurrent layer** emerging from an attention-like construction.

---

## 3. General Recurrent Layers

### 3.1 Abstract definition

> **Definition (Recurrent layer)**  
> Given a sequence of tokens $x_1, x_2, \dots$, a recurrent layer maintains a **state** $s_i \in \mathbb{R}^e$ and produces an **output** $h_i \in \mathbb{R}^o$ via:
> $$
> s_i = f(s_{i-1}, x_i),
> $$
> $$
> h_i = g(s_i, x_i),
> $$
> where $s_0 = 0$ by convention, and $e$, $o$ are hyperparameters.

- $f$: **state transition function**.
- $g$: **readout function**.

**Bidirectional layers**

- For non-causal tasks, we can use **two** recurrent layers:
  - One processes left-to-right.
  - One processes right-to-left.
- Their outputs are concatenated for each time step (bidirectional RNNs).

**Stacking**

- Recurrent neural networks (RNNs) can be formed by **stacking** recurrent layers over the sequence $h_1, \dots, h_n$.

**Expressivity**

- No requirement on sequence length; in principle, it can be unbounded.
- With unbounded precision or growing architectures, RNNs can be **Turing-complete**.

---

### 3.2 Implicit layers (fixed-point viewpoint)

Consider applying the same recurrent update repeatedly to a **single token** $x$:

$$
s_i = f(s_{i-1}, x),
$$

starting from some initialization $s_0$.

- This is like a **deep network** with many layers that all share the same parameters $f$.
- If we run this update **infinitely many times** and the dynamics are stable, we converge to a **fixed point** $s$ satisfying:
  $$
  s = f(s, x).
  $$

> **Definition (Implicit layer)**  
> A layer defined as the solution of the fixed-point equation
> $$
> s = f(s, x)
> $$
> is called an **implicit layer**.

Training:

- Use numerical solvers to find the fixed point $s$ efficiently.
- Use the **implicit function theorem** to compute gradients in the backward pass.
- Similar constructions exist for **graph diffusion** models, where the diffusion is run to a stable state.

---

## 4. “Vanilla” Recurrent Layers

### 4.1 Linear transition + nonlinearity

Historically, the simplest RNNs instantiate $f$ and $g$ with fully-connected layers:

- **Transition**:
  $$
  f(s_{i-1}, x_i) = \phi(A s_{i-1} + B x_i),
  $$
- **Readout**:
  $$
  g(s_i, x_i) = C s_i + D x_i,
  $$

where:

- $A \in \mathbb{R}^{e \times e}$, $B \in \mathbb{R}^{e \times c}$,
- $C \in \mathbb{R}^{o \times e}$, $D \in \mathbb{R}^{o \times c}$,
- $c$ = input dimension (size of each token),
- $\phi$ is a nonlinearity (e.g. $\tanh$ or ReLU),
- Biases are omitted for simplicity.

Names:

- **Vanilla RNN**, **Elman RNN**, or simply **recurrent layer**.
- If $A,B$ are **fixed/untrained** and only the output layer is trained, we get:
  - **Echo state networks (ESNs)** / **reservoir computing**:
    - Random recurrent “reservoir”, trained readout.
    - Effective for some time series forecasting tasks when the reservoir is initialized carefully.

---

### 4.2 Computational inefficiency and sequential nature

A vanilla RNN is inherently **sequential**:

```python
# x: (batch_size, sequence_length, features)
x = torch.randn(batch_size, sequence_length, features)

# s: (batch_size, state_size)
s = torch.zeros(batch_size, state_size)

state_update = nn.RNNCell(features, state_size)

for i in range(x.shape[1]):
    s = state_update(x[:, i, :], s)
```

- The for-loop cannot be **parallelized over time** because each step depends on $s_{i-1}$.
- Even with optimized CUDA kernels, RNNs are typically **slower** than alternatives (e.g. attention or convolution) that can be parallelized across time.

------

### 4.3 Backpropagation through time and gradient issues

Unroll the recurrent computation:

$$
\begin{aligned}
s_1 &= f(s_0, x_1),\
s_2 &= f(s_1, x_2),\
&\vdots\
s_n &= f(s_{n-1}, x_n).
\end{aligned}
$$

This is equivalent to a deep network with **$n$ layers sharing parameters**.

Focus on the gradient with respect to $A$.

Define the **cumulative product of input Jacobians** from step $n$ back to $i$:

$$
\tilde{s}_i =
\prod_{j=i+1}^{n}
\frac{\partial f(s_{j-1}, x_j)}{\partial s_{j-1}}.
$$
Then the gradient of $s_n$ w.r.t. $A$ has the form:

$$
\frac{\partial s_n}{\partial A}
 = \frac{\partial f(s_{n-1}, x_n)}{\partial A}
 + \sum_{i=1}^{n-1}
   \tilde{s}_i \,
   \frac{\partial f(s_{i-1}, x_i)}{\partial A}.
$$

Interpretation:

- Each time step contributes to the gradient.
- Contributions are **weighted** by a long product of Jacobians.
- Each Jacobian includes derivatives of the activation function $\phi$ and the recurrent matrix $A$.

Consequences:

- These chained products can cause **vanishing** or **exploding** gradients.
- Stability for long sequences is often possible only if the eigenvalues of $A$ are carefully controlled (e.g. spectrally constrained).
- Early analyses of gradient problems in RNNs came from this setting.

**Stabilization techniques**

- **Layer normalization** over states (originally proposed for RNNs).
- **Truncated BPTT**:
    - Only backpropagate gradients for a limited time window.
- **Gradient clipping**:
    - Threshold gradient norms to avoid explosions.

Despite these, training vanilla RNNs remains challenging for long sequences.

------

## 5. Gated Recurrent Networks

### 5.1 Motivation: sparsifying transitions

Issue:

- In vanilla RNNs, the **entire state** is overwritten at each time step.
- In many sequences, only some tokens contain important information (e.g. silent regions in audio).
- Ideally, we want many dimensions of the state to remain **unchanged** most of the time.

Idea:

- Introduce **gates** that produce values in $[0,1]$ to **mask** or **mix** new vs old state values.

Consider a gate over the state:

$$
\gamma(s_{i-1}, x_i) = \sigma(V s_{i-1} + U x_i),
$$

with $V, U$ similar in shape to $A, B$ and $\sigma$ a sigmoid.

Interpretation:

- If a component $\gamma \approx 0$, we keep the previous state component.
- If $\gamma \approx 1$, we update it fully.

**Li-GRU-style transition (single gate)**:

$$
f(s_{i-1}, x_i) =
\gamma(s_{i-1}, x_i) \odot \phi(A s_{i-1} + B x_i)
+ (1 - \gamma(s_{i-1}, x_i)) \odot s_{i-1}.
$$
- This is a **soft gate** between:
    - The new candidate state $\phi(A s_{i-1} + B x_i)$.
    - The old state $s_{i-1}$.
- Can be seen as:
    - A differentiable approximation of a hard binary gate.
    - A **convex combination** between a residual connection and a full update.
- We can regularize the gate outputs to be closer to ${0,1}$ if desired.

### 5.2 GRU and LSTM

More expressive variants:

- **GRU** (Gated Recurrent Unit):
    - Adds a **reset gate** in addition to the update gate.
- **LSTM** (Long Short-Term Memory):
    - Uses an **input gate**, **forget gate**, and **output gate**.
    - Maintains a separate **cell state** with more stable dynamics.

Gated RNNs:

- Have been very successful for sequence modeling, especially **LSTMs**.
- Research on LSTMs and their variants remains active.

------

## 6. Structured State Space Models (SSMs)

### 6.1 Linear recurrent layers as SSMs

Consider removing the nonlinearity in the transition:

> **Linear SSM (state space model)**
> $$
> s_i = A s_{i-1} + B x_i,
> $$
> $$
> y_i = C s_i + D x_i.
> $$

- Here $A,B,C,D$ are learned matrices.
- Such models are called **state space models (SSMs)** in control theory.
- In deep learning, “SSM” often refers specifically to these **linear** variants (sometimes called structured SSMs).

Though linear in the recurrence, expressivity can be recovered by:

- Adding nonlinearities after $y_i$,
- Or interleaving SSM layers with token-wise MLPs.

Recent work:

- Theoretical constructions (HiPPO matrices) show how to choose $A$ to compress 1D sequences effectively.
- Leads to S4, S5, and related layers; here we focus on a simplified **Linear Recurrent Unit (LRU)** style model.

------

### 6.2 Closed-form solution and convolution view

The recurrence
$$
s_i = A s_{i-1} + B x_i
$$
has a **closed-form** solution (with $s_0 = 0$):
$$
s_i = \sum_{j=1}^i A^{i-j} B x_j.
$$
Interpretation:

- The state at time $i$ is a linear combination of all past inputs, weighted by powers of $A$.

Define a **kernel tensor**:
$$
K = \operatorname{stack}(A^{n-1} B,\ A^{n-2} B,\ \dots,\ A B,\ B).
$$
Then, if $X \in \mathbb{R}^{n \times c}$ stacks inputs ${x_j}$ over time, we can compute all states via a **1D convolution**:
$$
S = \operatorname{Conv1D}(X, K),
$$
where $S$ stacks ${s_i}$.

Consequences:

- The SSM layer can be implemented as a **convolutional** operation.
- For single-channel SSMs, one can further speed up via **frequency-domain** methods (FFT-based “long convolutions”).

------

### 6.3 Associative scans (parallel prefix sums)

We want to exploit **parallelism over time** using the structure of linear SSMs.

General pattern:

- Given a sequence $(x_1, \dots, x_n)$ and an associative binary operator $\star$, compute **all prefixes**:
$$
    x_1,\ x_1 \star x_2,\ x_1 \star x_2 \star x_3,\ \dots,\ x_1 \star \dots \star x_n.
$$

Naively:

- Use a for-loop, cost $\mathcal{O}(n T)$ where $T$ is cost of $\star$.

Parallel scan:

- Exploit associativity to restructure computation into **logarithmic depth**.
- Use a tree of pairwise combinations computed in parallel.
- Complexity: $\mathcal{O}(T \log n)$ in parallel time.

**Applying to SSMs**

For an SSM, define each element as a pair:

$$
x_i = (A, B x_i),
$$

and define the operator:

$$
(Z, z) \star (V, v) = (VZ,\ V z + v).
$$

Then the prefix

$$
x_1 \star \dots \star x_i = (A^i,\ s_i),
$$

so a **parallel scan** over this operator yields:

- $A^i$ (powers of $A$),
- $s_i$ (the states) for all $i$.

However, naive computation of $A^i$ for a dense $A$ via matrix multiplication can be expensive (cubic in state dimension), motivating structured choices for $A$.

------

### 6.4 Diagonal and diagonalizable SSMs (LRU-style)

To make SSMs efficient and numerically stable, we can constrain $A$.

> **Diagonalizable matrix**
> A square matrix $A$ is diagonalizable if there exist an invertible matrix $P$ and a diagonal matrix $\Lambda$ such that:
> $$
> A = P \Lambda P^{-1}.
> $$

Properties:

- Powers are easy to compute:
    $$
    A^i = P \Lambda^i P^{-1}.
    $$
- If such a decomposition exists, we can reparameterize the SSM in the **eigenbasis**.

Starting from:

$$
s_i = \sum_{j=1}^i A^{i-j} B x_j,
$$

multiply both sides by $P^{-1}$ and define:

- New state: $\bar{s}_i = P^{-1} s_i$,
- New input matrix: $\bar{B} = P^{-1} B$.

Then:

$$
\bar{s}*i = \sum*{j=1}^i \Lambda^{i-j} \bar{B} x_j.
$$

Similarly, with $y_i = C s_i + D x_i$ and $\bar{C} = C P$:

$$
y_i = \bar{C} \bar{s}_i + D x_i.
$$

Thus, in the transformed basis we have an SSM with **diagonal transition**:

> **Diagonal SSM reparameterization**
> $$
> \bar{s}*i = \Lambda \bar{s}*{i-1} + \bar{B} x_i,
> $$
> $$
> y_i = \bar{C} \bar{s}_i + D x_i.
> $$

With:

- $\Lambda = \operatorname{diag}(\lambda)$, where $\lambda \in \mathbb{C}^e$,
- $\bar{B} \in \mathbb{R}^{e \times c}$, $\bar{C} \in \mathbb{R}^{o \times e}$, $D \in \mathbb{R}^{o \times c}$.

Not every $A$ is diagonalizable in $\mathbb{R}$, but:

- Approximate diagonalizations can be obtained in $\mathbb{C}$.
- Training often parameterizes $\lambda$ directly (possibly complex).
- Stability requires $|\lambda_k| < 1$ (for all $k$) to avoid divergent dynamics.

The **Linear Recurrent Unit (LRU)** and related SSM layers exploit these ideas:

- Diagonal (or low-rank plus diagonal) transitions.
- Efficient computation of powers $\Lambda^i$.
- Careful parameterization of eigenvalues for stability.

------

## 7. Recent Recurrent-like Architectures

### 7.1 Attention-Free Transformers (ATF)

Motivation:

- Even linearized attention has complexity quadratic in the **feature dimension** $e$.
- Aim: a layer **linear** in both sequence length $n$ and feature dimension $e$.

ATF replaces dot products with **element-wise** interactions.

> **ATF core formulation**
> $$
> h_i
> = \sigma(q_i) \odot
> \frac{\sum_j \exp(k_j) \odot v_j}
> {\sum_j \exp(k_j)},
> $$
> where exponentials and multiplications are applied element-wise.

Per-dimension form for channel $z$:

$$
h_{i z} =
\sigma(q_{i z})
\cdot
\frac{\sum_j \exp(k_{j z}) v_{j z}}
{\sum_j \exp(k_{j z})}.
$$
Interpretation:

- For each channel $z$:
    - We compute a **scalar attention** over time (no cross-channel mixing in attention).
    - Then modulate by a **sigmoid-transformed query** $\sigma(q_{iz})$.
- ATF can be seen as **channel-wise attention**.

**Relative embeddings**

To add positional information, introduce relative embeddings $W_{ij}$:
$$
h_i = \sigma(q_i) \odot
\frac{\sum_j \exp(k_j + W_{ij}) \odot v_j}
{\sum_j \exp(k_j + W_{ij})}.
$$
- $W$ can be large ($m \times m$ if $m$ is max sequence length).
- Practically, $W$ can be factorized (low-rank) to reduce parameters.

Causal versions:

- Obtain a **recurrent/causal** ATF by restricting the sums to $j \le i$.

------

### 7.2 RWKV: A large-scale pre-trained RNN

The **RWKV** model extends ATF ideas and is one of the first RNN-like architectures to match transformers at large scale.

Modifications:

1. **Simplified relative embeddings**

    - Use a single vector $w \in \mathbb{R}^e$.
    - Define offsets for positions $i > j$:
$$
w_{ij} = -(i - j) w.
$$

  - Additionally, introduce a special offset $u$ for the current position $i$.



In a causal form, the attention-style update (schematically) becomes:
$$
h_i

W_o \Bigg(
\sigma(q_i) \odot
\frac{
\sum_{j=1}^{i-1} \exp(k_j + w_{ij}) \odot v_j
+
\exp(k_i + u) \odot v_i
}{
\sum_{j=1}^{i-1} \exp(k_j + w_{ij})
+
\exp(k_i + u)
}
\Bigg),
$$

  where:

  - $q_i$ is called the **receptance**,
  - $W_o$ is an output projection.



2. **Modified MLP block**

    The standard transformer MLP is replaced by a gated, squared-ReLU MLP:

    $$
    y = \sigma(W_1 x) \odot W_2 \big(\max(0, W_3 x)\big)^2.
    $$

    - Left gate: $\sigma(W_1 x)$.
    - Nonlinearity: squared ReLU.



3. **Token shift**

    Inputs to projections (in both the attention-like and MLP blocks) are replaced by **convex combinations** of $x_i$ and $x_{i-1}$, a technique called **token shift**, to improve performance.

RWKV is thus a hybrid model:

- Attention-like weighted aggregation over time,
- Gating and specially designed MLP,
- Recurrent/causal structure that scales to long contexts.

------

### 7.3 Selective State Space Models (Mamba and relatives)

We now connect **linearized attention**, **SSMs**, and **gated RNNs**.

Recall the causal linearized attention without the denominator:

$$
S_i = S_{i-1} + \phi(k_i) v_i^\top,
$$
$$
h_i = \phi(q_i)^\top S_i.
$$

This is structurally similar to an SSM:

- $S_i$ is a **matrix-valued state**.
- Transition is linear:
    $$
    S_i = S_{i-1} + \text{(input-dependent term)}.
    $$
- Readout is linear in $S_i$ with input-dependent coefficients $\phi(q_i)$.

However:

- **Standard SSMs** are *time-invariant*:
    - $A, B, C, D$ do not depend on $i$ or $x_i$.
- The above form uses coefficients that **depend on $x_i$**, making it a **time-varying** (selective) system.

This motivates **selective SSMs**:

> **Selective SSM (schematic form)**
> $$
> s_i = A(x_i) s_{i-1} + B(x_i) x_i,
> $$
> $$
> h_i = C(x_i) s_i + D x_i,
> $$
> where $A(\cdot)$, $B(\cdot)$, $C(\cdot)$ are (typically linear) projections of the input $x_i$.

The **Mamba** layer is a key example:

- SSM parameters become **input-dependent**:
    - $A(x_i)$, $B(x_i)$, $C(x_i)$ are linear maps of $x_i$.
- $D$ remains a simple residual connection:
    - $D x_i$ added to the output.
- The SSM is applied **per channel** with a **diagonal** transition:
    - Each channel has its own scalar state dynamics, parameterized by a vector.

Because matrices depend on inputs:

- Mamba loses the simple parallel scan formulation.
- Instead, it relies on a highly optimized, hardware-aware kernel for efficient sequence computation.

**Gated interpretation**

- Mamba can be seen as a **degenerate gated RNN**, where:
    - Gates arise from how input-dependent parameters modulate state updates.
    - There is also an additional depthwise convolution over tokens for flexibility.

**Mamba block structure** (schematically):

- Linear projection(s) of input.
- Depthwise convolution over time.
- Selective SSM operation with input-dependent parameters.
- Gated combination (similar in spirit to gated attention units).
- Residual connections and normalization around the block (not shown in the core equations).

Overall, Mamba and related selective SSMs are attempts to combine:

- The **parallelizability** and **long-range modeling** of SSMs,
- With the **adaptive, input-dependent weighting** reminiscent of attention,
- While retaining a recurrent interpretation.

------

## 8. Concluding Remarks

This chapter offers a broad tour of **recurrent models** and recurrent-like architectures in modern deep learning:

- Starting from **linearized attention**, we saw how attention can be reinterpreted in a recurrent form with constant-time per token.
- We formalized **recurrent layers** as dynamical systems, including their **implicit** fixed-point variants.
- We revisited **vanilla RNNs**, their historical significance, and their serious training difficulties (sequential computation, vanishing/exploding gradients).
- We saw how **gating** (Li-GRU, GRU, LSTM) allows selective updating of state, improving stability and performance.
- We then moved to **structured state space models**:
    - Linear recurrences that admit convolutional and parallel-scan implementations.
    - Diagonal/diagonalizable parameterizations that ensure efficiency and stability.
- Lastly, we examined **recent architectures** (ATF, RWKV, Mamba) that blend attention, gating, and SSM ideas to build large-scale models competitive with transformers.

The broader message:

- There is a rich design space beyond pure transformers.
- Balancing **convolution**, **recurrence**, and **attention** is an active research frontier.
- Practical deployment also depends on engineering considerations (distributed training, serving, system design), and future work continues to explore how these theoretical ideas translate to scalable, robust systems.
