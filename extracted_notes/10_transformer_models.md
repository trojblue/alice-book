# Chapter 10 – Transformer Models

[TOC]

- Transformers address long-range and input-dependent dependencies that are hard to capture with standard convolutions or simple RNNs.
- The core building block is **multi-head self-attention (MHA)**, derived from **non-local / long convolutions**, using **queries, keys, and values**.
- Self-attention is **permutation-equivariant**, so transformers require **positional embeddings** (absolute or relative) to encode order.
- A full transformer model stacks **MHA + MLP blocks** with **residual connections** and **LayerNorm**, then adds a task-specific head.
- The architecture’s set-like nature allows adding special tokens such as **class tokens** and **register tokens**, and supports multi-view / multimodal setups.

---

## 1. From Convolutions to Non-local Models

### 1.1 Historical context

- Around 2016–2017, **transformers** were introduced for NLP to model **long-range dependencies** efficiently.
- They showed strong **scaling laws**, and were later extended to **images, time series, and graphs**, becoming state-of-the-art in many domains.
- Architecturally, transformers are largely **data-agnostic**; the data type is handled by the **tokenizer**, which maps input (text, images, etc.) to token vectors.
- Historically:
  - Before transformers, **CNNs** and **RNNs** (and their attention variants) were the main tools for text.
  - **Multi-head attention** (MHA) was first introduced as a component for **RNNs**, then became the core of the transformer.
  - Recently, improved **RNN variants** (e.g., linearized RNNs) have re-emerged as competitive models for language modeling.

---

### 1.2 Limitations of standard 1D convolutions

Consider a sentence tokenized and embedded as a matrix  
$X \sim (n, e)$, where $n$ is the number of tokens, $e$ the embedding dimension, and $x_i$ the $i$-th token embedding.

A 1D convolution with (odd) kernel size $2k+1$ centered at token $i$ can be written schematically as
$$
h_i = \sum_{j=1}^{2k+1} W_j \, x_{i + k + 1 - j},
$$
where each offset within the receptive field has its own weight matrix $W_j$.

Key properties:

- Each output token $h_i$ depends only on tokens inside a **fixed receptive field** around $i$.
- To model **long-range dependencies**, we must:
  - Increase the receptive field (larger kernels, more layers, or dilations).
  - This typically increases parameters and/or depth.
- The weights depend only on the **relative offset** (position), not the actual content of the tokens.

This makes standard convolutions inefficient for **very long sequences** and **input-dependent** dependency patterns.

---

### 1.3 Long (continuous) convolutions

Idea: instead of learning a **separate matrix per offset**, we learn a **function** that generates those matrices.

Let
$$
g : \mathbb{R} \to \mathbb{R}^{e \times e}
$$
be a neural block that takes a scalar offset $i-j$ and returns a weight matrix. Then we write
$$
h_i = \sum_{j=1}^n g(i - j)\,x_j.
$$

Properties:

- The sum now ranges over **all tokens**: the convolution spans the entire sequence.
- This is called a **long convolution**, or a **continuous convolution**:
  - “Continuous” because $g$ can be evaluated at arbitrary (even non-integer) offsets, allowing **intermediate positions** or **variable resolutions**.
- Number of parameters depends only on $g$, **not on $n$** (sequence length).

We can recover a standard convolution by a particular choice of $g$. For example:
$$
g(i-j) =
\begin{cases}
W_{i-j}, & |i-j| \le k, \\
0,       & \text{otherwise},
\end{cases}
$$
which effectively zeroes out contributions outside the receptive field.

However:

- This solves **long-range** dependence (every token can “see” every other),  
- But still **does not handle input-dependent dependencies**: the weight given to token $j$ depends only on the offset $i-j$, not on the actual content $x_i, x_j$.

---

### 1.4 Non-local (content-dependent) models

To make the weights **conditional on the input**, we let $g$ depend on both token contents:
$$
h_i = \sum_{j=1}^n g(x_i, x_j)\,x_j.
$$

- Here, $g(x_i, x_j)$ is an $e \times e$ matrix that depends on the pair of token embeddings.
- This is often called a **non-local model** (or **non-local network**, especially in vision):
  - Any token can attend to any other, with **content-dependent** weighting.

Conceptually:

- **Standard convolution**: local, position-based weights.
- **Continuous/long convolution**: global, still position-based weights.
- **Non-local model**: global, content-based weights.

Self-attention arises as a **simplified, scalable** version of this non-local operator.

---

## 2. The Attention Layer

### 2.1 Dot-product similarity and normalization

We simplify the non-local model by using **scalar** (not matrix) weights derived from a similarity function between tokens.

A simple choice:
$$
g(x_i, x_j) = x_i^\top x_j.
$$

To control the scale, we normalize by the embedding dimension $e$:
$$
g(x_i, x_j) = \frac{1}{\sqrt{e}}\,x_i^\top x_j.
$$

Heuristic motivation:

- Suppose $x_i, x_j \sim \mathcal{N}(0, \sigma^2 I)$.
- Then $x_i^\top x_j$ has variance on the order of $\sigma^4 e$ (grows with $e$).
- Dividing by $\sqrt{e}$ keeps the variance **bounded**, preventing excessively large logits.

We then normalize over the set of tokens with a **softmax**:
$$
h_i = \sum_{j=1}^n \operatorname{softmax}_j \big( g(x_i, x_j) \big)\, x_j,
$$
where $\operatorname{softmax}_j$ denotes softmax taken over the index $j$ for fixed $i$.

Interpretation:

- For each token $i$, the numbers
  $$
  \alpha_{ij} = \operatorname{softmax}_j(g(x_i, x_j))
  $$
  are **attention scores**: a probability distribution over tokens $j$.
- Token $i$ has a fixed “budget” of attention it allocates over all tokens. Increasing $g(x_i, x_j)$ for one $j$ decreases others because of the softmax normalization.

---

### 2.2 Queries, keys, and values

So far, $g$ uses the raw embeddings $x_i, x_j$.
To add **trainable flexibility**, we introduce three linear projections:

- **Key projection**:  
  $$
  k_i = W_k^\top x_i
  $$
- **Value projection**:  
  $$
  v_i = W_v^\top x_i
  $$
- **Query projection**:  
  $$
  q_i = W_q^\top x_i
  $$

where $W_k, W_q, W_v$ are trainable matrices and $k_i,q_i,v_i$ are the corresponding projected vectors.

We then compute **self-attention (SA)** as:
$$
h_i = \sum_{j=1}^n \operatorname{softmax}_j\big( g(q_i, k_j) \big)\, v_j,
$$
typically with
$$
g(q_i, k_j) = \frac{1}{\sqrt{k}}\, q_i^\top k_j,
$$
where $k$ is the dimensionality of queries and keys.

Vectorized form:

Let
$$
K = X W_k, \quad V = X W_v, \quad Q = X W_q,
$$
with shapes:
- $K \sim (n, k)$ (keys),
- $V \sim (n, v)$ (values),
- $Q \sim (n, k)$ (queries).

Then
$$
\operatorname{SA}(X) = \operatorname{softmax}\!\left( \frac{Q K^\top}{\sqrt{k}} \right) V,
$$
where softmax is applied **row-wise**.

We can also express this directly in terms of $X$:
$$
\operatorname{SA}(X) =
\operatorname{softmax}\!\left(
\frac{X W_q W_k^\top X^\top}{\sqrt{k}}
\right) X W_v.
$$

**Definition (Self-attention layer)**  
For input $X \sim (n, e)$, the self-attention layer with dot-product scoring is:
$$
\operatorname{SA}(X) =
\operatorname{softmax}\!\left(
\frac{X W_q W_k^\top X^\top}{\sqrt{k}}
\right) X W_v,
$$
with trainable parameters $W_q, W_k, W_v$ and parameter count independent of sequence length $n$.

---

### 2.3 Dictionary analogy: queries, keys, values

A **Python dictionary** stores pairs (key, value). Querying with a key returns its unique value:

```python
d = dict()
d["Alice"] = 2
d["Alice"]  # returns 2
d["Alce"]   # error: key not found
```

- Keys are **IDs**.
- Queries must match exactly; otherwise, an error occurs.

If we:

- Allow a **similarity** measure between keys and queries,
- Define the dictionary to return the value corresponding to the **closest** key (via argmax similarity),

then with vector-valued keys, queries, and values this becomes a **“hard attention”** mechanism:

- For each query, pick $j^* = \arg\max_j g(q_i, k_j)$,
- Output $v_{j^*}$.

Self-attention can be seen as a **soft relaxation** of this:

- Instead of a hard $\arg\max$, we use a softmax over all $j$.
- Each token gets a **weighted combination** of all values rather than one discrete choice.

The **hard** variant is hard to train with gradient descent because $\arg\max$ is non-differentiable (gradients are zero almost everywhere). The standard SA layer avoids this by using a differentiable softmax.

------

## 3. Multi-head Attention (MHA)

### 3.1 Definition

A single self-attention head may need to capture **multiple, distinct relationships** between tokens (e.g., “cat–table” vs “cat–mother”). Multi-head attention allows separate modeling of these relationships.

We choose a number of heads $h$ and instantiate per-head projections:
$$
K^{(e)} = X W_{k,e}, \quad
V^{(e)} = X W_{v,e}, \quad
Q^{(e)} = X W_{q,e}, \quad e = 1, \dots, h.
$$

Each head computes its own self-attention:
$$
\operatorname{SA}_e(X) =
\operatorname{softmax}!\left(
\frac{Q^{(e)} {K^{(e)}}^\top}{\sqrt{k}}
\right) V^{(e)}.
$$

We then concatenate the outputs from all heads and apply a final projection:
$$
\operatorname{MHA}(X) =
\big[, \operatorname{SA}_1(X) ,\Vert, \dots ,\Vert, \operatorname{SA}_h(X) ,\big] W_o,
$$
where:

- $\Vert$ denotes concatenation along the feature dimension,
- Each $\operatorname{SA}_e(X) \sim (n, v)$,
- The concatenated tensor has shape $(n, h v)$,
- $W_o \sim (h v, o)$ is an output projection matrix.

This allows each head to focus on **different types of dependencies** or “circuits” in the data.

------

### 3.2 Heads and circuits: residual stream interpretation

In practice, the MHA layer is used inside a **residual block**. With a residual connection, the update for token $i$ can be written schematically as
$$
x_i \leftarrow x_i + \sum_{e=1}^h \sum_{j=1}^n
\alpha^{(e)}(x_i, x_j) , W^{(e)} x_j,
$$
where:

- $\alpha^{(e)}(x_i,x_j)$ is the attention score for head $e$ from token $i$ to token $j$,
- $W^{(e)}$ combines the value and output projections for head $e$.

Interpretation:

- The embeddings $x_i$ form a **residual stream** that flows through the network.
- Each head:
    - **Reads** from the residual stream via attention (selecting $x_j$),
    - Then **writes back** a linear combination of those tokens via $W^{(e)}$.
- This viewpoint is central in **mechanistic interpretability**, where people try to identify meaningful **circuits** implemented by specific heads and neurons.

------

## 4. Positional Embeddings

### 4.1 Permutation equivariance of self-attention

We study how an MHA layer behaves if we **permute** the order of tokens.

**Definition (Permutation matrix)**

A permutation matrix $P \in {0,1}^{n \times n}$ has exactly one 1 per row and per column:
$$
\mathbf{1}^\top P = \mathbf{1}^\top, \quad P \mathbf{1} = \mathbf{1},
$$
and satisfies $P^\top P = I$.

Effect:

- Left-multiplying a matrix by $P$ **reorders its rows** (tokens).
- Right-multiplying by $P^\top$ **reorders columns**.

For self-attention, with dot-product scoring:

1. Softmax over rows/columns is compatible with permutation:
    $$
    \operatorname{softmax}(P X P^\top) = P \operatorname{softmax}(X) P^\top.
    $$
2. This leads to permutation equivariance of SA:
    $$
    \operatorname{SA}(P X) = P , \operatorname{SA}(X).
    $$

Reasoning:

- For a fixed token $i$, its output is a **sum over all tokens** of pairwise functions, hence invariant under reordering of the other tokens.
- For the full matrix, permuting the input simply permutes the outputs accordingly.

Conclusion:

- **SA and MHA are permutation-equivariant**: they treat the input as a **(multi)set**, not as an ordered sequence.
- For tasks like text, this is **undesirable**: reversing all tokens just reverses the outputs, although the reversed sentence may be meaningless.

We therefore need to inject **positional information**.

------

### 4.2 Absolute positional embeddings

Let $X \sim (n, e)$ be the token embeddings of a sequence, and fix a **maximum sequence length** $m$.

Introduce **positional embeddings**
$$
S \sim (m, e),
$$
where $S_i$ encodes “position $i$” in the sequence.

We then add them to the input:
$$
X' = X + S_{1:n},
$$
so that the $i$-th row
$$
X'_i = x_i + S_i
$$
encodes both **content** and **position**.

Because $S$ itself is **not permuted** when we reorder the tokens (it is fixed to positions), the overall layer is no longer permutation-equivariant:
$$
\operatorname{MHA}(P X + S) \neq P , \operatorname{MHA}(X + S).
$$

#### 4.2.1 How to build absolute positional embeddings

Several strategies:

1. **Learned positional embeddings**

    - Treat $S$ as a trainable parameter matrix, like a lookup table:
        - Position $i$ has its own learned vector $S_i$.
    - Simple and works well when sequence length is relatively stable.

2. **Naive deterministic strategies (poor choices)**

    - A single scalar $p = i/m$ added to each token:
        - Too weak; adding one scalar to a high-dimensional embedding has limited expressiveness.
    - One-hot encoding of positions in $\mathbb{R}^m$:
        - High-dimensional and extremely sparse; not efficient.

3. **Sinusoidal positional embeddings (original transformer)**

    - Use sines (and cosines) of various frequencies to encode position:
        - For a scalar position $i$ and frequency $\omega$:
            $$
            y = \sin(\omega i).
            $$
        - Different $\omega$ give different oscillation speeds.

    Analogy: a clock

    - Second-hand: frequency $1/60$ Hz (once per minute) distinguishes time within a minute.
    - Minute-hand: much slower frequency distinguishes minutes within an hour.
    - Hour-hand: slower again, distinguishes hours within a day.
    - Together, the three hands (three frequencies) can distinguish time within a day.

    Similarly, we use **multiple frequencies** to represent position:

    - For position $i$, define
        $$
        S_i =
        \big[
        \sin(\omega_1 i), \cos(\omega_1 i),
        \dots,
        \sin(\omega_{e/2} i), \cos(\omega_{e/2} i)
        \big],
        $$
        where the frequencies $\omega_j$ form a **geometric progression**, e.g.
        $$
        \omega_j = \frac{1}{10000^{2(j-1)/e}},
        $$
        so that they span a range from about $1$ down to $1/10000$.

    Key property:

    - For sinusoidal embeddings, positions $i$ and $i+\Delta$ are related by a **linear transformation** (a rotation depending only on $\Delta$).
    - This makes **relative offsets** easy to represent and manipulate by linear layers.

------

### 4.3 Relative positional embeddings

Absolute positional embeddings encode an **absolute index** $i$. For very long sequences, a more flexible approach is to encode **relative offsets** $i-j$.

We modify the attention scoring function to:
$$
g(x_i, x_j) \ \to\ g(x_i, x_j, i - j),
$$
so attention depends on both the content and the relative distance.

Example: add a **trainable bias matrix** $B \sim (m, m)$:
$$
g(x_i, x_j) = x_i^\top x_j + B_{ij},
$$
where $B_{ij}$ depends only on the offset (or absolute pair) of positions $(i,j)$.

Notes:

- **Absolute embeddings** are added once at the input.
- **Relative embeddings** must be incorporated **every time** an MHA layer is used.

Variants:

- **ALiBi (Attention with Linear Biases)**:
    - Each head has a single trainable scalar that multiplies a function of the offset (e.g., a linear slope).
    - Encourages monotonic decay of attention with distance, and extrapolates better to longer sequences.
- **RoPE (Rotary Positional Embeddings)**:
    - Encodes relative positions via complex-valued or 2D rotations in feature space.
    - Also operates directly within the dot-product between queries and keys.

These methods enrich the attention mechanism with **position-awareness** while preserving much of its flexibility and scalability.

------

## 5. Building the Transformer Model

### 5.1 Transformer block: MHA + MLP + normalization

A transformer block typically alternates:

- **Token mixing**: multi-head self-attention (MHA) across tokens.
- **Channel mixing**: a position-wise MLP applied independently to each token.

#### 5.1.1 MLP block

For a token $x \in \mathbb{R}^e$, define:
$$
\operatorname{MLP}(x) = W_2 ,\phi(W_1 x),
$$
where:

- $W_1 \sim (p, e)$ expands to a **bottleneck** dimension $p$,
- $W_2 \sim (e, p)$ projects back to dimension $e$,
- $p$ is often $3e$ or $4e$,
- $\phi$ is a nonlinearity (e.g., ReLU, GELU),
- Biases are often omitted (the widened hidden dimension gives enough flexibility).

Analogy:

- MHA mixes **across tokens** (sequence dimension).
- MLP mixes **across channels** (embedding dimension).
- This is similar in spirit to **depthwise-separable convolutions**: separate spatial and channel mixing.

#### 5.1.2 Residual connections and LayerNorm: pre- vs post-norm

To train deep models efficiently, we add:

- **Residual connections** around MHA and MLP,
- **Layer normalization (LN)** before or after them.

Two common variants:

**Post-normalized transformer block (original)**

1. Apply MHA, then residual, then LN:
    $$
    H^{(1)} = \operatorname{LN}\big( X + \operatorname{MHA}(X) \big).
    $$
2. Apply MLP, residual, then LN:
    $$
    Y = \operatorname{LN}\big( H^{(1)} + \operatorname{MLP}(H^{(1)}) \big).
    $$

**Pre-normalized transformer block (modern default)**

1. LN, then MHA, then residual:
    $$
    H^{(1)} = X + \operatorname{MHA}(\operatorname{LN}(X)).
    $$
2. LN, then MLP, then residual:
    $$
    Y = H^{(1)} + \operatorname{MLP}(\operatorname{LN}(H^{(1)})).
    $$

Empirically:

- **Post-norm**: matches the original transformer design.
- **Pre-norm**: often more **stable and faster to train**, especially for very deep transformers.

Many other block variants exist, but they are refinements of this core pattern.

------

### 5.2 Full transformer encoder: end-to-end structure

A basic transformer model for a sequence task:

1. **Tokenization and embedding**

    - Convert raw input (e.g., text) into a sequence of tokens.
    - Embed tokens into vectors:
        $$
        X \sim (n, e).
        $$

2. **Positional embeddings**

    - If using absolute embeddings, add them:
        $$
        X \leftarrow X + S_{1:n}.
        $$

3. **Transformer blocks**

    - Apply $L$ transformer blocks (each containing MHA + MLP + residual + LN).
    - Output:
        $$
        H \sim (n, e),
        $$
        same number of tokens and same embedding dimension as input.

4. **Task-specific head**

    Example: **classification** with simple pooling

    - Pool over tokens, e.g. average:
        $$
        \bar{h} = \frac{1}{n} \sum_{i=1}^n H_i.
        $$
    - Apply a classification MLP and softmax:
        $$
        y = \operatorname{softmax}\big( \operatorname{MLP}(\bar{h}) \big).
        $$

Key architectural properties:

- The transformer **does not change**:
    - The number of tokens $n$ (no spatial pooling).
    - The embedding dimension $e$ (due to residual connections).
- The model treats inputs as a **set of tokens** (with positional information injected separately),
    which makes it easy to:
    - Add or remove tokens,
    - Reinterpret tokens as arbitrary objects (subwords, image patches, views, graph nodes, etc.).

------

## 6. Class Tokens and Register Tokens

### 6.1 Class token

Instead of pooling over tokens, we can introduce a dedicated **class token** $c \in \mathbb{R}^e$:

1. Add $c$ as an extra token:
    $$
    X \leftarrow
    \begin{bmatrix}
    c^\top \
    X
    \end{bmatrix},
    $$
    so $X \sim (n+1, e)$.
2. Run the transformer blocks to obtain:
    $$
    H \sim (n+1, e).
    $$
3. Use **only the class token** (e.g. last or first row) for classification:
    $$
    y = \operatorname{softmax}\big( \operatorname{MLP}(H_{n+1}) \big)
    $$
    (assuming the class token is at index $n+1$).

Interpretation:

- Attention heads learn to **route** task-relevant information **into the class token’s residual stream**.
- The class token acts as an **information sink** that aggregates the sequence for downstream prediction.

### 6.2 Register tokens

We can also add extra tokens that are **never directly read by the classification head**:

- Call them **registers**: $r_1, \dots, r_R \in \mathbb{R}^e$.
- They are concatenated to the input tokens (like class tokens).
- During training, attention can learn to:
    - Use registers as **scratch space**,
    - Store intermediate global information,
    - Improve attention quality or expressiveness.

Findings:

- Adding register tokens can improve performance and interpretability:
    - The model can offload some computations into these special slots,
    - Regular tokens can focus more on representing local content.

This flexibility is a direct consequence of the **set-like nature** of the transformer architecture.

------

## 7. Practical Exercise: Multi-view Model with a Transformer Head

The chapter suggests a concrete exercise combining **CNN backbones** with a **transformer block** to build a **multi-view classifier**.

Scenario:

- For each object (e.g., an image-classification sample), we have multiple **views**:
    - Different augmentations,
    - Different camera angles,
    - Different frames from a monitoring system.
- We want a **single prediction per object**, **invariant** to the **order** of views.

### 7.1 Step 1 – Construct multi-view inputs

Given an image $x \sim (h, w, c)$:

- Apply $v$ random data augmentations to obtain views $x^{(1)}, \dots, x^{(v)}$.
- Stack them:
    $$
    x' \sim (v, h, w, c),
    $$
    where $v$ is the number of views for this object.

Notes:

- Every view $x^{(i)}$ shares the same label $y$ as the original image.
- The number of views $v$ can vary between examples / mini-batches.

### 7.2 Step 2 – Convolutional backbone per view

Let $g$ be a CNN (e.g., any backbone from earlier chapters) that maps one view to a fixed-dimensional embedding:

- For each view:
    $$
    h_i = g(x^{(i)}) \in \mathbb{R}^e.
    $$
- Stack them:
    $$
    H =
    \begin{bmatrix}
    h_1^\top \
    \vdots \
    h_v^\top
    \end{bmatrix}
    \sim (v, e).
    $$

Here, each view is now a **token** in the transformer sense.

### 7.3 Step 3 – Transformer block over views

We want the full model to be **permutation invariant** over the views:

- Reordering the views should not change the prediction.

To achieve this:

- Any module applied on $H$ should be **permutation-equivariant**.
- A **transformer block** satisfies this (ignoring explicit positional embeddings over views).

So:

1. Apply a transformer block (or several) to $H$:
    $$
    H' = \operatorname{TransformerBlock}(H).
    $$
2. Since token dimension is “views”, and MHA is permutation-equivariant, changing the order of views just permutes rows of $H'$.

Implementation notes:

- You can implement MHA manually in PyTorch or using helper libraries like `einops`.
- You can compare with the built-in `torch.nn.MultiheadAttention`.

### 7.4 Step 4 – Pooling and classification

After the transformer block, we still have shape $H' \sim (v, e)$.

To get a **permutation-invariant** representation:

- Apply a permutation-invariant operation over views, e.g. **average pooling**:
    $$
    \bar{h} = \frac{1}{v} \sum_{i=1}^v H'_i.
    $$
- Then apply a classification head:
    $$
    y = \operatorname{softmax}\big( \operatorname{MLP}(\bar{h}) \big).
    $$

Optionally:

- Instead of average pooling, you can:
    - Add a **class token** to $H$ and let it aggregate information via attention,
    - Then use that token as in Section 6.1.

Baseline:

- Removing the transformer block and just averaging CNN embeddings over views is a valid baseline:
    - Simple average is already a permutation-invariant operation.
    - The transformer block offers a more expressive, content-aware way of combining views.

The exercise illustrates:

- How transformers can operate over **sets** (here, a set of views),
- How **MHA + pooling** yields a permutation-invariant model,
- How convolutional backbones and transformer heads can be combined in practice.
