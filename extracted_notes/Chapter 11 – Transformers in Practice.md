# Chapter 11 – Transformers in Practice

[TOC]

---

## 0. High-Level Overview

- Introduces **encoder–decoder transformers** for sequence-to-sequence (seq2seq) tasks such as machine translation.
- Shows how to make transformers **causal** via **masked self-attention** and how to build **cross-attention** between separate sequences.
- Interprets attention as a form of **neural memory**, relating cross-attention to standard **MLPs**.
- Describes the **original encoder–decoder Transformer** and contrasts it with **encoder-only** and **decoder-only** architectures.
- Analyzes **computational costs** of attention: quadratic time and memory, **linear-time approximations**, **online softmax**, **FlashAttention**, and the **KV cache**.
- Extends transformers beyond text: **Vision Transformers (ViT)**, **audio transformers** (Wav2Vec, Whisper), **neural audio codecs**, **time series**, and **multimodal** models.
- Surveys **transformer variants**: parallel blocks, multi-query attention, mixer/MetaFormer models, gated units (GLUs, GAU, gMLP, LLaMA MLPs), and MONet.
- Closes with **practical exercises** for implementing ViT, mixer models, and GPT-like models.

---

## 1. Encoder–Decoder Transformers

### 1.1 Seq2seq Tasks and Encoder–Decoder (ED) Design

The transformer model described earlier (Chapter 10) can be used for **regression** or **classification** on a single input sequence. These are often called **encoder-only transformers**: they process an input sequence and produce a fixed or sequence-level output (e.g., sentiment score).

However, many tasks are **sequence-to-sequence (seq2seq)**:

- Both **input** and **output** are sequences.
- There is generally **no trivial token-wise alignment** between input and output.
- Example: **machine translation**, where the output is a sentence in another language, with different length and word order.

A common differentiable architecture for seq2seq is the **encoder–decoder (ED)** design:

- **Encoder**:
  - Takes input sequence $X$.
  - Produces a **transformed representation** $H$ (sequence of embeddings, possibly compressed).
- **Decoder**:
  - Autoregressively generates the output sequence token-by-token.
  - Conditions on both:
    - previously generated output tokens, and
    - the encoder representation $H$ of the input.

To build an ED transformer:

- Use a **standard transformer encoder** (multi-head self-attention + MLP + residuals + normalization).
- Build a **decoder** that adds:
  1. **Causality** (for autoregression).
  2. **Cross-attention** to the encoder’s output.

---

### 1.2 Causal Multi-Head Attention

To make a transformer **causal** (suitable for autoregressive generation), we only need to modify the part where tokens interact: the **multi-head attention (MHA)** block. MLPs and layer norms are token-wise, so they do not break causality.

Let $Q, K, V \in \mathbb{R}^{n \times k}$ (ignoring head structure) denote query, key, and value matrices for a sequence of length $n$ and key dimension $k$. Standard self-attention:

$$
\text{SA}(X)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{k}}\right)V.
$$

To enforce **causality** (token $i$ cannot attend to future tokens $j>i$), introduce a **mask matrix** $M \in \mathbb{R}^{n\times n}$ and define:

$$
\text{Masked-SA}(X)
= \operatorname{softmax}\left(\frac{QK^\top \odot M}{\sqrt{k}}\right)V.
$$

Here $\odot$ is element-wise multiplication and $M$ is chosen so that **future positions are effectively removed** from softmax:

$$
M_{ij} =
\begin{cases}
-\infty & \text{if } i > j \quad \text{(disallow attending to the future)} \\
1 & \text{otherwise.}
\end{cases}
$$

In practice, $-\infty$ is replaced with a large negative constant (e.g. $-10^9$), because numerical libraries cannot represent true $-\infty$ directly.

#### Why the Mask Goes Inside the Softmax

A wrong implementation would be:

$$
\text{Wrong}(X)
= \left[ \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{k}}\right)\odot M \right] V.
$$

This fails for two key reasons:

- The **softmax denominator** still includes *all* tokens, including non-causal ones, so their probabilities influence the normalization.
- Setting $M_{ij}=0$ for disallowed positions is not enough, because softmax depends on $\exp(\cdot)$ before masking. Using $0$ in the logits corresponds to a weight $\exp(0)=1$, not suppression.

Correct behavior relies on setting logits of forbidden positions to $-\infty$, so their softmax weight is $\exp(-\infty)=0$. Therefore, **masking must be applied to the logits before softmax**.

Because the **only** interaction between tokens is in attention, replacing self-attention with **causal (masked) self-attention** is sufficient to make the entire transformer block causal.

---

### 1.3 Cross-Attention

We now consider how to condition an attention-based model on a **separate input sequence**, such as the output of an encoder.

Write the self-attention (ignoring heads) more explicitly as:

$$
\text{SA}(X_1, X_2, X_3)
= \operatorname{softmax}\left(
\frac{X_1 W_q (X_2 W_k)^\top}{\sqrt{k}}
\right)
X_3 W_v.
$$

- $X_1$ produces **queries** $Q = X_1 W_q$.
- $X_2$ produces **keys** $K = X_2 W_k$.
- $X_3$ produces **values** $V = X_3 W_v$.

Standard self-attention is the special case:

$$
\text{SA}(X) = \text{SA}(X, X, X),
$$

which explains the term “self” (queries, keys, and values all derive from the same $X$).

To define **cross-attention** between a sequence $X$ and a separate sequence $Z$, let:

- Queries come from $X$.
- Keys and values come from $Z \in \mathbb{R}^{m\times e}$.

Then the **cross-attention (CA)** operation is

$$
\text{CA}(X, Z)
= \operatorname{softmax}\left(
\frac{X W_q (Z W_k)^\top}{\sqrt{k}}
\right)
Z W_v,
$$

which we can also write as

$$
\text{CA}(X, Z) = \text{SA}(X, Z, Z).
$$

**Interpretation**:

- Each token in $X$ is updated according to its similarity to a set of **external key–value pairs** derived from $Z$.
- We say that $X$ is **cross-attending on** $Z$.
- Conceptually, this is similar to concatenating the tokens from $X$ and $Z$ into a single sequence and then applying self-attention with a carefully designed **attention mask** that allows information to flow from $Z$ to $X$, but not necessarily vice versa.

---

### 1.4 Cross-Attention as a Neural Memory and Relation to MLPs

Consider a simplified version of cross-attention where keys and values are **explicit parameters**, rather than derived from an input sequence. Define:

$$
\text{NeuralMemory}(X)
= \operatorname{softmax}\left(\frac{X W_q K^\top}{\sqrt{k}}\right)V.
$$

Here:

- $W_q$ is a **learned query projection**.
- $K$ and $V$ are **learned matrices** whose rows can be seen as **memory slots**:
  - Rows of $K$ are **keys**.
  - Rows of $V$ are **stored values**.

This is often called a **memory layer**:

- The model can **store patterns** (templates, prototypes, etc.) in $(K, V)$.
- At inference, each input $X$ generates queries that retrieve appropriate values from this memory via an attention-like softmax.

If we simplify further by:

- Setting $W_q = I$ (identity).
- Ignoring the $\frac{1}{\sqrt{k}}$ term.
- Replacing the softmax with a generic activation $\phi$.

we obtain:

$$
\text{MLP}(X) = \phi(XK) V.
$$

This is precisely a **two-layer MLP**:

- First layer: $X \mapsto \phi(XK)$.
- Second layer: $\phi(XK) \mapsto \phi(XK) V$.

Thus:

> **Key Idea**: The usual MLP block in a transformer can be interpreted as an approximation to attention over a fixed, trainable set of keys and values.

In practice, inspecting the nearest neighbors induced by key–value memory in trained models often reveals **human-understandable clusters and patterns**.

---

### 1.5 Encoder–Decoder Transformer Architecture

With **causal attention** and **cross-attention** in hand, we can describe the **original encoder–decoder Transformer** (from [VSP+17]):

1. **Encoder**:
   - Input sequence $X$ is embedded and passed through a stack of transformer blocks (self-attention + MLP).
   - Produces an **encoder output sequence** $H$ of contextual embeddings.

2. **Decoder**:
   - Predicts the **output sequence autoregressively**.
   - Each decoder block has three components:
     1. **Masked self-attention** over the partially generated output sequence (to enforce causality).
     2. **Cross-attention** where:
        - Queries come from the current decoder hidden states.
        - Keys and values come from the encoder output $H$.
     3. A **token-wise MLP** applied to each position.

3. **Output projection**:
   - Final decoder states are mapped to logits over the output vocabulary, enabling next-token prediction.

#### Decoder-Only Models

A **decoder-only transformer** removes the cross-attention block (component 2 above). Each block then consists of:

- Masked self-attention (causal MHA).
- MLP with residual and normalization.

Modern **large language models (LLMs)** like GPT-2 and most open-source GPT-style models (e.g., LLaMA) are **decoder-only** models trained to predict the **next token** in a sequence of text.

Many seq2seq tasks can be handled by decoder-only models by **concatenating the input and output sequences**:

- Example: `[prompt + input tokens] ⟶` model generates output tokens appended to that sequence.
- This reduces the need for a separate encoder, which is why **encoder–decoder transformers are now less common** for generic text tasks, even though they remain conceptually clean for seq2seq.

---

## 2. Computational Considerations

### 2.1 Time Complexity and Linear-Time Transformer Variants

Consider the simple self-attention implementation:

- $Q, K, V \in \mathbb{R}^{n\times k}$.
- Compute the attention matrix $A = QK^\top \in \mathbb{R}^{n \times n}$.
- Apply softmax row-wise on $A$ and multiply by $V$.

The dominant computation, $QK^\top$, has **time complexity**:

$$
\mathcal{O}(n^2 k),
$$

since it involves $n^2$ dot products of dimension $k$.

**Comparison with 1D convolution**:

- A 1D convolution with fixed kernel size has **linear complexity** in $n$ (up to a constant factor depending on kernel size).
- For very long sequences, the **quadratic growth** of attention in $n$ can become a serious bottleneck.

This has motivated:

- **Faster autoregressive decoding** algorithms (e.g., speculative decoding).
- **Linear or sub-quadratic transformer variants**, where attention is approximated or replaced to reduce the $n^2$ factor.

One important idea: replace full self-attention with cross-attention to a **fixed-size set of latent tokens** $Z$:

- $Z \in \mathbb{R}^{m\times e}$, with $m$ chosen as a hyper-parameter (latent bottleneck size).
- Complexity becomes roughly $\mathcal{O}(nm)$ with $m \ll n$.
- This idea underlies architectures like the **Perceiver**, which distill long sequences into a smaller latent space.

#### Memory-Bound Behavior

On modern hardware, a naive self-attention implementation is often **memory-bound**:

- The time spent moving data (reads/writes) can dominate the floating-point compute.
- As a result, theoretical gains from “linear-time attention” do **not always translate into real wall-clock speedups**, especially when:
  - Approximations reduce accuracy.
  - Highly optimized quadratic attention kernels are available.

This trade-off has driven the development of **carefully engineered attention implementations**, such as **FlashAttention-2**, which achieve very high efficiency without changing the exact attention formula.

---

### 2.2 Online Softmax and FlashAttention-Style Implementations

The naive attention implementation also has **quadratic memory complexity**:

- It explicitly materializes the full attention matrix $QK^\top \in \mathbb{R}^{n\times n}$.
- This costs $\mathcal{O}(n^2)$ memory and limits sequence length.

However, this can be significantly improved with a **blockwise (chunked) computation** of softmax, sometimes referred to as **online softmax**:

#### Derivation for a Single Query

For a single query vector $q$ and keys $K = [k_1^\top; \dots; k_n^\top]$, values $V = [v_1^\top; \dots; v_n^\top]$, standard attention is:

$$
\text{SA}(q, K, V)
= \frac{\sum_{j=1}^n \exp(k_j^\top q)\, v_j}{\sum_{j=1}^n \exp(k_j^\top q)}.
$$

Now split $K$ and $V$ into two chunks:

$$
K = \begin{bmatrix} K_1 \\ K_2 \end{bmatrix},
\quad
V = \begin{bmatrix} V_1 \\ V_2 \end{bmatrix}.
$$

For each chunk $i \in \{1,2\}$ define:

$$
h_i = \sum_j \exp(k_{ij}^\top q)\, v_{ij},
\quad
L_i = \sum_j \exp(k_{ij}^\top q).
$$

Then:

$$
\text{SA}(q, K, V)
= \frac{h_1 + h_2}{L_1 + L_2}.
$$

More generally, if we split into $m$ chunks:

$$
\text{SA}(q, K, V)
= \frac{\sum_{i=1}^m h_i}{\sum_{i=1}^m L_i}.
$$

#### Algorithmic Consequences

We can now design an **iterative algorithm**:

1. Initialize running numerator $\tilde{h} \gets 0$ and denominator $\tilde{L} \gets 0$.
2. For each block $i$:
   - Load $K_i, V_i$ from memory.
   - Compute $h_i$ and $L_i$.
   - Update $\tilde{h} \gets \tilde{h} + h_i$, $\tilde{L} \gets \tilde{L} + L_i$.
3. After processing all blocks, output $\tilde{h} / \tilde{L}$.

This avoids ever forming the full $n\times n$ attention matrix in memory. With careful attention to **numerical stability** (e.g., using log-sum-exp normalizations) and **kernel fusion**, this yields attention implementations that:

- Have **linear memory usage** in $n$.
- Significantly improve **runtime** on GPUs/TPUs.
- Exhibit non-obvious behavior (e.g., sometimes running faster at **longer** sequence lengths) due to better hardware utilization.

These ideas underlie efficient implementations such as **FlashAttention-2** and can be extended to **distributed attention** schemes (e.g., RingAttention), where:

- Different devices handle different subsets of queries.
- Blocks of keys/values are rotated among devices to approximate full attention.

---

### 2.3 KV Cache for Autoregressive Decoding

In a **decoder-only** model generating a sequence autoregressively:

- At timestep $t$, we compute attention for the **new token** only.
- The new query $q_t$ must attend to:
  - Past keys $k_1, \dots, k_t$.
  - Past values $v_1, \dots, v_t$.

Crucially, keys and values for **previous positions** do not change. Therefore:

- We can **cache** the matrices $K_{1:t-1}$ and $V_{1:t-1}$.
- For each new token:
  - Compute new $k_t, v_t$.
  - Append them to the cache.
  - Compute attention scores from $q_t$ to all cached keys.

This is called the **KV cache**:

- It **reuses most of the previous computation** for each new token.
- Per-step compute cost grows roughly linearly in $t$, but remains efficient in practice.
- The total cache memory grows **linearly** with sequence length (and head dimension).

Contrast with **causal convolutions**:

- A causal convolution only needs to remember activations within its finite **receptive field**, providing a **fixed upper bound** on memory usage.
- The unbounded growth of the KV cache motivates exploring architectures with **fixed memory cost** during autoregressive generation (a topic for later chapters, e.g., Chapter 13).

---

## 3. Transformers Beyond Text

### 3.1 Transformers for Language Modeling: Encoder-Only, Decoder-Only, Encoder–Decoder

For text, three main architectural patterns are common:

1. **Decoder-Only (GPT-style)**:
   - Example: **GPT-2** and most modern LLMs (e.g., LLaMA).
   - Architecture: stack of **masked self-attention** + **MLP** blocks.
   - Training objective: **next-token prediction** (autoregressive language modeling).
   - Strength: direct text generation, flexible in-context behavior.

2. **Encoder-Only (BERT-style)**:
   - Example: **BERT** and related models.
   - Architecture: stack of **bidirectional self-attention** + **MLP** blocks (no causality constraint).
   - Training objective: **masked language modeling** (predicting randomly masked tokens).
   - Strength: producing rich **contextual embeddings**; used for classification, retrieval, etc.
   - Limitation: cannot directly generate text autoregressively (no causal mechanism).

3. **Encoder–Decoder (T5-style)**:
   - Example: **T5** and related “text-to-text” models.
   - Encoder processes input; decoder generates output autoregressively with cross-attention to the encoder.
   - Can be very effective for seq2seq tasks, but have become **less common** in general LLM practice because:
     - Decoder-only models can handle many tasks via **prompting** and concatenation of input/output.
     - Maintaining both encoder and decoder sometimes increases complexity without enough benefit.

Note: Large-scale language models often undergo **post-training** stages such as **instruction tuning** and **alignment**, but these are not detailed here.

---

### 3.2 Transformers for Images – Vision Transformers (ViTs)

Transformers can be extended to images by designing an appropriate **tokenization** scheme and **positional embeddings**.

**Problem**: Treating each pixel as a token is usually too expensive:

- For an image of size $H \times W$ with $HW$ tokens:
  - Self-attention has complexity $\mathcal{O}((HW)^2)$.

**Vision Transformers (ViTs)** solve this by:

1. **Patch Tokenization**:
   - Split the image into **non-overlapping patches** of size $p \times p$.
   - Suppose the image has $c$ channels.
   - Each patch has shape $p \times p \times c$, flattened to a vector of length $p^2 c$.
   - Each patch is then linearly projected to an embedding of dimension $e$:
     $$
     x_i = W_{\text{patch}}\, \text{patch}_i + b \in \mathbb{R}^{e}.
     $$

2. **Implementation Options**:
   - A **convolutional layer** with kernel size $p$ and stride $p$ naturally implements this “patchify + linear projection”.
   - Libraries like **einops** provide reshaping operators (e.g., `rearrange`) to:
     - Restructure images into patch sequences.
     - Flatten patches and map to embeddings.

3. **Positional Embeddings and Class Token**:
   - Add positional encodings (often **learned**) to each patch embedding.
   - Introduce an additional **[CLS] token** that aggregates information for classification via attention.

Once tokenized, the image is processed by a **standard transformer encoder**. For **image generation**, ViTs can:

- Predict patches in a fixed ordering (row-major or column-major).
- Operate over **discrete patch tokens** obtained from a **vector-quantized autoencoder**.
- Or operate directly on **continuous patch embeddings**.

However, for image generation, **diffusion models** and **flow-matching models** have become more popular, so transformer-based image generation is often not the default choice.

---

### 3.3 Transformers for Audio

Transformers also work well for **audio**, especially speech, once we design a suitable tokenization front-end:

1. **Tokenization via 1D Convolution**:
   - A small convolutional network with stride greater than 1 converts raw waveforms or features (e.g., log-mel spectrograms) into a **sequence of frame-level embeddings**.
   - These embeddings are then fed to a transformer (typically encoder-only or encoder–decoder).

2. **Wav2Vec (Encoder-Only)**:
   - Architecture: encoder-only transformer.
   - Supervised objective: **connectionist temporal classification (CTC) loss**, which aligns output embeddings with a text transcription without requiring frame-level alignment labels.
   - Pretraining: self-supervised learning on unlabeled audio via **masked prediction** (similar in spirit to masked language modeling).

3. **Whisper (Encoder–Decoder)**:
   - Architecture: encoder–decoder transformer.
   - Encoder processes the audio sequence.
   - Decoder **autoregressively** generates the transcription (and possibly other metadata) as text.
   - Advantages:
     - More flexible generation behavior.
     - Reduced need for precisely aligned training labels.
   - Risks:
     - Because the model is generative, it can **hallucinate** content not present in the audio.

4. **Neural Audio Codecs and Generative Audio**:
   - **Neural audio codecs** compress audio into a sequence of **discrete tokens**.
   - A transformer can model the distribution of these tokens (similar to text tokens).
   - This enables **text-to-speech** and other generative tasks:
     - A text encoder produces conditioning embeddings.
     - A transformer generates an audio-token sequence.
     - A decoder reconstructs the audio waveform from tokens.

---

### 3.4 Time Series, Graphs, and Multimodal Transformers

Transformers have been adapted to various data types:

- **Time-series**:
  - Sequences of real-valued vectors (e.g., sensor readings, financial series).
  - Models often incorporate inductive biases for forecasting, irregular time steps, etc.

- **Graphs**:
  - Nodes and edges are converted into sequences (or sets) of tokens.
  - Specialized graph transformers are covered in the next chapter.

- **Multimodal models**:
  - Different modalities (image, audio, text, etc.) are each processed by their own **tokenizers**.
  - Outputs are concatenated into a **single sequence** of tokens and passed through a transformer.
  - Example: an image–text model:
    - Image tokens from an image tokenizer.
    - Text tokens from a text tokenizer (e.g., words or subwords).
    - The transformer processes the concatenated sequence and can be prompted with text like “Describe the image” to generate a textual description.

This separation of **data-specific tokenization** from a largely **modality-agnostic architecture** (the transformer itself) is a core reason for the wide applicability of transformers.

---

## 4. Transformer Variants

This section describes architectural variations and generalizations that preserve the **core transformer pattern** (token mixing + channel mixing + normalization + residuals), but modify the details of the interaction.

### 4.1 Efficiency Tweaks: Parallel Blocks and Multi-Query Attention

1. **Parallel Blocks**:
   - In standard transformers, each block usually applies:
     1. MHA (with residual and normalization),
     2. then MLP (with residual and normalization), sequentially.
   - A **parallel** variant instead computes MHA and MLP in parallel from the same input:
     $$
     H \leftarrow H + \text{MLP}(H) + \text{MHA}(H).
     $$
   - This allows **fusing the initial and final linear projections** of MLP and MHA in optimized implementations, slightly improving efficiency.

2. **Multi-Query Attention (MQA)**:
   - Standard multi-head attention uses separate $(W_q, W_k, W_v)$ per head.
   - **Multi-query MHA** shares the same **key** and **value** projections across all heads and only varies the **queries**.
   - This:
     - Reduces both **parameter count** and **KV cache size**.
     - Lowers **memory bandwidth usage** during decoding.
   - Often used in large LLMs where caching and bandwidth dominate.

---

### 4.2 Mixer Models and MetaFormers

Transformer blocks can be viewed as instances of a broader pattern:

> Alternate between **token-mixing** (across sequence positions) and **channel-mixing** (across embedding dimensions), with residual connections and normalization.

**Mixer models** (e.g., **MLP-Mixer**) exploit this by replacing attention with simpler operations, particularly when the sequence length is fixed (e.g., fixed number of image patches).

Let $H \in \mathbb{R}^{n \times d}$ be a sequence of length $n$ with embedding dimension $d$.

Ignoring normalization, a **mixer block** can be written as:

1. **Channel mixing**:
   $$
   H \leftarrow \text{MLP}_{\text{channel}}(H) + H.
   $$
2. **Token mixing** (operate on the transposed matrix):
   $$
   H \leftarrow \big[ \text{MLP}_{\text{token}}(H^\top) + H^\top \big]^\top.
   $$

These correspond to equations of the form:

$$
H = \text{MLP}(H) + H,
$$

$$
H = \big[\text{MLP}(H^\top) + H^\top\big]^\top.
$$

Variants include:

- **S2-MLP**:
  - Replaces expensive mixing operations with a very simple MLP applied to a **shifted version** of the input along the sequence dimension.
  - Emphasizes that even relatively simple token mixing can work within a strong overall architecture.

The broader class of such architectures, which alternate token and channel mixing without requiring attention, has been dubbed **MetaFormers**. The key observation is:

> Much of the success of transformers comes from the **MetaFormer template** itself (token mixing + channel mixing + residuals + normalization), not from attention per se.

---

### 4.3 Gated Units: GLU, gMLP, LLaMA MLPs, and GAU

Many transformer variants use **gated (multiplicative) interactions** to combine information from different paths.

A generic **gated unit** can be written as:

$$
f(X) = \phi_1(X) \odot \phi_2(X),
$$

where:

- $\phi_1$ and $\phi_2$ are trainable transformations (e.g., MLPs, linear maps, or attention).
- $\odot$ is element-wise (Hadamard) product.

A classical example is the **Gated Linear Unit (GLU)**:

- Let $A$ and $B$ be learned weight matrices.
- Define:
  $$
  \phi_2(X) = XA, \quad
  \phi_1(X) = \sigma(XB),
  $$
  where $\sigma$ is a sigmoid.
- Then:
  $$
  f(X) = (XA) \odot \sigma(XB).
  $$

This structure:

- Allows one branch to act as a **content stream**, the other as a **gate** that modulates it.
- Introduces a flexible but efficient nonlinearity.

**Examples of gated architectures**:

- **gMLP**:
  - Works in a mixer-style architecture.
  - Uses gated units in place of standard channel-mixing MLPs to improve expressivity.

- **LLaMA family**:
  - Uses GLU-like units instead of standard transformer MLPs.
  - Often improves training stability and quality at scale.

- **Gated Attention Unit (GAU)**:
  - Uses a simplified attention-like computation for one branch (e.g., a single attention head) and a linear map for the other.
  - Reduces complexity compared to full MHA while still capturing important interactions.

Gating is also important in **recurrent models**, where similar structures (e.g., LSTM, GRU) use multiplicative gates to control information flow over time.

---

### 4.4 MONet: Multilinear Operator Networks

The **Multilinear Operator Network (MONet)** proposes a minimalist block that uses only **linear projections** and **element-wise multiplications**, removing explicit activation functions entirely.

A MONet block can be written as:

$$
H = E(AX \odot BX + DX),
$$

where:

- $A, B, D, E$ are learned linear operators.
- $DX$ plays a role similar to a **residual connection**.
- $AX \odot BX$ introduces **multiplicative interactions** between features.
- $B$ is often implemented using a **low-rank decomposition** to reduce parameter count.

To ensure **token mixing**, MONet applies a **token-shift operation** (e.g., shifting the sequence positions) in all odd-numbered blocks, so information can propagate between tokens over depth.

Despite the absence of classical nonlinearities like ReLU or GELU, stacking such blocks with residuals and multiplicative interactions yields a highly expressive model, illustrating how far one can simplify transformer-like architectures while retaining power.

---

## 5. From Theory to Practice – Suggested Exercises

The chapter closes with suggestions for hands-on experiments, assuming you now understand the building blocks of differentiable models.

### 5.1 Implementing a Vision Transformer (ViT)

- Choose an image classification dataset.
- Tokenize the image using **patch extraction**:
  1. Use **einops** (e.g., `rearrange`) or
  2. A single convolution with **kernel size = stride = patch size**.
  3. For very small images, you can even treat **each pixel as a token**.
- Add **positional embeddings**:
  - Start with **trainable positional embeddings** (as in original ViT).
  - Experiment with **sinusoidal** or **relative** positional encodings from earlier chapters.
- Stack encoder blocks (self-attention + MLP) and train a classifier.

Note: Training a ViT **from scratch** on small datasets can be challenging and may underperform without many samples (often millions), so some underwhelming results are normal.

### 5.2 Implementing Mixer-Style Models

As a simpler alternative:

- Implement a **Mixer model**:
  - Use the patch tokenization of ViT.
  - Replace attention with **token-MLPs** and **channel-MLPs** as described in Section 4.2.
- Or try **S2-MLP**:
  - Use **shift-based token mixing** plus channel MLP.

These models are often easier to get working on modest datasets and hardware.

### 5.3 Implementing a Small GPT-like Model

Finally, you can implement a small **decoder-only transformer**:

- Use a simple **byte-pair encoding** or character-level tokenizer.
- Build stacked **masked self-attention + MLP** blocks.
- Train on a text corpus with **next-token prediction**.

Educational repositories like **minGPT** provide a concise, didactic skeleton that you can study or use as inspiration for your own implementation.

---

This completes the structured summary of **Chapter 11: Transformers in Practice**, covering encoder–decoder transformers, causality and cross-attention, computational efficiency techniques, applications beyond text, and key architectural variants.