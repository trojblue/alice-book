# Chapter 8 – Convolutions Beyond Images

[TOC]

- Convolutional networks extend naturally from images to 1D (time series, audio, text) and 3D (video, volumes) data.
- Many sequence-like data share a structure: ordered (temporal or spatial) axes plus a feature/channel axis.
- Variable-length sequences are handled via padding and masking; these ideas are central for modern sequence models (including transformers).
- Text processing with CNNs requires tokenization and vector embeddings; embeddings are learned lookup tables with rich algebraic structure.
- Dilated (atrous) convolutions enlarge receptive fields exponentially in depth without increasing parameter count.
- Forecasting (next-step prediction) can be trained in a self-supervised way and extended autoregressively to generate long sequences.
- Causal convolutions (via masking) ensure outputs only depend on the past, enabling flexible forecasting and generative use.
- Autoregressive models define a full generative distribution over sequences via the probability chain rule and can be sampled greedily or with beam search.
- Conditional generative modeling (e.g., prompt-based text generation, multimodal conditioning) is implemented by conditioning the same autoregressive mechanism on extra context.

---

## 8.1 Convolutions for 1D and 3D Data

### 8.1.1 Beyond Images: Time Series, Audio, Video, Text

Many datasets share the same basic structure as images: one or more **ordered** dimensions plus a **feature/channel** dimension.

#### (a) Time series

- A multivariate time series with length $t$ and $c$ measurements per time step (e.g., multiple sensors, multiple stocks) can be represented as:
  $$
  X \in \mathbb{R}^{t \times c},
  $$
  where each row corresponds to a time instant, each column to a feature/sensor.
- Analogy to images:
  - Time index $\leftrightarrow$ pixel position along one axis.
  - Feature index $\leftrightarrow$ channel index (e.g., RGB channels).

#### (b) Audio

- A (raw) audio waveform can also be seen as:
  $$
  X \in \mathbb{R}^{t \times c},
  $$
  where $t$ is the number of samples, $c$ is the number of channels (1 for mono, 2 for stereo, etc.).

**Frequency-domain view**

- Audio can be converted to a **time–frequency image** (e.g., spectrogram, MFCC features):
  - Window the signal in time, perform frequency analysis on each window.
  - Result is approximately an image with axes: time $\times$ frequency, and possibly multiple channels.
- Once in this form, **standard 2D CNNs** for images can be applied.

#### (c) Video and volumetric data

- A video can be modeled as:
  $$
  X \in \mathbb{R}^{t \times h \times w \times c},
  $$
  where:
  - $t$: number of frames,
  - $(h, w)$: spatial resolution of each frame,
  - $c$: channels per frame.
- A volumetric medical scan is similar, with $t$ acting as depth of the volume.

#### (d) Sampling rates and resolution

All of these have an associated **sampling rate** (samples per unit time), akin to spatial resolution in images:

- EEG: around $240$ Hz (240 measurements per second).
- Stock prices: e.g., sampled every minute $\Rightarrow \frac{1}{60}$ Hz.
- Audio: common rates like $44.1 \text{ kHz}$.
- Video: typical frame rates around $24$ fps.

**Key analogy**:
- Image resolution, audio sampling rate, and video frame rate all control the **precision** with which the underlying signal is captured.
- Images can often be resized to a fixed resolution without major semantic changes (up to a point).
- Audio/video **duration** varies significantly; reshaping in time is not meaningful, so datasets naturally contain **variable-length sequences**.
- Example: 3-minute audio at $44.1 \text{ kHz}$ has about $8 \times 10^6$ samples.

#### (e) Spatial vs temporal dimensions; forecasting

- Spatial dimensions (e.g., image axes) are often roughly symmetric (a horizontally flipped image is usually still valid).
- The time axis is **asymmetric**:
  - Reversing an audio or time series produces something generally not “valid” as a forward-time signal.
- This asymmetry is exploited via **causality**:
  - Models often must not “look into the future”.
  - We are often interested in **forecasting**: predicting future values from past observations.

#### (f) Text as a sequence

- Example sentence: “the cat is on the table”.
- It can be split into units (tokens) in many ways:
  - Character-level (or syllable-level) sequences,
  - Word-level sequences,
  - Subword-level sequences.
- **Key difference** from previous examples: tokens are **categorical**, not numerical.
  - We must:
    - **Tokenize**: split the text into tokens.
    - **Embed**: map each token to a numerical vector suitable for neural networks.

In the remainder of the chapter, key issues like variable-length inputs, causality, forecasting, tokenization, embedding, and masking are developed systematically.

---

## 8.1.2 1D and 3D Convolutional Layers

We now extend convolutions to 1D and 3D data, where “dimensionality” here refers to the number of axes along which the convolution slides (not counting channels).

### 8.1.2.1 1D convolution

Let a sequence be represented as:
$$
X \in \mathbb{R}^{t \times c},
$$
where $t$ is sequence length, $c$ is the number of channels (features).

- Choose an **odd** kernel size:
  $$
  s = 2k + 1.
  $$
- For each position $i$, define a local **patch**:
  $$
  P_k^{(i)} \in \mathbb{R}^{s \times c},
  $$
  consisting of rows of $X$ from indices $i-k$ to $i+k$ (zero-padded at the boundaries).

A 1D convolutional layer
$$
H = \mathrm{Conv1D}(X) \in \mathbb{R}^{t \times c'}
$$
with output dimension $c'$ is defined row-wise by:
$$
h_i = \phi\bigl(W \, \mathrm{vec}(P_k^{(i)}) + b\bigr),
$$
where:

- $W \in \mathbb{R}^{c' \times (s c)}$ and $b \in \mathbb{R}^{c'}$ are trainable,
- $\mathrm{vec}(\cdot)$ flattens the patch,
- $\phi$ is a nonlinearity (e.g., ReLU).

**Properties**:

- **Locality**: each $h_i$ depends only on a local neighborhood around time $i$.
- **Translation equivariance**: shifting the input in time leads to a correspondingly shifted output.

An index-wise form replaces vectorization with explicit sums. Using a rank-3 kernel tensor $W \in \mathbb{R}^{s \times c' \times c}$, we can write:
$$
H_{i z} = \sum_{i'=1}^{2k+1} \sum_{d=1}^{c} W_{i', z, d} \, X_{i' + t(i), d},
$$
where $t(i)$ appropriately maps local indices $i'$ to global time indices in $X$, accounting for padding.

### 8.1.2.2 3D convolution

For 3D data (e.g., video) with shape:
$$
X \in \mathbb{R}^{t \times h \times w \times c},
$$
a 3D kernel with size $s = 2k + 1$ in each dimension and output channels $c'$ is:
$$
W \in \mathbb{R}^{s \times s \times s \times c' \times c}.
$$

The 3D convolution at position $(p, i, j)$ (time, height, width) is:
$$
H_{p i j z} = 
\sum_{p'=1}^{2k+1}
\sum_{i'=1}^{2k+1}
\sum_{j'=1}^{2k+1}
\sum_{d=1}^{c}
W_{p', i', j', z, d} \,
X_{p' + t(p), \; i' + t(i), \; j' + t(j), \; d}.
$$

Again, this is local and translation equivariant in the 3D (spatiotemporal) domain. Vectorized versions and 1D/3D max pooling follow similarly.

---

## 8.2 1D and 3D Convolutional Models

We now discuss how to build full models, focusing on 1D data and challenges like variable-length sequences and text.

### 8.2.1 Dealing with Variable-Length Inputs

Consider two sequences (e.g., audio clips, time series, or token sequences):
- $X_1 \in \mathbb{R}^{t_1 \times c}$,
- $X_2 \in \mathbb{R}^{t_2 \times c}$,

with the same number of channels $c$ but different lengths $t_1 \neq t_2$.

Let $g$ be a stack of 1D convolutions and pooling layers (a **feature extractor**):
$$
H_1 = g(X_1), \quad H_2 = g(X_2),
$$
with:
- $H_1 \in \mathbb{R}^{\tilde t_1 \times c'}$,
- $H_2 \in \mathbb{R}^{\tilde t_2 \times c'}$,

where $\tilde t_1, \tilde t_2$ depend on strides/pooling.

After **global average pooling**, length dependence disappears:
$$
h_1 = \frac{1}{\tilde t_1} \sum_{i=1}^{\tilde t_1} (H_1)_i, 
\quad
h_2 = \frac{1}{\tilde t_2} \sum_{i=1}^{\tilde t_2} (H_2)_i,
$$
yielding fixed-size representations that can be passed to a classifier.

**Conceptual issue**: convolutional layers themselves handle variable-length input, but **mini-batch training** requires loading multiple sequences into a single tensor, which requires handling unequal lengths.

#### Strategy 1: Zero-padding in mini-batches

Given a batch of sequences with varying lengths, e.g.,
- $X_1 \in \mathbb{R}^{t_1 \times c}$,
- $X_2 \in \mathbb{R}^{t_2 \times c}$,
- $X_3 \in \mathbb{R}^{t_3 \times c}$,

we find $t_{\max} = \max(t_1, t_2, t_3)$ and pad each sequence with zeros to length $t_{\max}$. We then **stack** them into:
$$
X \in \mathbb{R}^{B \times t_{\max} \times c},
$$
where $B$ is batch size.

- For audio, zero-padding is like adding silence, often harmless.
- In PyTorch, this can be done with utilities like `pad_sequence` and custom collate functions in a `DataLoader`.

#### Strategy 2: Masking

To avoid treating padding as real data, we build a **mask** for each batch, e.g.,
$$
M \in \{0,1\}^{B \times t_{\max}},
$$
where each row is of the form:
$$
M_b = [\underbrace{1, \dots, 1}_{t_b}, \underbrace{0, \dots, 0}_{t_{\max} - t_b}].
$$

This mask:
- Flags **valid** time steps (1) vs **padding** (0).
- Can be used to:
  - Exclude padding from loss computation.
  - Zero out invalid positions before operations.
  - Control attention or other sequence operations in more advanced models.

**Key idea**: **Padding + masking** is a general pattern used for CNNs, RNNs, and transformers alike.

---

### 8.2.2 CNNs for Text Data

Processing text requires extra steps before applying convolution.

#### 8.2.2.1 Tokenization

**Tokenization** splits raw text into discrete **tokens**. Common types:

1. **Character tokenization**  
   - Each character is a token.
   - Pros: simple, no OOV (out-of-vocabulary) issue.
   - Cons: sequences are long; local patterns are more fragmented.

2. **Word tokenization**  
   - Each word is a token.
   - Pros: intuitive, tokens align with semantic units.
   - Cons: large vocabularies, strong OOV issues (rare words, typos, neologisms).

3. **Subword tokenization**  
   - Tokens are units between characters and whole words (e.g., “perplex”, “ing”).
   - Built via algorithms such as **Byte-Pair Encoding (BPE)**.
   - Pros:
     - Balances vocabulary size and expressivity.
     - Handles rare words by composing them from more frequent subunits.

In all cases, we define a **vocabulary**:
- A list of allowed tokens with size $n$.
- The tokenizer maps text $\to$ tokens, then to integer IDs in $\{0, \dots, n-1\}$.
- Rare or unknown tokens are mapped to a special **OOV** (out-of-vocabulary) token.
- Additional **special tokens** may be included, such as:
  - `[BOS]` – beginning of sequence,
  - `[EOS]` – end of sequence,
  - other control tokens.

Large subword vocabularies can behave surprisingly:
- Some numbers (e.g., “52”) may be a single token.
- Others (e.g., “2512”) may be split into multiple tokens (like “251” + “2”).
- Visualizing tokenization is often crucial for debugging.

There is active research on:
- **Byte-level tokenizers** and other schemes that operate at a lower-level representation (bytes rather than characters).
- Tokenizers trained **end-to-end** with the model.
- Specialized numerical tokenizers for handling numbers more systematically.

#### 8.2.2.2 Embeddings via a lookup table

After tokenization, we have sequences of integer IDs:
$$
x = (x_1, \dots, x_m), \quad x_i \in \{0, \dots, n-1\},
$$
where $n$ is vocabulary size.

We want to map each token ID to a **dense vector**.

- Fix an embedding dimension $e$.
- Create an **embedding matrix**:
  $$
  E \in \mathbb{R}^{n \times e},
  $$
  where row $E_{j}$ is the embedding of token $j$.

**Lookup operation**:
$$
X = \mathrm{LookUp}(x) =
\begin{bmatrix}
E_{x_1} \\
E_{x_2} \\
\vdots \\
E_{x_m}
\end{bmatrix}
\in \mathbb{R}^{m \times e}.
$$

This replaces discrete IDs with a sequence of embedding vectors. We can then apply any 1D model, e.g.:
$$
\hat y = \mathrm{CNN}(X).
$$

- The embedding matrix $E$ is trained via gradient descent **jointly** with all other model parameters.
- Instead of sparse one-hot vectors, embeddings are dense and low-dimensional.

A typical **TextCNN** architecture:

- Input: integer token IDs, shape `(batch, length)`.
- Embedding layer: `(batch, length, e)`.
- Transpose for Conv1D (depending on framework) to `(batch, e, length)`.
- 1D convolutions (possibly several layers with ReLU).
- Max pooling in time or global average pooling.
- A linear head for classification.

#### 8.2.2.3 Embeddings as a general concept

**Key idea (Embeddings)**:  
Any mapping that converts complex objects (tokens, words, sentences, images, graphs) into numeric vectors that:

- Capture semantic properties,
- Allow algebraic operations (e.g., distances, averages),

is an **embedding**.

Examples:

- Word embeddings: semantically similar words have nearby vectors.
- Image embeddings: CNN backbones + global pooling yield vectors for images; similar images can be found via nearest neighbors in embedding space.

Embeddings are foundational for search and retrieval systems across many modalities.

---

### 8.2.3 Dealing with Long Sequences: Dilated Convolutions

Standard convolutions have **local receptive fields**. To cover a longer temporal context, one might:

- Use larger kernels, or
- Stack more layers.

However:

- Larger kernels increase the number of parameters.
- Deeper models increase compute cost and can make training harder.

**Dilated (atrous) convolutions** introduce an extra hyperparameter: the **dilation rate** $r$.

- Dilation rate $r = 1$: standard convolution.
- Dilation rate $r = 2$: the patch elements are sampled every two positions.
- Dilation rate $r = 4$: sample every four positions, etc.

For a 1D convolution with kernel size $s = 2k+1$ and dilation $r$, the patch for position $i$ uses indices:
$$
i - r k, \, i - r(k-1), \dots, i, \dots, i + r(k-1), \, i + r k.
$$

**Effects**:

- The **number of parameters** is unchanged (still $s$ positions each with $c$ input channels and $c'$ output channels).
- The **effective receptive field** grows with both $s$ and $r$.
- Stacking layers with exponentially increasing dilation rates ($1, 2, 4, 8, \dots$) makes the receptive field grow **exponentially** with depth.

This idea was popularized by **WaveNet** for audio generation:
- Many dilated causal convolutions are stacked to provide a very large receptive field without a huge parameter count.

---

## 8.3 Forecasting and Causal Models

### 8.3.1 Forecasting Sequences

We often want to predict **future elements** of a sequence:

- Examples: energy prices, turbulence flows, call center loads, future tokens in a text.
- Forecasting is central to large language models (LLMs), which typically predict the next token given previous ones.

**Self-supervised formulation**:

- We only need unlabeled sequences; the targets are derived from the sequences themselves.

#### Stationarity (brief remark)

Real-world time series may have trends, seasonality, and non-stationary behavior. In many practical machine learning setups:

- Audio, text, and similar sequences are treated as approximately stationary for modeling.
- For very large datasets and models, detailed statistical preprocessing often becomes less critical.

#### Supervised forecasting setup

Fix a **context length** (sequence length) $t$.

From a dataset of sequences, extract many length-$t$ subsequences:
- Each subsequence:
  $$
  X \in \mathbb{R}^{t \times c}
  $$
- Target: the **next element**:
  $$
  y \in \mathbb{R}^{c}
  $$
  (or a discrete token).

We obtain training pairs $(X, y)$ and train a model:
$$
f(X) \approx y.
$$

- For continuous outputs: minimize MSE.
- For discrete tokens: minimize cross-entropy.

#### Autoregressive generation

While training uses **single-step** prediction, at inference we can generate multiple steps by **feeding predictions back into the input** (autoregression).

Given an observed window:
$$
X = 
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_t
\end{bmatrix},
$$
the model predicts:
$$
\hat y = f(X).
$$

To predict further:

1. Construct a **shifted window** with the prediction:
   $$
   X' = 
   \begin{bmatrix}
   x_2 \\
   x_3 \\
   \vdots \\
   x_t \\
   \hat y
   \end{bmatrix}.
   $$
2. Compute the next prediction $\hat y' = f(X')$.
3. Repeat as needed.

For **discrete sequences** (e.g. tokens), $f(X)$ outputs a probability distribution over tokens; we can:
- Take the **argmax** (greedy),
- Or **sample** according to the probabilities.

**Limitation**: this setup relies on a **fixed context length** $t$; the model only sees the most recent $t$ elements.

---

### 8.3.2 Causal Models

Autoregressive forecasting with a fixed sequence length can be generalized with **causal layers** and **causal convolutions**.

#### **Definition (Causal layer)**

A layer $H = f(X)$ with sequence input $X = (x_1, \dots, x_t)$ is **causal** if each output $H_i$ depends only on the **past and present**, not on the future:

$$
H_i = f(x_{1:i}), \quad i = 1, \dots, t.
$$

A model composed exclusively of causal layers is causal as a whole.

- A 1D convolution with kernel size $1$ is inherently causal.
- A standard convolution with kernel size $3$ (one step to the left and right) is **not** causal, because it uses $x_{i+1}$ (future) when computing $H_i$.

#### Causal convolution via masking

We can convert a standard convolution into a causal one by **masking weights** corresponding to future inputs.

Recall:
$$
h_i = \phi\bigl(W \, \mathrm{vec}(P_k^{(i)}) + b\bigr).
$$

Introduce a mask matrix $M$ with the same shape as $W$ and define:
$$
h_i = \phi\bigl((W \odot M) \, \mathrm{vec}(P_k^{(i)}) + b\bigr),
$$
where:

- $\odot$ is elementwise multiplication.
- $M_{j} = 0$ if the weight connects to any input position strictly **after** $i$, and $1$ otherwise.

In the simplest single-channel case, $M$ is a **lower-triangular** matrix, enforcing that each output depends only on current and past positions.

- This mask effectively reduces parameters from $(2k+1)cc'$ to $(k+1)cc'$.
- Combining **causal** and **dilated** convolutions yields efficient autoregressive models (e.g. WaveNet).

#### Causal models and multi-step prediction

Stack several causal convolutional layers with no temporal pooling. Then:
$$
\hat Y = f_{\text{causal}}(X),
$$
where both $X$ and $\hat Y$ have length $t$, and:
- $\hat Y_i$ depends only on $x_{1:i}$.

We can define a target sequence:
$$
Y = 
\begin{bmatrix}
x_2 \\
x_3 \\
\vdots \\
x_{t} \\
y_{t+1}
\end{bmatrix},
$$
i.e. at each position we want the model to predict the **next** element.

For continuous outputs, a natural loss is:
$$
\ell(\hat Y, Y) = \sum_{i=1}^{t} \|\hat Y_i - Y_i\|_2^2.
$$

**Advantages**:

- Each training window yields **$t$ loss terms** instead of just one.
- This improves gradient flow and training efficiency.
- The model learns to forecast from all possible prefix lengths up to $t$.

#### Autoregressive inference with causal CNNs

Suppose $t = 4$, and we have observed only two elements $x_1, x_2$:

1. Build a padded input (e.g., zeros for missing past):
   $$
   X^{(1)} = 
   \begin{bmatrix}
   x_1 \\
   x_2 \\
   0 \\
   0
   \end{bmatrix}.
   $$
2. Run $\hat Y^{(1)} = f_{\text{causal}}(X^{(1)})$. Extract the prediction for the **next** element (here, at position 3) as $\hat x_3$.
3. Form the next input:
   $$
   X^{(2)} = 
   \begin{bmatrix}
   x_1 \\
   x_2 \\
   \hat x_3 \\
   0
   \end{bmatrix},
   $$
   run $f_{\text{causal}}$ again to get $\hat x_4$, and so on.
4. Once the buffer is full, we keep a **sliding window** of length $t$, dropping the oldest element as we generate further steps.

This is an **autoregressive** use of a causal CNN that can handle any prefix length up to the trained maximum.

#### Teacher forcing

During training, the model is always fed **true** past values, not its own predictions. This is called **teacher forcing**.

- Pros: training is stable and efficient.
- Cons: at inference it must operate on its own (possibly imperfect) predictions, which may lead to exposure bias.

Variants gradually replace some training tokens with model predictions to reduce this mismatch.

#### Text generation as a special case

For text:

- We often start from a single `[BOS]` token.
- Autoregressively generate tokens until an `[EOS]` token or a maximum length is reached.
- To condition on a user prompt, we start from `[BOS] + \text{prompt tokens}` and generate the continuation.

The same causal mechanisms used for time series thus underpin modern text generation.

---

## 8.4 Generative Models

### 8.4.1 A Probabilistic Formulation

An autoregressive model is a special case of a **generative model**.

In supervised learning, data pairs $(x, y)$ are assumed to come from some unknown joint distribution $p(x, y)$. A model that approximates:

- $p(y \mid x)$ is called **discriminative**,
- $p(x)$ or $p(x \mid y)$ is **generative**, because we can sample new $x$’s.

Here, we focus on sequences $x = (x_1, \dots, x_T)$ from some space $\mathcal{X}$ (e.g., all token sequences). Many sequences are nonsensical, but some may occur in large corpora.

A **language model** is a probability distribution:
$$
p(x), \quad x \in \mathcal{X}.
$$

Our goal is to learn to **sample** from $p(x)$:
$$
x \sim p(x).
$$

Using the **chain rule of probability**, we can factorize:
$$
p(x) = p(x_1, \dots, x_T) = \prod_{i=1}^{T} p(x_i \mid x_{1:i-1}),
$$
where $x_{1:0}$ is taken to be the empty context.

A **causal model** $f$ can parameterize each conditional:
$$
p(x_i \mid x_{1:i-1}) \approx \mathrm{Categorical}(x_i \mid f(x_{1:i-1})),
$$
where $f(x_{1:i-1})$ outputs logits for a categorical distribution over tokens.

Training by **maximum likelihood** of $p(x)$ is equivalent to minimizing the **cross-entropy loss** over predicted token distributions at each position.

---

### 8.4.2 Sampling in an Autoregressive Model

Sampling from a generic high-dimensional distribution is difficult, but the factorization
$$
p(x) = \prod_{i=1}^{T} p(x_i \mid x_{1:i-1})
$$
enables a straightforward **ancestral sampling** procedure:

1. Sample $x_1 \sim p(x_1)$.  
   In practice, we usually **fix** $x_1$ to a special token like `[BOS]`.

2. For $i = 2$ to $T$:
   - Sample
     $$
     x_i \sim p(x_i \mid x_{1:i-1}),
     $$
     using the model’s predicted distribution.

We already encountered a deterministic variant:

- **Greedy decoding**:
  $$
  x_i = \arg\max_j f_j(x_{1:i-1}),
  $$
  where $f_j$ is the logit/probability for token $j$.

#### Temperature and randomness

Let $z(x_{1:i-1})$ be the logits output by the model. We form probabilities via a **softmax** with **temperature** $\tau$:
$$
p_\tau(j \mid x_{1:i-1}) = \frac{\exp(z_j / \tau)}{\sum_k \exp(z_k / \tau)}.
$$

- $\tau \to 0$: distribution becomes peaked; sampling approximates greedy argmax.
- Larger $\tau$: more uniform distribution; more diversity, but also more risk of incoherence.

#### Local vs global probability

The probability of a full sequence is:
$$
p(x) = \prod_{i=1}^{T} p(x_i \mid x_{1:i-1}).
$$

Greedy decisions are **local**; they maximize each conditional probability in isolation. This does **not** imply that the resulting sequence has high **global** probability $p(x)$:

- An early token choice that is locally likely may funnel the process into a region of sequence space where subsequent continuations are low probability.
- This motivates more advanced decoding methods.

---

### 8.4.3 Beam Search (Beam Decoding)

**Beam search** is a heuristic decoding strategy to approximate higher-probability sequences.

- Maintain a set of $k$ **beams** (partial sequences).

Algorithm sketch:

1. **Initialization**:
   - From the start state (e.g. `[BOS]`), compute probabilities over next tokens.
   - Keep the top $k$ tokens as initial beams.

2. **Expansion**:
   - For each beam, expand it by all possible next tokens (or a large subset).
   - Evaluate the joint probability (product of per-step probabilities, or sum of log-probabilities).

3. **Pruning**:
   - Among all expanded candidates (up to $k \times V$, where $V$ is vocab size), keep only the top $k$ sequences.

4. **Repeat** until:
   - All beams have emitted an `[EOS]` token, or
   - A maximum length is reached.

Interpretation:

- Beam search approximates a **breadth-first search** on a tree where each node is a partial sequence and each edge adds one token.
- Beam width $k$ controls the tradeoff between:
  - Computational cost,
  - Quality of sequences (in terms of probability and perceived coherence).

Additional techniques can incorporate **external rewards** or constraints, biasing decoding toward sequences that satisfy extra objectives beyond pure likelihood.

---

### 8.4.4 Conditional Modeling

Often we want to generate **conditioned** sequences, e.g., responses conditioned on a user prompt, captions conditioned on an image, etc.

We consider a conditional distribution:
$$
p(x \mid c),
$$
where $c$ is the **conditioning context** (prompt, image, metadata, etc.).

By the chain rule:
$$
p(x \mid c) = \prod_{i=1}^{T} p(x_i \mid x_{1:i-1}, c).
$$

We parameterize:
$$
p(x_i \mid x_{1:i-1}, c) \approx \mathrm{Categorical} \big(x_i \mid f(x_{1:i-1}, c)\big).
$$

For **text-only** conditioning, a simple and powerful trick is to **concatenate**:
- Let $c$ be a sequence of prompt tokens.
- Let $x$ be the sequence to be generated.
- Form $x' = [c \,\Vert\, x]$ and train a single **causal** model on full sequences.

At inference:

- Feed `$[BOS], c$` to the model,
- Generate the continuation tokens $x$ autoregressively.

Example (schematic):

- Input prompt tokens: `[The, capital, of, France]`.
- Model outputs:
  - Next token “is”,
  - Then “Paris”,
  - Then `[EOS]`, etc.

**Multimodal extension**:

- Convert other modalities (e.g., images) to sequences of tokens or embeddings of the same dimension as text tokens.
- Concatenate “conditioning” tokens (e.g., image tokens) with text tokens.
- Apply the **same** autoregressive model on the joint sequence:
  - The initial segment acts as the conditioning $c$,
  - The rest as the generated $x$.

This provides a unified view of conditional generative modeling across modalities.

---

## 8.5 Practical Notes and Tools

Working with text data is more involved than image classification, due to:

- Tokenization choices and vocabulary design.
- Handling variable-length sequences.
- Special characters and encoding issues.
- Formatting datasets into suitable tensor shapes.

### 8.5.1 Libraries and datasets

- **PyTorch**:
  - Has its own text library `torchtext` and pipeline utilities (via `torchdata`).
  - These can handle datasets and tokenization, but may lag main PyTorch in documentation and stability.

- **Hugging Face Datasets**:
  - Provides many standard text datasets (e.g., IMDB movie reviews).
  - Comes with pre-trained tokenizers.
  - Integrates smoothly with PyTorch.

### 8.5.2 Example workflow: Text classification with a 1D CNN

1. Choose a dataset (e.g., IMDB for sentiment classification).
2. Tokenize texts $\to$ sequences of token IDs (e.g., using a pretrained subword tokenizer).
3. Build a dataset of pairs $(x, y)$:
   - $x$: list of token IDs,
   - $y$: label (e.g., positive/negative).
4. Use padding (and masks if needed) to form mini-batches.
5. Define a model similar to the TextCNN structure:
   - Embedding layer with trainable matrix $E$,
   - One or more `Conv1d` layers with non-linearities,
   - Temporal pooling layers (e.g., max or average pooling),
   - A linear head for classification.
6. Train with cross-entropy loss; experiment with:
   - Number of layers,
   - Number of channels,
   - Kernel sizes,
   - Pooling strategies.

### 8.5.3 Causal convolutions in practice

- Many frameworks lack built-in **causal Conv1D** layers.
- Causality can be emulated by:
  - Carefully designed padding and slicing,
  - Or manually applying a weight mask on convolution kernels.
- These techniques are important when implementing autoregressive CNNs before moving to transformer-based architectures.

---

**Summary of Key Concepts**

- CNNs generalize naturally to 1D (time series, audio, text) and 3D (video, volumetric) data by sliding kernels along temporal or spatiotemporal axes.
- Variable-length sequences are handled using padding and masking; this pattern is fundamental across modern sequence models.
- Text processing with CNNs relies on tokenization and learned embeddings, transforming discrete symbols into dense vectors.
- Dilated convolutions allow exponentially large receptive fields with modest parameter counts, particularly valuable for long sequences.
- Forecasting and autoregressive generation are self-supervised tasks where models predict the next token given a context window.
- Causal convolutions enforce dependence only on past and present, enabling flexible autoregressive inference and richer training signals.
- Autoregressive models define a generative distribution via the probability chain rule and support various decoding strategies (greedy, temperature sampling, beam search).
- Conditional autoregressive models unify unconditional generation, prompt-based interaction, and multimodal generation within a single framework.