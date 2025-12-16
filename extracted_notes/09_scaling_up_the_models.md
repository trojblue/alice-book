# Chapter 9 – Scaling up the models

[TOC]

---

## 0. High-level summary

- Scaling convolutional networks to **dozens or hundreds of layers** requires stabilizing training.
- The **ImageNet challenge (2010–2017)** drove rapid progress: from kernel methods to deep CNNs like **AlexNet** and then very deep **ResNets**.
- General training strategies:
  - **Weight regularization** ($\ell_2$, $\ell_1$, structured sparsity).
  - **Early stopping** using a validation set.
  - **Data augmentation** (noise, geometric transforms, mixup, cutmix).
- Deep-learning-specific techniques:
  - **Dropout**: stochastic removal of units during training.
  - **Normalization layers**: especially **batch norm**, **layer norm**, and **RMSNorm**.
  - **Residual connections**: make very deep networks trainable; connect to ensembles and ODEs.
- These ideas remain central today, not only for CNNs but also for architectures like **transformers**.

---

## 1. Motivation: Scaling up the models

- Convolutional models have a **receptive field that grows linearly with depth**.
- To capture large-scale structure in images, we need **deep architectures** (tens or hundreds of layers).
- But naive deepening leads to:
  - Slow or stuck optimization.
  - Gradient explosion/vanishing.
  - Numerical instability.
- The chapter presents **practical techniques** that make deep training feasible:
  - Classical ML ideas (regularization, data augmentation, early stopping).
  - Deep-specific layers (dropout, normalization, residual connections).

---

## 2. The ImageNet challenge (9.1)

### 2.1 Problem and setup

- **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)**:
  - Annual competition from 2010–2017.
  - Subset of ImageNet with roughly **$10^6$ images** and **1000 classes**.
  - Standard benchmark for **image classification**; also related tasks (detection, localization in later years).
  - Performance commonly reported with **top-5 error** and **top-1 accuracy**.

### 2.2 Pre-deep-learning and AlexNet

- **2010–2011 winners**:
  - Non-deep methods: **linear-kernel classifiers** on top of **hand-crafted feature descriptors**.
  - Top-5 error around **28% (2010)** and **26% (2011)**.
  - Convolutional models trained via gradient descent existed but were **niche** and not competitive.

- **2012 breakthrough – AlexNet**:
  - AlexNet (Krizhevsky, Sutskever, Hinton, 2012):
    - Achieved **15.3% top-5 error**, about **10 percentage points better** than the best non-neural competitors.
    - Architecture: **5 convolutional layers + 3 fully connected layers**, roughly **60M parameters**.
  - Marked a **“Copernican revolution”** in computer vision:
    - After 2012, almost all top entries used **deep convolutional networks**.

### 2.3 Scaling law flavor

- Over a few years, models:
  - Became **much deeper** (up to hundreds of layers).
  - Achieved **>95% top-1 accuracy** on ImageNet, effectively saturating the benchmark.
- This illustrates a **scaling law**:
  - **More parameters + more compute** → better accuracy,
  - up to a **saturation point determined by the dataset**.
- However, simply “making it bigger” is **non-trivial**:
  - Deep CNNs need **careful training strategies** (regularization, normalization, residual connections) to be optimizable at all.

---

## 3. Data and training strategies (9.2)

### 3.1 Weight regularization (9.2.1)

#### 3.1.1 Regularized loss and MAP view

- Let:
  - $w$ = vector of all model parameters.
  - $\mathcal{D}_n$ = dataset (e.g. $n$ examples).
  - $L(w,\mathcal{D}_n)$ = base loss (e.g. average cross-entropy).
- Introduce a **regularization functional** $R(w)$ encoding prior preferences (e.g. small weights):
  $$
  L_{\text{reg}}(w) = L(w,\mathcal{D}_n) + \lambda R(w),
  $$
  where $\lambda \ge 0$ balances data fit and regularization.
  - $\lambda = 0$: no regularization.
  - $\lambda \to \infty$: dominated by prior preference $R(w)$.

- **Bayesian / MAP interpretation**:
  - Assume prior $p(w)$ and likelihood $p(\mathcal{D}_n\mid w)$.
  - Maximum-a-posteriori (MAP) estimate:
    $$
    w^\* = \arg\max_w \Bigl\{ \log p(\mathcal{D}_n\mid w) + \log p(w) \Bigr\}.
    $$
  - Regularization corresponds to a **non-uniform prior**:
    - $R(w)$ is $- \log p(w)$ up to scaling.

#### 3.1.2 $\ell_2$ regularization and weight decay

- **$\ell_2$ penalty**:
  $$
  R(w) = \|w\|_2^2 = \sum_i w_i^2.
  $$
- Encourages **small-magnitude weights**, meaning:
  - Output changes less sharply for small input changes.
  - Equivalent (in MAP view) to a **Gaussian prior** with covariance $\sigma^2 I$.

- Gradient of regularized loss:
  $$
  \nabla L_{\text{reg}}(w) = \nabla L(w,\mathcal{D}_n) + 2\lambda w.
  $$

- For simple (stochastic) gradient descent (SGD), adding this term is equivalent to **weight decay**: weights are shrunk toward zero every update.

- For more complex optimizers, we distinguish:

  - Let $g(\nabla L(w,\mathcal{D}_n))$ be the **post-processed gradient** (e.g., after momentum, Adam statistics).
  - **Decoupled weight decay (AdamW-style)** update:
    $$
    w_t = w_{t-1}
          - g\bigl(\nabla L(w_{t-1},\mathcal{D}_n)\bigr)
          - \lambda w_{t-1}.
    $$
  - This is **not** identical to pure $\ell_2$ regularization (where $2\lambda w$ would be inside $g(\cdot)$), but works better in practice for optimizers like Adam.

#### 3.1.3 $\ell_1$ and structured sparsity

- **$\ell_1$ regularization**:
  $$
  R(w) = \|w\|_1 = \sum_i |w_i|
  $$
  - Promotes **sparse** solutions with many exact zeros.
  - Corresponds to a **Laplace prior** on weights.
- Can be extended to **group sparsity**:
  - Penalties on groups of parameters (e.g., per-neuron or per-channel), encouraging entire groups to be zero.
  - Useful for **structured sparsity** and hardware efficiency.

- In deep nets, plain $\ell_1$ can interact poorly with strong **non-convexity** and gradient-based optimization.
  - One workaround: **reparameterize**:
    - Introduce $a,b$ with same shape as $w$, and set
      $$
      w = a \odot b, \qquad
      \|w\|_1 \approx \|a\|_2^2 + \|b\|_2^2.
      $$
    - This turns a sparsity goal into a (more optimization-friendly) sum of $\ell_2$ penalties on $a$ and $b$, at the cost of extra parameters.

#### 3.1.4 Constrained view and geometry

- Regularized problem with convex loss (e.g., least squares) is equivalent to a **constrained optimization**:
  $$
  \min_w L(w,\mathcal{D}_n)
  \quad \text{subject to } R(w)\le \mu.
  $$
- Relationship:
  - Different $\lambda$ correspond to different constraint radii $\mu$ via Lagrange multipliers.

- **Geometric intuition**:
  - $\ell_2$ constraint $R(w)=\|w\|_2^2 \le \mu$:
    - Solution lies inside a **sphere** (ball) centered at the origin.
  - $\ell_1$ constraint $\|w\|_1 \le \mu$:
    - Feasible region is a **polytope** (e.g., a diamond in 2D).
    - Optima often at **vertices** that lie on coordinate axes → sparse solutions.

---

### 3.2 Early stopping (9.2.2)

#### 3.2.1 Optimization vs generalization

- Pure optimization goal: find $w_t$ such that gradient nearly vanishes:
  $$
  \|\nabla L(w_t)\| \approx 0
  $$
  or equivalently
  $$
  \|L(w_t) - L(w_{t-1})\|^2 \le \varepsilon
  $$
  for small $\varepsilon > 0$.
- But **lowest training loss** does **not** guarantee best generalization, especially with limited data:
  - Over-parameterized models can severely **overfit** if trained too long.

#### 3.2.2 Early stopping via validation set

- Use a **validation dataset** separate from training and test.
- After each epoch $t$:
  - Compute some metric on validation set, e.g. accuracy, F1-score; denote it by $a_t$.
- Choose a **patience** hyperparameter $k$ (number of epochs to wait).

- **Rule** (conceptual):
  - If the most recent validation score is **no better** than any of the last $k$ scores:
    $$
    a_t \le a_i \quad \forall i \in \{t-1,\dots,t-k\},
    $$
    then **stop training** (we assume we are in overfitting regime).
- If checkpointing is used, we can **roll back** to the weights from the best epoch (e.g. epoch $t-k$ or the argmax of $a_i$ so far).

- Interpretation:
  - Early stopping is a form of **model selection** where the “model” dimension is **number of epochs**.
  - We can select based on any metric, including **non-differentiable** ones (F1, accuracy, etc.).

#### 3.2.3 When early stopping helps

- Very helpful:
  - **Small datasets** or low data regimes.
  - Models that can overfit dramatically.
- Less straightforward for **large, over-parameterized models**:
  - Validation error can show **multiple descent** phases (non-monotone behavior: down–up–down).
  - Validation loss can remain flat for a long time then suddenly drop.
- In such settings, early stopping may prematurely stop training; its benefit is more nuanced.

---

### 3.3 Data augmentation (9.2.3)

#### 3.3.1 Motivation

- Strongest lever on performance: **more data**.
- Obstacles:
  - Labeling is expensive and time-consuming.
  - Synthetic data from generative models requires complicated pipelines.
- **Data augmentation**: artificially enlarge dataset by applying **label-preserving transformations** to inputs.

#### 3.3.2 Vector data: additive noise

- For generic vector input $x$, consider:
  $$
  x' = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I).
  $$
- Repeatedly sampling noise around $x$ yields a (practically infinite) cloud of nearby points.
- This makes the model more **robust** and is mathematically related to **$\ell_2$ regularization**.
- Caveat:
  - For **unstructured vector data**, large noise can generate **invalid points** (outside the data manifold).

#### 3.3.3 Image data: structured transformations

- Images admit many **semantic-preserving transforms**:
  - Small rotations, translations, zooms.
  - Brightness/contrast changes.
  - Flips, color jitter, etc.

- Let $T(x; c)$ be a transformation parameterized by $c$ (e.g., rotation angle), and $p(c)$ a distribution over allowed parameters:
  $$
  x' = T(x;c),\quad c\sim p(c).
  $$
- Most transforms include the **identity** as special case (e.g., rotation angle $0^\circ$).

- Training setup:
  - Each epoch, every training example is seen once, but **each time with a newly sampled transform**.
  - This yields a **virtually unbounded** stream of distinct examples without storing them all.

- Design questions:
  - Which transforms to allow?
  - What ranges for parameters?
  - How to compose multiple transforms?
  - Must ensure transforms preserve label (e.g., horizontal flip may break text recognition).

- **RandAugment**:
  - A practical strategy: define a large pool of possible transforms.
  - For each mini-batch, randomly sample a small number (e.g., 2–3) and apply them sequentially with a chosen magnitude.

- Implementation detail:
  - In PyTorch, augmentations are often specified via `torchvision.transforms` or `transforms.v2` and applied in data loaders.
  - In other frameworks (TensorFlow/Keras), augmentation is sometimes implemented as **model layers**.

#### 3.3.4 Mixup and cutmix

**Mixup (for vectors)**

- Given two examples $(x_1, y_1)$ and $(x_2, y_2)$:
  - Sample $\lambda \in [0,1]$.
  - Create a **mixed** example:
    $$
    x = \lambda x_1 + (1-\lambda)x_2,
    $$
    $$
    y = \lambda y_1 + (1-\lambda)y_2.
    $$
- Effect:
  - Encourages **linear behavior** of the model between training data points.
  - Geometric view: we move along the line segment between $x_1$ and $x_2$ on (an approximation of) the data manifold.

**Cutmix (for images)**

- Simple pixel-wise mixing of images produces **blurry** and unrealistic images.
- Cutmix instead:
  - Sample a **patch** (e.g., $32\times 32$) from image $x_1$.
  - Let $M$ be a binary mask: $M=1$ inside the patch, $0$ outside.
  - Combine:
    $$
    x = M\odot x_1 + (1-M)\odot x_2.
    $$
  - Use **label interpolation** as in mixup:
    $$
    y = \lambda y_1 + (1-\lambda)y_2.
    $$
- Intuition:
  - Model must learn to focus on **patch-level evidence** and be robust to partial occlusion.
  - Often combined with standard geometric augmentations (e.g., rotations).

---

## 4. Dropout and normalization (9.3)

### 4.1 Dropout as regularization (9.3.1)

#### 4.1.1 Core idea

- Data augmentation made training harder at the **input level**, improving robustness.
- **Dropout** applies this idea to **internal activations**:
  - Randomly **zero out** some units during training.
  - Forces network to distribute information and avoid reliance on any single feature.
  - Acts as a form of regularization.

#### 4.1.2 Definition (fully-connected case)

- Consider a mini-batch of activations:
  - $X \in \mathbb{R}^{n\times c}$ (batch size $n$, features $c$).
- Sample a **binary mask**:
  $$
  M \sim \text{Binary}(n, c), \quad M_{ij} \sim \text{Bern}(p),
  $$
  where $p$ is the **keep probability** (hyperparameter).
- Dropout transform:
  $$
  \text{Dropout}(X) = M \odot X.
  $$
- Properties:
  - Approximately $(1-p)\cdot 100\%$ of units per example are zeroed.
  - No trainable parameters; only hyperparameter $p$ (often $0.5$ for dense layers, higher for others).

- Dropout layers can be inserted:
  - On **inputs**.
  - After fully connected / embedding layers.
  - On attention maps or outputs (common in transformers).

#### 4.1.3 Stochastic training vs deterministic inference

Let there be $m$ dropout layers:

- Let $M = (M_1,\dots,M_m)$ collective mask, with $M_i$ for layer $i$.
- Let $p(M)$ be the joint distribution of these masks.
- Define deterministic network output given masks as $f(x;M)$.

- **Training**:
  - Sample fresh $M \sim p(M)$ each forward pass.
  - Output: random variable $f(x;M)$.

- **Inference objective**:
  - We would ideally compute the **expectation**:
    $$
    f(x) = \mathbb{E}_{p(M)}[f(x;M)].
    $$

- **Monte Carlo dropout**:
  - Approximate expectation by averaging $k$ independent mask samples:
    $$
    \mathbb{E}_{p(M)}[f(x;M)] \approx \frac{1}{k} \sum_{i=1}^k f(x; Z_i),
    \quad Z_i \sim p(M).
    $$
  - Pros:
    - Can estimate **uncertainty** (variance across passes).
  - Cons:
    - Requires multiple forward passes; computationally expensive.

#### 4.1.4 Inverted dropout (standard implementation)

- Compute per element:
  $$
  \mathbb{E}[M_{ij} X_{ij}] = p X_{ij}.
  $$
- To avoid rescaling at inference, use **inverted dropout**:

  - **Training**:
    $$
    \text{Dropout}(X) = \frac{M\odot X}{p}.
    $$
    So $\mathbb{E}[\text{Dropout}(X)] = X$.

  - **Inference**:
    $$
    \text{Dropout}(X) = X.
    $$

- In this scheme, dropout layers are **no-ops at inference** and can conceptually be removed.

- Framework detail:
  - Libraries distinguish training vs inference mode (e.g. `model.train()` vs `model.eval()` in PyTorch), so dropout layers behave accordingly.
  - Monte Carlo dropout at test-time often uses **training mode** but averages predictions.

#### 4.1.5 Variants and when to use dropout

- Very effective for:
  - Fully connected layers in classifiers.
  - Attention weights/outputs in transformer models.

- Less effective for standard convolutional feature maps, where random sparsity may harm structure.
  - **Spatial dropout**: drop entire channels.
  - **Cutout**: drop rectangular regions in the spatial dimensions.

- **DropConnect**:
  - Instead of masking activations, mask weights:
    $$
    \text{DropConnect}(x) = (M \odot W)x + b.
    $$
  - Inference can be approximated analytically via moment matching.
  - Less commonly used than dropout, BN, and residual connections.

---

### 4.2 Batch and layer normalization (9.3.2)

#### 4.2.1 Motivation

- Classical preprocessing for tabular data:
  - **Standardization** to zero mean, unit variance per feature.
  - Or **min–max scaling** to fixed ranges.
- For deep networks, we want similar control over **intermediate activations**:
  - Well-scaled inputs to each layer improve optimization.
  - But hidden activations change each step due to weight updates.

- **Batch normalization (BN)** approximates these statistics **using the mini-batch** instead of the full dataset.

#### 4.2.2 Batch normalization for fully-connected layers

- For activations $X\in\mathbb{R}^{n\times c}$:

  - **Mini-batch mean** (per feature $j$):
    $$
    \mu_j = \frac{1}{n}\sum_{i=1}^n X_{ij}.
    $$

  - **Mini-batch variance**:
    $$
    \sigma_j^2 = \frac{1}{n}\sum_{i=1}^n (X_{ij} - \mu_j)^2.
    $$

- **Normalize**:
  $$
  \tilde{X} = \frac{X - \mu}{\sqrt{\sigma^2 + \varepsilon}},
  $$
  with $\varepsilon > 0$ to avoid division by zero.

- **Affine transform** with learnable parameters $\alpha,\beta \in \mathbb{R}^c$:
  $$
  \text{BN}(X) = \alpha \tilde{X} + \beta.
  $$

- **Definition (BN layer)**:
  - Input: $X \in \mathbb{R}^{n\times c}$.
  - Compute $\mu,\sigma^2$ over batch.
  - Output $\text{BN}(X)$ as above.
  - Learnable parameters: $\alpha,\beta$.
  - Hyperparameters: (effectively) $\varepsilon$; otherwise none.

- Typical placement:
  $$
  H = (\text{ReLU} \circ \text{BN} \circ \text{Linear})(X).
  $$
  - The bias in `Linear` is redundant with $\beta$ and can be removed.

- Empirical effects:
  - Stabilizes gradient magnitudes.
  - Allows **larger learning rates**.
  - Eases training of very deep architectures.

#### 4.2.3 Batch norm for images

- For image tensors $X\in\mathbb{R}^{n\times h\times w\times c}$:
  - **Channel-wise BN** aggregates over batch and spatial dimensions:
    $$
    \mu_z = \frac{1}{nhw}\sum_{i,j,k} X_{ijkz},
    $$
    with analogous variance.
- BN is then applied per channel, with $\alpha,\beta$ also per channel.

#### 4.2.4 Batch norm at inference time

Issue: during training, the output depends on **which samples are in the current mini-batch**; at test time, predictions should not depend on other test samples.

Two main strategies:

1. **Full-dataset statistics (conceptual)**:
   - After training, run a forward pass over the whole training set.
   - Compute **true dataset means/variances** per feature.
   - Use these fixed statistics for inference.

2. **Running statistics (practical)**:
   - Maintain **exponential moving averages** of $\mu$ and $\sigma^2$ during training:
     $$
     \hat{\mu} \leftarrow \lambda \hat{\mu} + (1-\lambda)\mu, \quad 0<\lambda<1,
     $$
     and similarly for $\hat{\sigma}^2$.
   - After training, use $\hat{\mu},\hat{\sigma}^2$ as the fixed statistics.
   - In frameworks like PyTorch, these are stored as non-trainable **buffers**.

In inference mode, BN uses these precomputed statistics instead of recalculating them from the current batch.

#### 4.2.5 Limitations of BN and variants

- BN drawbacks:
  - **Small batch sizes** → noisy estimates, instability.
  - **Distributed training**: each device sees different batch subset.
  - **Train–test mismatch**: train-time mini-batch stats vs fixed stats at test-time.

To mitigate, several alternatives based on changing **normalization axes**:

**Layer normalization (LN)**

- For $X\in\mathbb{R}^{n\times c}$:

  - Mean per example $i$:
    $$
    \mu_i = \frac{1}{c}\sum_{j=1}^c X_{ij},
    $$
  - Variance per example $i$:
    $$
    \sigma_i^2 = \frac{1}{c}\sum_{j=1}^c (X_{ij}-\mu_i)^2.
    $$

- Normalize each row (each sample) and apply per-feature affine transform.
- **Crucially**:
  - LN does **not** depend on batch statistics → works well for small or variable batch sizes.
  - Widely used in **transformers** (often per token).

**For images**, $X\in\mathbb{R}^{b\times h\times w\times c}$, typical LN choices:

- **Variant A**: normalize across $(h,w,c)$ jointly per example.
- **Variant B**: normalize across channels $c$ for each spatial location $(h,w)$ independently (common in transformer-style architectures for images treated as patches/tokens).

**Group normalization & instance normalization**

- **Group normalization**:
  - Split channels into groups and normalize within each group.
  - Trades off between BN and LN; useful when batch sizes are very small.

- **Instance normalization**:
  - Special case where each channel of each example is normalized separately.

**RMSNorm**

- Simplified layer normalization that removes explicit mean-centering and shift:
  - For $x\in\mathbb{R}^c$:
    $$
    \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{c}\sum_i x_i^2}}\odot \alpha,
    $$
    with learnable $\alpha$.
- If $x$ is already zero-centered and $\beta=0$, RMSNorm and LN coincide.

---

## 5. Residual connections (9.4)

### 5.1 Residual connections and residual networks (9.4.1)

#### 5.1.1 The degradation problem

- Consider two models:
  - Shallow: $g_1(x) = (f_3 \circ f_1)(x)$.
  - Deeper: $g_2(x) = (f_3 \circ f_2 \circ f_1)(x)$.
- In principle, $f_2$ could learn the **identity** $f_2(x)\approx x$, so $g_2$ should be at least as good as $g_1$.
- In practice, deeper plain networks often show **higher training error** than shallower ones:
  - Optimization becomes harder.
  - Gradients vanish or explode.

#### 5.1.2 Residual block definition

- Idea: **bias** each block toward the identity via a **skip connection**.
- Given a transformation $f(x)$ with input and output same shape, define:
  $$
  r(x) = f(x) + x.
  $$
- Interpretations:
  - $f(x)$ learns **residual** deviations from identity.
  - If $f(x) = 0$, the block is exactly the identity, so deeper models can easily collapse to shallower ones.

- A network composed mainly of such blocks is a **residual network (ResNet)**.

- If input/output shapes differ (e.g., changing number of channels), we adapt the skip path:
  - For images, common choice:
    $$
    r(x) = f(x) + \text{Conv2D}_{1\times 1}(x),
    $$
    where the $1\times 1$ conv matches channel dimensions.

#### 5.1.3 Gradient flow view

- Consider the vector-Jacobian product (VJP) of the residual block for a gradient vector $v$:
  $$
  \mathrm{vjp}_r(v) = \mathrm{vjp}_f(v) + v^\top.
  $$
- Backward pass:
  - Standard gradient through $f$ plus an **identity contribution** along the skip path.
  - This helps preserve gradient magnitude over many layers, reducing vanishing/exploding gradient issues.

#### 5.1.4 Designing residual blocks

- Consider a canonical sequence of operations:
  $$
  h = (\text{ReLU}\circ \text{BN}\circ \text{Conv2D})(x).
  $$
  - Because $\text{ReLU} \ge 0$, such blocks **cannot reduce** activations below $x$; they can only increase or zero them.
  - Stacking these inside residual connections can bias the network toward **non-decreasing activations**, which is suboptimal.

- Original ResNet design instead:
  - Use **two Conv–BN–ReLU stages**, but **omit** the final nonlinearity:
    $$
    f(x) = (\text{BN} \circ \text{Conv2D} \circ \text{ReLU} \circ \text{BN} \circ \text{Conv2D})(x),
    $$
    and then
    $$
    h = f(x) + x.
    $$
  - This allows both positive and negative residual corrections.

- Typical CNN architecture:
  - A **stem**:
    - Non-residual early layers (e.g., large-kernel conv + pooling) to shrink spatial resolution.
  - Followed by many **residual blocks** of this kind.

#### 5.1.5 Evolution: ResNet vs ResNeXt-style blocks

- **Original ResNet block**:
  - 1×1 conv to **reduce channels**.
  - 3×3 conv to process features.
  - 1×1 conv to **increase channels back** to original.
  - This is a kind of **bottleneck block**.

- **ResNeXt and similar modern blocks**:
  - Use **depthwise or grouped convolutions** to increase receptive field with fewer parameters.
  - May **expand** channels inside the block (e.g., 3×–4×) and compress back with 1×1 conv.
  - Often switch from BN to LN and from ReLU to **GELU** in some modern designs.

---

### 5.2 Additional perspectives on residual connections (9.4.2)

#### 5.2.1 ResNets as ensembles of paths

- Consider two residual blocks:
  $$
  h_1 = f_1(x) + x,
  $$
  $$
  h_2 = f_2(h_1) + h_1.
  $$
- Expand $h_2$:
  $$
  h_2 = f_2(f_1(x) + x) + f_1(x) + x.
  $$
- Seen as a combination of multiple **paths**:
  - Direct identity path $x$.
  - One-block paths: $f_1(x)$, $f_2(x)$ (approximately).
  - Two-block path: $f_2(f_1(x))$.

- For many residual blocks:
  - Number of possible paths grows **exponentially** with depth.
  - ResNet behaves like an **ensemble of many shallower networks** sharing parameters.
  - Empirically, ResNets are robust to **removing or perturbing individual blocks**, consistent with this ensemble-like behavior.

#### 5.2.2 ResNets and neural ODEs

- Consider a continuous-time **differential equation**:
  $$
  \frac{\partial x_t}{\partial t} = f(x_t, t),
  $$
  with initial value $x_0$.
- Solution at time $T$:
  $$
  x_T = x_0 + \int_0^T f(x_t, t)\,dt.
  $$

- **Euler discretization** with step size $h$:
  $$
  x_t = x_{t-1} + h\,f(x_{t-1}, t).
  $$
- This is structurally a **residual update**:
  - Each step adds a function of current state to itself.
- A ResNet with shared residual function $f$ across layers can be interpreted as a **discrete-time approximation** of such an ODE.
- **Neural ODEs**:
  - Parameterize $f(x,t)$ with a neural network.
  - Use ODE solvers to integrate from $x_0$ to $x_T$.
  - Backpropagation is formulated in continuous time (e.g., adjoint methods).
  - There are deep links to **normalizing flows** and continuous-time generative models.

---

## 6. Practical considerations and exercises

- All main techniques in this chapter are **widely implemented**:
  - **Weight decay**: built into optimizers (SGD, AdamW, etc.).
  - **Data augmentation**: libraries like `torchvision.transforms`, Keras preprocessing layers.
  - **Dropout, BN, LN, RMSNorm**: standard layers in modern deep-learning frameworks.
  - **Residual blocks / ResNets / ResNeXt**: available as reference implementations in vision libraries.

- Practice directions:
  1. Implement a simple BN or dropout layer **from scratch** using basic tensor operations and check it against the library version.
  2. Take a small convolutional classifier (e.g., from an earlier chapter) and:
     - Increase depth,
     - Add normalization, dropout, and simple residual connections,
     - Observe training stability and final accuracy.
  3. Implement a canonical ResNet or ResNeXt block following its original paper description.
  4. Explore **fine-tuning**:
     - Load a pretrained model (e.g., ResNet-50 trained on ImageNet),
     - Replace the final classification head for a smaller dataset like CIFAR-10,
     - Train only the head or lightly fine-tune the full network.

These techniques together form the core toolkit for **scaling neural networks** while keeping training stable and generalization strong.