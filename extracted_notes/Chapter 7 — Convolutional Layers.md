# Chapter 7 — Convolutional Layers

[TOC]

- Convolutional layers are built from two ideas: **locality** (nearby pixels interact) and **parameter sharing** (same filter everywhere).
- Images are treated as tensors with spatial dimensions and a **channel/feature** dimension; fully-connected layers ignore this structure and become hugely overparameterized.
- **Local layers** restrict each output pixel to depend only on a small **patch**; **convolutional layers** further impose weight sharing, giving **translation equivariance**.
- Convolutions are linear maps with a special **Toeplitz** structure and can be seen as applying learned **FIR filters** over the image.
- Stacking convolutions grows the **receptive field**; combining them with pooling or strides reduces spatial resolution and moves from equivariance toward **translation invariance**.
- **Max-pooling**, **global pooling**, and **strided convolutions** are main tools to aggregate spatial information.
- Full CNNs are built as a **backbone** (Conv blocks + pooling) plus a **head** (MLP classifier or similar), trained end-to-end via backprop and cross-entropy.
- Practical variants like **1×1**, **depthwise**, **groupwise**, and **depthwise separable** convolutions drastically reduce parameters, crucial for efficient architectures.
- Modern frameworks (PyTorch, JAX/Equinox, torchvision) provide highly optimized implementations; custom implementations are mainly didactic (or for special long convolutions via FFT).

---

## 1. Motivation: Why Convolutional Layers?

### 1.1 Fully-connected layers and structured data

Fully-connected (dense) layers are a flexible default for **unstructured/tabular** data and multi-layer perceptrons (MLPs), but on such data they are often outperformed in practice by models like random forests or well-tuned SVMs.

For **structured data** (images, time series, audio, graphs, videos), the input has additional geometry or ordering that a plain MLP ignores. Convolutional layers are designed to exploit this extra structure.

---

### 1.2 Image tensors, channels, and features

An image is modeled as a rank-3 tensor
$$
X \sim (h, w, c),
$$
where

- $h$ = height,
- $w$ = width,
- $c$ = number of channels (e.g. $c = 1$ for grayscale, $c = 3$ for RGB, higher for hyperspectral images).

A mini-batch of images is a rank-4 tensor
$$
X \sim (b, h, w, c),
$$
where $b$ is the batch size.

- The spatial dimensions $(h, w)$ have a natural **geometry** (neighbors, distances).
- The channel dimension $c$ behaves like a **feature dimension per pixel** and is not inherently ordered (RGB vs GBR is a convention).

**Remark (Channels as features).**  
We reuse the “feature” notation for channels $c$ because each pixel is described by $c$ features that are updated in parallel by the model. A convolutional layer maps
$$
(h, w, c) \;\longrightarrow\; (h, w, c')
$$
i.e., it computes a new embedding of size $c'$ for each of the $h w$ pixels.

---

### 1.3 Fully-connected layers on images: flattening and its problems

To apply a fully-connected layer to an image, we must **vectorize** it. Let $\mathrm{vect}(X)$ denote flattening $X$ into a 1D vector (e.g. `x.reshape(-1)` in PyTorch). For a generic rank-$n$ tensor
$$
x \sim (i_1, i_2, \dots, i_n)
$$
we have
$$
\mathrm{vect}(x) \sim \bigl( \textstyle\prod_{j=1}^n i_j \bigr).
$$

A dense layer then applies
$$
h = \phi\bigl(W \cdot \mathrm{vect}(X)\bigr),
$$
where $\phi$ is a nonlinearity.

Two main issues:

1. **Loss of composability.**  
   - Input is an image $X \sim (h,w,c)$, output is a vector $h$ (no spatial structure).
   - To stack multiple such layers and stay in the image world, we must “unflatten”:
     $$
     H = \mathrm{unvect}\bigl(\phi(W \cdot \mathrm{vect}(X))\bigr),
     $$
     where $\mathrm{unvect}$ reshapes back to $(h, w, c')$, assuming the number of pixels is unchanged.

2. **Huge parameter count.**  
   - For an RGB image of shape $(1024, 1024, 3)$, with the same number of pixels and channels in output:
     - Input dimension: $1024 \cdot 1024 \cdot 3 = hwc$.
     - Output dimension: $1024 \cdot 1024 \cdot 3 = hwc'$ (with $c'=3$).
   - The weight matrix $W$ thus has
     $$
     (hwc) \cdot (hwc') = (hw)^2 c c'
     $$
     parameters, on the order of $10^{13}$ — completely impractical.

Intuitively, this layer allows **every output channel of every pixel** to depend on **all channels of all pixels** in the input, ignoring spatial locality.

---

### 1.4 A note on reshaping and strides

Frameworks like PyTorch store tensors in **strided layouts**. For example:

```python
torch.randn(32, 32, 3).stride()
# (96, 3, 1)
```

- The stride gives the step in memory to move 1 index along each axis.
- Here, the last dimension (channels) is contiguous (stride 1), and moving one step along the first dimension (height) requires $32 \cdot 3 = 96$ steps.
- This is **row-major** ordering (also “raster order” in image analysis).

All reshape/flatten operations follow this underlying strided ordering, which determines how image values are packed into vectors.

------

### 1.5 A 1D toy example

Consider a 1D “image” with 4 pixels and 1 channel:
$$
x = [x_1, x_2, x_3, x_4]^\top.
$$

A dense layer with output dimension 4 (and $c' = 1$) can be written as 
$$
\begin{bmatrix} h_1 \ h_2 \ h_3 \ h_4 \end{bmatrix}

\begin{bmatrix}
W_{11} & W_{12} & W_{13} & W_{14} \
W_{21} & W_{22} & W_{23} & W_{24} \
W_{31} & W_{32} & W_{33} & W_{34} \
W_{41} & W_{42} & W_{43} & W_{44}
\end{bmatrix}
\begin{bmatrix}
x_1 \ x_2 \ x_3 \ x_4
\end{bmatrix}.
$$
This is a convenient sandbox for understanding how locality and convolution change the structure of $W$.

------

## 2. Locality and Image Patches

### 2.1 A distance on the image grid

The pixel grid $(i,j)$ has a natural notion of distance. A convenient choice is the **Chebyshev distance**:
$$
d\bigl((i,j),(i',j')\bigr) = \max(|i - i'|, |j - j'|).
$$
This distance measures how far apart two pixels are in terms of the maximum axis-wise deviation.

------

### 2.2 **Definition — Image patch**

Given an image $X \sim (h,w,c)$, for each pixel $(i,j)$ and radius $k$, define the **patch**
$$
P_k^{(i,j)} = X_{,i-k:i+k,; j-k:j+k,: :},
$$
i.e., the sub-image centered at $(i,j)$ containing all pixels whose distance from $(i,j)$ (using the Chebyshev distance) is at most $k$.

- Shape: $(s, s, c)$ where $s = 2k + 1$.
- We call $s$ the **filter size** or **kernel size**.
- This definition assumes $(i,j)$ is at least $k$ pixels away from the image borders; boundary handling is postponed (zero-padding will fix this).

The key idea: each patch captures a **local neighborhood** around a pixel.

------

### 2.3 **Definition — Local layer**

Let $f$ be a layer mapping $X \sim (h,w,c)$ to $H = f(X) \sim (h,w,c')$.

We say that $f$ is **local** if there exists some radius $k$ such that for every pixel $(i,j)$:
$$
[ f(X) ]_{ij} = f!\bigl(P_k^{(i,j)}\bigr),
$$
i.e., the output at $(i,j)$ depends **only** on the patch of size $s \times s$ around $(i,j)$.

------

### 2.4 Locally-connected layers

We can convert our dense layer into a **local** one by zeroing all weights that connect a pixel to inputs outside its local patch.

For each output pixel $(i,j)$, we define a position-dependent weight matrix
$$
W_{ij} \sim (c', s^2 c)
$$
and set
$$
H_{ij} = \phi\bigl( W_{ij} \cdot \mathrm{vect}(P_k^{(i,j)}) \bigr).
$$
- This yields a **locally-connected layer**.
- The total number of parameters is
$$
    \text{params} = h w \cdot s^2 c c'.
$$
- Compare with the dense layer’s
    $$
    (hw)^2 c c'.
    $$
    We get a reduction factor of roughly
    $$
    \frac{s^2}{hw}.
    $$

So we have exploited locality, but not yet parameter sharing.

------

### 2.5 1D example and zero-padding

Returning to the 1D example with 4 pixels and $k = 1$ (so $s = 3$), a locally-connected layer (ignoring boundaries) might look like: 

$$
\begin{bmatrix} h_1 \ h_2 \ h_3 \ h_4 \end{bmatrix}

\begin{bmatrix}
W_{12} & W_{13} & 0 & 0 \
W_{21} & W_{22} & W_{23} & 0 \
0 & W_{31} & W_{32} & W_{33} \
0 & 0 & W_{41} & W_{42}
\end{bmatrix}
\begin{bmatrix}
x_1 \ x_2 \ x_3 \ x_4
\end{bmatrix}.
$$

Here, the operation is undefined for the outermost pixels if we insist on a 3-point neighborhood, hence the “shortened” filter near edges.

**Zero-padding** fixes this by artificially extending the signal with zeros at the borders, e.g. 
$$
\begin{bmatrix} h_1 \ h_2 \ h_3 \ h_4 \end{bmatrix}

\begin{bmatrix}
W_{11} & W_{12} & W_{13} & 0 & 0 & 0 \
0 & W_{21} & W_{22} & W_{23} & 0 & 0 \
0 & 0 & W_{31} & W_{32} & W_{33} & 0 \
0 & 0 & 0 & W_{41} & W_{42} & W_{43}
\end{bmatrix}
\begin{bmatrix}
0 \ x_1 \ x_2 \ x_3 \ x_4 \ 0
\end{bmatrix}.
$$

In 2D images:

- With kernel size $s = 2k+1$, we need exactly $k$ rows/columns of zeros on each side to keep output shape $(h,w,c')$.
- Without padding, the output shape becomes $(h - 2k,, w - 2k,, c')$.

These are often called **“same”** (with padding) and **“valid”** (no padding) convolutions in libraries.

**Remark (Patch definition).**
The patch-based definition assumes an odd kernel size $s = 2k+1$ and is slightly nonstandard compared to signal-processing definitions, but it greatly simplifies notation. Even kernel sizes are possible in practice but less common.

------

## 3. Translation Equivariance and Convolutional Layers

### 3.1 **Definition — Translation equivariance (via patches)**

A layer $H = f(X)$ is **translation equivariant** if whenever two patches are identical,
$$
P_k^{(i,j)} = P_k^{(i',j')},
$$
the corresponding outputs are identical:
$$
f\bigl(P_k^{(i,j)}\bigr) = f\bigl(P_k^{(i',j')}\bigr).
$$

Informally: if an object shifts from location $(i,j)$ to $(i',j')$, activations associated with that object shift with it, instead of changing in some arbitrary way.

Locally-connected layers **do not** satisfy this property, because they use different matrices $W_{ij}$ and $W_{i'j'}$.

------

### 3.2 Weight sharing → the convolutional layer

A simple way to ensure translation equivariance is **weight sharing**: use the same weight matrix $W$ for all positions:
$$
H_{ij} = \phi\bigl(W \cdot \mathrm{vect}(P_k^{(i,j)})\bigr).
$$

Adding a bias vector $b \sim (c')$ and omitting $\phi$ for the pure linear map gives:

**Definition (Convolutional layer, 2D).**
Given $X \sim (h,w,c)$ and kernel size $s = 2k+1$, a (linear) convolutional layer $H = \mathrm{Conv2D}(X)$ is
$$
H_{ij} = W \cdot \mathrm{vect}(P_k^{(i,j)}) + b,
$$
with

- $W \sim (c', s^2 c)$,
- $b \sim (c')$,

and hyperparameters $k$, $c'$ and the choice of padding (“same” vs “valid”).

This layer:

- is **local** (depends only on a patch around each pixel),
- uses **shared parameters** (same $W$ for all $(i,j)$),
- is **translation equivariant** with respect to shifts inside the padded region,
- has parameter count independent of $h,w$: only $s^2 c c' + c'$.

------

### 3.3 Example: 1D convolution and Toeplitz matrices

In the 1D toy case with $k = 1$ (3-element kernel) and shared weights $W_1, W_2, W_3$, we obtain
$$
\begin{bmatrix}
h_1 & h_2 & h_3 & h_4
\end{bmatrix}
\begin{bmatrix}
W_2 & W_3 & 0 & 0 \\
W_1 & W_2 & W_3 & 0 \\
0 & W_1 & W_2 & W_3 \\
0 & 0 & W_1 & W_2
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{bmatrix}.
$$
This matrix has constant values along each diagonal (e.g., all $W_2$ on the main diagonal). Such matrices are called **Toeplitz matrices**.

- Toeplitz structure is typical of convolution and is crucial for efficient implementations.
- A convolution is still a **linear operation**, just with a highly structured weight matrix compared to a generic dense layer.

------

### 3.4 Implementation note: PyTorch `conv2d`

In PyTorch (using channels-first convention):

```python
from torch.nn import functional as F

x = torch.randn(16, 3, 32, 32)   # (batch, channels, height, width)
w = torch.randn(64, 3, 5, 5)     # (out_channels, in_channels, k, k)

F.conv2d(x, w, padding="same").shape
# -> torch.Size([16, 64, 32, 32])
```

- Kernel tensor $w$ has shape $(c', c, k, k)$.
- `padding="same"` preserves spatial dimensions; `padding="valid"` applies no padding.

------

### 3.5 Signal-processing view of convolution

We can rewrite the convolution in a form closer to signal processing.

Reshape $W$ from matrix $(c', s^2 c)$ to tensor
$$
W \sim (s, s, c', c),
$$
and define an offset mapping
$$
t(m) = m - k - 1, \quad m \in {1, \dots, 2k+1},
$$
so $t(m) \in {-k, \dots, k}$.

Then each output channel $z$ at position $(i,j)$ is
$$
H_{ijz} =
\sum_{i'=1}^{2k+1}
\sum_{j'=1}^{2k+1}
\sum_{d=1}^c
W_{i' j' z d} ,
X_{,i'+t(i),, j'+t(j),, d}.
$$
Interpretation:

- For fixed $z$, the slice $W_{:,:,z,:}$ acts as a **filter** applied to local neighborhoods of the input.
- This corresponds (up to a sign convention) to a discrete **convolution** with a finite impulse response (FIR) filter.

**Example (ridge-detection kernel, single-channel).**
A classical hand-designed $3 \times 3$ filter for ridge detection:
$$
W =
\begin{bmatrix}
-1 & -1 & -1 \
-1 & 8 & -1 \
-1 & -1 & -1
\end{bmatrix}.
$$
In CNNs, instead of manually designing such kernels, we:

- initialize them randomly,
- learn them via gradient descent.

------

### 3.6 Activation maps

Because a convolutional layer preserves spatial structure, each output channel $z$ defines an **activation map**
$$
H_{:,:,z}
$$
showing how strongly the corresponding filter responds at each spatial location.

- These maps are useful for visualization and interpretability.
- They form the basis for many techniques to inspect CNNs (covered in more detail elsewhere).

------

## 4. Convolutional Models (CNNs)

### 4.1 **Definition — Receptive field**

Let $X$ be an input image and $H = g(X)$ be an intermediate tensor in a convolutional model (e.g., after several layers).

The **receptive field** $R(i,j)$ of a pixel $(i,j)$ in $H$ is the subset of $X$ that contributes to its computation:
$$
[ g(X) ]_{ij} = g\bigl(R(i,j)\bigr), \quad R(i,j) \subseteq X.
$$
- For a single convolutional layer with radius $k$:
$$
    R(i,j) = P_k^{(i,j)}.
$$
- For two stacked convolutional layers with the same kernel size, the receptive field becomes larger:
    - Two layers $\Rightarrow R(i,j) \approx P_{2k}^{(i,j)}$,
    - Three layers $\Rightarrow R(i,j) \approx P_{3k}^{(i,j)}$, etc.

So the receptive field grows **linearly with depth**. A deep enough stack yields a **global receptive field**, even if each individual layer is local.

------

### 4.2 Linear collapse and the need for nonlinearities

Since convolution is linear, a stack of purely linear convolutions is equivalent to a **single convolution with a larger kernel**.

To avoid this collapse and gain representational power, we alternate convolutions with nonlinearities:
$$
H = (\phi \circ \mathrm{Conv} \circ \cdots \circ \phi \circ \mathrm{Conv})(X).
$$

- Each convolution changes the number of channels.
- The spatial dimensions remain the same (with “same” padding) or shrink slightly (with “valid” padding).

------

### 4.3 Equivariance vs invariance

Convolutional layers are **translation equivariant**. For classification, we usually want **translation invariance**: the model’s label should not depend on where an object appears.

**Informal definitions (for a transformation $T$):**

- $f$ is **equivariant** if
    $$
    f(Tx) = T f(x)
    $$
    (outputs transform in the same way as inputs).
- $f$ is **invariant** if
    $$
    f(Tx) = f(x)
    $$
    (outputs are unchanged by the transformation).

The set of all such transformations forms a **group**, and each transformation can be represented by a matrix (a group representation). Convolution is closely tied to the group of translations; more general groups lead to other equivariant/invariant architectures.

One simple route from equivariance to invariance: **reduce over spatial dimensions**.

------

### 4.4 Global pooling

Consider a feature map $H \sim (h', w', c')$. Simple invariant operations include:

- **Global sum pooling:**
    $$
    H' = \sum_{i,j} H_{ij}.
    $$
- **Global max pooling:**
    $$
    H' = \max_{i,j} H_{ij}.
    $$

Both are translation-invariant over the spatial dimensions. However, they completely discard **where** features occurred.

------

### 4.5 **Definition — Max-pooling layer**

To partially reduce spatial dimensions while retaining some structure, we use **max-pooling**.

Given $X \sim (h,w,c)$, a 2×2 max-pooling layer produces
$$
\mathrm{MaxPool}(X) \sim \left(\tfrac{h}{2}, \tfrac{w}{2}, c\right),
$$
defined element-wise by
$$
[\mathrm{MaxPool}(X)]_{ijc}
= \max \Bigl\{
X_{pqc} \;:\;
p \in \{2i-1,\,2i\},\;
q \in \{2j-1,\,2j\}
\Bigr\}.
$$

Interpretation:

- For each channel $c$, we take **non-overlapping 2×2 windows** in space.
- We keep only the **maximum** value per window.
- This halves each spatial dimension and preserves the number of channels.

Max-pooling is a form of **local translation invariance** within each window, while still preserving coarse spatial layout at the level of pooled locations.

------

### 4.6 Convolutional “blocks”

A common building unit is the **convolutional block**:
$$
\mathrm{ConvBlock}(X) = (\mathrm{MaxPool} \circ \phi \circ \mathrm{Conv} \circ \cdots \circ \phi \circ \mathrm{Conv})(X),
$$
i.e.:

- a few convolution + nonlinearity layers,
- followed by a pooling operation (often max-pooling).

We can then build deeper networks by **stacking blocks**:
$$
H = (\mathrm{ConvBlock} \circ \mathrm{ConvBlock} \circ \cdots \circ \mathrm{ConvBlock})(X).
$$

Hyperparameters include:

- number of convolutional layers per block,
- kernel sizes,
- number of channels per layer,
- where to place pooling (and type).

To simplify design, architectures often enforce **regular patterns**. For example, VGG-style networks:

- fix kernel size (e.g. $3 \times 3$ everywhere, $k=1$),
- keep the number of channels constant within a block,
- double the number of channels between blocks.

------

### 4.7 Strided convolutions

Instead of (or in addition to) max-pooling, we can reduce spatial resolution via **strided convolutions**:

- Stride 1: standard convolution, output computed at every pixel.
- Stride 2: output computed every 2 pixels; resolution halves.
- Stride 3: output every 3 pixels; resolution shrinks by factor 3, etc.

Strides and pooling can be combined (or one can be omitted) depending on architectural design and hardware efficiency.

------

### 4.8 A prototypical CNN for classification

A typical convolutional-network classifier has three stages:

1. **Backbone (feature extractor):**
    $$
    H = (\mathrm{ConvBlock} \circ \cdots \circ \mathrm{ConvBlock})(X).
    $$
2. **Global pooling over spatial dimensions:**
    $$
    h = \frac{1}{h' w'} \sum_{i=1}^{h'} \sum_{j=1}^{w'} H_{ij},
    $$
    where $H \sim (h', w', c')$ and $h \sim (c')$.
3. **Head (classifier, usually an MLP):**
    $$
    y = \mathrm{MLP}(h).
    $$

Alternatively, one may **flatten** $H$ to a vector and feed it into an MLP, but global pooling is more parameter-efficient and more robust to input resolution changes.

**Example architecture (image classification with 10 classes).**

- Input: shape $(64, 64, 3)$.
- Conv layer, 32 filters → $(64, 64, 32)$.
- 2×2 max-pooling → $(32, 32, 32)$.
- Conv layer, 64 filters → $(32, 32, 64)$.
- 2×2 max-pooling → $(16, 16, 64)$.
- Global average pooling → $(64)$.
- Fully-connected layer with 10 units → $(10)$ outputs (class logits).

Here:

- The **backbone** is everything up to the pooled vector $h$.
- The final fully-connected layer is the **classifier head**.

This decomposition is key for **transfer learning**: we often reuse a pre-trained backbone and swap/finetune the head for a new task.

------

### 4.9 Training and properties of CNNs

1. **Training procedure.**
    The model is trained like standard networks:
    - Add a softmax to the final logits for classification.
    - Use a loss like cross-entropy.
    - Optimize all parameters (including convolution and dense layers) via backpropagation.
2. **Input resolution.**
    With global pooling, the model is conceptually independent of exact input resolution, as long as the convolution + pooling stack can process it. In practice, training and inference usually fix a resolution for efficient batching (variable-length inputs are discussed elsewhere).
3. **Backbone vs head.**
    - The **feature-extraction part** (equation above for $H$ and $h$) is the backbone.
    - The **classification part** ($\mathrm{MLP}(h)$) is the head.
    - This separation is central to reusing CNNs as generic image encoders.

------

## 5. Variants of Convolution

### 5.1 **1×1 convolution**

A special but extremely useful case is $k = 0$, i.e. kernel size $1 \times 1$.

For each pixel $(i,j)$ and output channel $z$:
$$
H_{ijz} = \sum_{t=1}^c W_{z t} , X_{ij t},
$$
with
$$
W \sim (c', c).
$$

Interpretation:

- Each pixel’s $c$-dimensional embedding is transformed to $c'$ dimensions independently of all other pixels.
- This is equivalent to a fully-connected layer applied **per pixel**.
- Commonly used to:
    - change channel dimensionality (e.g. bottleneck layers),
    - mix channel information without enlarging the receptive field.

------

### 5.2 Depthwise and groupwise convolutions

Consider an “orthogonal” idea to the $1 \times 1$ convolution: keep channels separate but mix spatial neighbors.

A **depthwise convolution** with radius $k$ and kernel size $s = 2k+1$ acts as
$$
H_{ijc} =
\sum_{i'=1}^{2k+1}
\sum_{j'=1}^{2k+1}
W_{i' j' c} ,
X_{,i'+t(i),, j'+t(j),, c},
$$
where $W \sim (s, s, c)$ and $t(\cdot)$ is the offset mapping as before.

- Each channel $c$ has its **own spatial filter** but does not mix with other channels.
- The weight tensor is rank-3: one kernel per channel.

A **groupwise convolution** generalizes this by splitting channels into groups:

- Within each group, channels mix with each other and in space.
- Different groups are processed independently.
- Depthwise convolution is the extreme case where the group size is 1.

These variants reduce compute and parameters while retaining spatial modeling.

------

### 5.3 Depthwise separable convolutions

We can combine 1×1 and depthwise convolutions into a **depthwise separable convolution**:

1. A **depthwise** convolution to mix nearby pixels **within each channel**.
2. A **1×1** convolution to mix information **across channels**.

If a standard convolution has parameter count
$$
s^2 c c',
$$
a depthwise separable convolution has
$$
s^2 c + c c'.
$$

- For typical sizes, this is significantly smaller, which is crucial for mobile/low-power devices.
- Many efficient architectures (e.g. mobile-optimized CNNs) use this pattern.
- More broadly, such decompositions — alternately processing along different axes — reappear in other architectures (e.g., transformers: attention vs feed-forward operating along different dimensions).

------

## 6. Practical Implementations and Exercises

### 6.1 Framework support

All key layers in this chapter are provided in common deep-learning libraries:

- **PyTorch (`torch.nn`)**:
    - `Conv2d` for convolutions (with configurable kernel size, stride, padding, groups, etc.),
    - pooling layers such as `MaxPool2d`.
- **torchvision**:
    - datasets and loaders (e.g., MNIST, CIFAR-10),
    - image transforms (cropping, normalization, augmentation) useful for training CNNs.
- **JAX + Equinox** (and similar libraries):
    - analogous abstractions for building and training convolutional models.

A recommended learning path:

- Work through a basic image classification tutorial (e.g., CIFAR-10) using `torchvision`.
- Re-implement the same logic in JAX/Equinox using comparable convolutional layers.

------

### 6.2 Implementing convolution from scratch

Implementing convolution directly can be an instructive exercise:

- One approach uses PyTorch’s **`unfold`/`fold`** operations to explicitly extract all patches:
    - `unfold` turns local patches into columns,
    - then a matrix multiplication with a weight matrix implements the convolution,
    - `fold` can reassemble the output if needed.

However:

- Library-provided convolution implementations are heavily optimized (e.g., use specialized kernels, low-level parallelism).
- Custom implementations are therefore primarily **didactic**, not for production.

------

### 6.3 Convolution via frequency domain (FFT)

From a signal-processing viewpoint, convolution can be implemented as **multiplication in the frequency domain**:

- Apply FFT to the input and the kernel,
- Multiply pointwise in the frequency domain,
- Apply inverse FFT to return to the spatial domain.

For the **small kernels** typical in CNNs (e.g. $3 \times 3$, $5 \times 5$), FFT-based convolution is usually not worth the overhead. But for **very large kernels** (“long convolutions”), FFT-based approaches can be much more efficient.

Modern frameworks provide:

- Differentiable FFT primitives,
- Libraries that implement FFT-based convolutions for such large-kernel scenarios.

------

### 6.4 Summary of key ideas

- **Locality:** Each output unit should depend only on a local neighborhood — captured by patches $P_k^{(i,j)}$ and local layers.
- **Parameter sharing:** Convolutional layers share weights across spatial locations, yielding translation equivariance and dramatic parameter reduction.
- **Receptive field:** Stacking convolutions (with nonlinearities) expands the region of the input that any output depends on.
- **Pooling and strides:** Max-pooling and strided convolutions coarsen spatial resolution and help transition from equivariance to invariance.
- **Architecture pattern:** CNNs separate into a backbone (convolutional feature extractor) and a head (classifier), enabling transfer learning.
- **Convolution variants:** 1×1, depthwise, groupwise, and depthwise separable convolutions trade off expressivity and efficiency, especially important for resource-constrained settings.
- **Implementation:** Deep-learning libraries offer efficient, differentiable building blocks; custom implementations are most useful for understanding and for special cases like long convolutions.
