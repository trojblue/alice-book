# Chapter 6 – Automatic Differentiation

[TOC]

- Contrast **numerical**, **symbolic**, and **automatic** differentiation.
- Formalize **computational graphs**, **primitives**, and **Jacobians** (including shape / memory issues).
- Define **forward-mode AD** (F-AD): concept, algorithm, and its computational limitations.
- Derive **reverse-mode AD** (R-AD / backpropagation): adjoints, dual program, memory trade-offs, checkpointing.
- Introduce **vector–Jacobian products (VJPs)** and show how they give efficient backward passes for linear layers and activations.
- Sketch how modern frameworks (JAX, PyTorch) implement AD.
- Use AD to motivate activation-function choices (sigmoid vs ReLU; vanishing / exploding gradients).
- Discuss **subgradients** and non-smoothness, especially for ReLU at 0.
- Connect theory to practice: implementing small AD engines and custom primitives.

---

## 1. Motivation and Context

We want an **efficient, automatic procedure** to compute gradients of complex programs, especially neural networks. The key setting:

- A program computes a **scalar loss** from parameters and data.
- The computation can be seen as a **computational graph** of simple building blocks (**primitives**).
- We need **all gradients w.r.t. parameters** to run numerical optimization (e.g., SGD).

**Automatic differentiation (AD)**, and in particular **reverse-mode AD** (backpropagation), provides a principled way to do this:

- Works for *any* program built from differentiable primitives.
- Is far more efficient than naive numerical differentiation.
- Is more scalable and structured than generic symbolic differentiation.
- Underlies modern frameworks (TensorFlow, PyTorch, JAX).

---

## 2. Problem Setup and Notation

### 2.1 Computational graphs and primitives

We consider a program as a sequence of **primitive operations**:
- Each primitive is a function
  $$
  y = f_i(x, w_i),
  $$
  where:
  - $x$ is an input vector (activations),
  - $w_i$ is a parameter vector for the $i$-th primitive,
  - $y$ is the output vector.

**Primitives** are flexible:

- They can be:
  - Basic linear algebra ops (e.g., matrix multiplication).
  - Whole layers (e.g., fully-connected with activation).
  - Larger blocks or even full models.
- This **composability** mirrors normal programming, and AD respects this hierarchy.

### 2.2 Jacobians and shapes

For each primitive $f(x, w)$ we assume we can compute:

- **Input Jacobian**
  $$
  \frac{\partial}{\partial x} f(x, w)
  $$
- **Weight Jacobian**
  $$
  \frac{\partial}{\partial w} f(x, w)
  $$

These are matrices (or higher-rank tensors before flattening).

**Example: fully-connected layer with mini-batch**

Consider
$$
f(X, W) = XW + b
$$

- $X$ has shape $(n, c)$ (mini-batch size $n$, input features $c$).
- $W$ has shape $(c, c')$.
- $b$ has shape $(c')$.
- Output $f(X, W)$ has shape $(n, c')$.

Without flattening, the Jacobians are **rank-4** tensors, e.g.:

- Input Jacobian shape: $(n, c', n, c)$.
- Weight Jacobian shape: $(n, c', c, c')$.

To simplify notation:

- Flatten $X$ to $x = \operatorname{vect}(X) \in \mathbb{R}^{nc'}$.
- Flatten parameters to $w = [\operatorname{vect}(W); b]$.

Then:

- Input Jacobian has shape $(nc', nc)$.
- Weight Jacobian has shape $(nc', cc')$.

When we say “input dimension $c$” we really mean **the product of all input shapes**, including mini-batch and spatial dimensions. This is crucial:

- Even if we *know* how to compute Jacobians,
- We **do not want** to *materialize* them explicitly (too big in memory).

---

## 3. Automatic Differentiation: Problem Statement

We consider a sequential program:

- Intermediate variables:
  $$
  \begin{aligned}
  h_1 &= f_1(x, w_1), \\
  h_2 &= f_2(h_1, w_2), \\
  &\vdots \\
  h_\ell &= f_\ell(h_{\ell-1}, w_\ell),
  \end{aligned}
  $$
- Final scalar output (e.g., sum of per-example losses):
  $$
  y = \sum_{k} h_{\ell, k}.
  $$

We abbreviate the whole program as $F(x)$.

### **Definition**: Automatic Differentiation (AD)

Given a program $F(x)$ that is a composition of differentiable primitives, **automatic differentiation** is the task of **simultaneously and efficiently** computing all weight gradients using:

- The computational graph structure.
- The primitives’ input and weight Jacobians.

Formally:
$$
\text{AD}(F(x)) = \left\{ \frac{\partial y}{\partial w_i} \right\}_{i=1}^\ell.
$$

There are two main flavors:

- **Forward-mode AD (F-AD)**.
- **Reverse-mode AD (R-AD)**, aka backpropagation.

Reverse-mode is the one that scales well when you have:

- Many parameters (large $w$),
- But **scalar output** (loss).

---

## 4. Numerical, Symbolic, and Automatic Differentiation

Before AD, two standard approaches:

### 4.1 Numerical differentiation

Use finite differences to approximate:
$$
\frac{\partial f}{\partial x_k}(x) \approx \frac{f(x + \varepsilon e_k) - f(x - \varepsilon e_k)}{2\varepsilon}.
$$

Problems:

- For each scalar you differentiate, you need $\approx 2$ function calls.
- For $P$ parameters, that’s $O(P)$ full forward passes.
- Only practical for **small-scale numerical checks** of implementations, not for full training.

### 4.2 Symbolic differentiation

Given a closed-form expression like
$$
f(x) = a \sin(x) + b x \sin(x),
$$
a symbolic engine (e.g., SymPy) can compute $\frac{df}{dx}$ as a symbolic expression.

Issues:

- Naive symbolic differentiation may recompute common subexpressions (like $\sin(x)$, $\cos(x)$) unnecessarily.
- Finding an **optimal implementation** of the Jacobian that eliminates all redundant computations is **NP-complete** (optimal Jacobian accumulation).
- For large, arbitrary programs (with loops, conditionals, etc.) symbolic expressions become unwieldy.

### 4.3 AD as a middle way

AD:

- Uses the **actual program** (its computational graph).
- Applies the **chain rule** systematically, organizing computations to:
  - Reuse intermediate results.
  - Avoid explicit huge Jacobians.
- Is essentially **symbolic differentiation on traces** of the program, but with controlled structure and sharing.

---

## 5. Forward-Mode Automatic Differentiation (F-AD)

### 5.1 Chain rule in Jacobian form

For two composed functions
$$
h = f_1(x), \quad y = f_2(h),
$$
the Jacobian chain rule says:
$$
\frac{\partial y}{\partial x}
= \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial x}.
$$

If:

- $\dim x = a$, $\dim h = b$, $\dim y = c$,
- Then:
  - $\frac{\partial y}{\partial h}$ is $c \times b$,
  - $\frac{\partial h}{\partial x}$ is $b \times a$,

and we perform a **matrix–matrix multiplication** $(c\times b)(b\times a)$.

Interpretation:

- After computing $h = f_1(x)$ and its Jacobian $\frac{\partial h}{\partial x}$,
- When we apply $f_2$, we can **update the gradient** by multiplying with its Jacobian.

### 5.2 F-AD as tangent propagation

We process primitives **in forward order**. For the $i$-th primitive:
$$
h_i = f_i(h_{i-1}, w_i).
$$

We maintain, for each parameter block $w_j$, a **tangent matrix**
$$
\dot{W}_j \approx \frac{\partial h_i}{\partial w_j}
$$
(current value of gradient wrt earlier weights).

Algorithm sketch:

1. **Initialize for first primitive**:
   $$
   h_1 = f_1(x, w_1), \quad
   \dot{W}_1 = \frac{\partial h_1}{\partial w_1}.
   $$

2. **For each subsequent primitive** $i = 2, \dots, \ell$:
   - Update all previous tangents:
     $$
     \dot{W}_j \leftarrow
       \frac{\partial h_i}{\partial h_{i-1}} \;\dot{W}_j,
       \qquad \forall j < i
     $$
   - Initialize the new tangent:
     $$
     \dot{W}_i = \frac{\partial h_i}{\partial w_i}.
     $$

3. Final operation is the **sum** producing scalar $y$:
   - Gradient for each $w_i$ is:
     $$
     \nabla_{w_i} y = \mathbf{1}^\top \dot{W}_i,
     $$
     where $\mathbf{1}$ is a vector of ones of appropriate dimension.

### 5.3 Complexity and limitations of F-AD

Key operation: updating $\dot{W}_j$:

- Requires multiplying matrices shaped roughly like:
  - $(\text{output size}) \times (\text{input size})$,
  - $(\text{input size}) \times (\text{parameter size})$.

Example:

- Inputs/outputs have shape $(n, d)$ (batch size $n$, feature dimension $d$).
- A naive matrix–matrix multiplication scales like
  $$
  O(n^2 d^2 p_j)
  $$
  for parameter block of size $p_j$.

Consequences:

- F-AD tends to be **quadratic** in batch size and feature dimension.
- For high-dimensional inputs (e.g., images), this is **prohibitively expensive**.

Observation:

- The final gradient with respect to each $w_i$ is a **matrix–vector** product (because the output is scalar), not a full matrix–matrix product.
- This suggests a more efficient direction: **reverse-mode**.

---

## 6. Reverse-Mode Automatic Differentiation (R-AD / Backpropagation)

### 6.1 Unrolling a single gradient term

Take a single block $w_i$ and its contribution to the scalar output $y$:
$$
\nabla_{w_i} y
= \mathbf{1}^\top
\left( \prod_{j=i+1}^{\ell} \frac{\partial h_j}{\partial h_{j-1}} \right)
\frac{\partial h_i}{\partial w_i}.
$$

This is a chain of Jacobian multiplications:

- Leftmost: vector $\mathbf{1}^\top$.
- Middle: product of **input Jacobians**.
- Right: **weight Jacobian** for $w_i$.

Define the **adjoint** (also called reverse sensitivity) for $h_i$:
$$
\tilde{h}_i
:= \mathbf{1}^\top
\prod_{j=i+1}^{\ell} \frac{\partial h_j}{\partial h_{j-1}}.
$$

Then:
$$
\nabla_{w_i} y = \tilde{h}_i \frac{\partial h_i}{\partial w_i}.
$$

Important observations:

1. The product starts from a **vector** on the left, so we always have **vector–matrix** products, not matrix–matrix.
2. The adjoints $\tilde{h}_i$ can be computed **recursively** from the end.

### 6.2 R-AD algorithm (dual program)

Algorithm steps:

1. **Forward pass (primal program)**:
   - Run the original program once:
     $$
     x \to h_1 \to h_2 \to \dots \to h_\ell \to y,
     $$
   - **Store all intermediate outputs** $h_1, \dots, h_\ell$ (and any other needed values).

2. **Initialize adjoint at the output**:
   - Because $y$ is scalar and we consider
     $$
     y = \mathbf{1}^\top h_\ell,
     $$
     we set:
     $$
     \tilde{h}_\ell = \mathbf{1}^\top.
     $$

3. **Backward pass (reverse program)**:
   - For $i$ from $\ell$ down to $1$:
     1. Compute gradient with respect to parameters:
        $$
        \nabla_{w_i} y
        = \tilde{h}_i \frac{\partial h_i}{\partial w_i}.
        $$
     2. Propagate adjoint backward:
        $$
        \tilde{h}_{i-1}
        \leftarrow
        \tilde{h}_i \frac{\partial h_i}{\partial h_{i-1}}.
        $$

This backward pass is the **dual program** to the forward pass.

Terminology in neural networks:

- Forward pass: compute $h_1, \ldots, h_\ell, y$.
- Backward pass: propagate $\tilde{h}_\ell, \ldots, \tilde{h}_0$.

### 6.3 Complexity and memory trade-offs

- Each backward step uses **vector–matrix** products:
  - Complexity is **linear** in all relevant dimensions, much cheaper than F-AD’s matrix–matrix multiplies.
- With a scalar output, reverse-mode AD is typically **$O(1)$ in cost per parameter block** (up to constant factors), making it ideal when:
  - Parameters are numerous,
  - Outputs are low-dimensional (e.g., scalar loss).

Trade-off:

- R-AD requires **storing all intermediate values** needed for Jacobians:
  - This creates a large **memory footprint**.
  - In deep networks, memory often becomes the limiting resource.

### 6.4 Gradient checkpointing

To reduce memory:

- We store only some intermediate activations (**checkpoints**).
- During the backward pass:
  - When we need an activation that wasn’t stored,
  - We **recompute** it from the nearest checkpoint by re-running part of the forward pass.

Effect:

- Decreases memory usage.
- Increases computation cost (extra forward passes).
- Example: with a particular checkpointing scheme, you might do $\approx 1.25\times$ more computation than standard backprop.

Checkpointing is a carefully tuned balance between **memory** and **compute**.

---

## 7. Vector–Jacobian Products (VJPs)

R-AD only needs operations of the form:

- Row vector $v$ times Jacobian of $f$ with respect to some argument.

This motivates packaging these as **vector–Jacobian products (VJPs)**.

### 7.1 Definition

Let $y = f(x)$ with:

- $x \in \mathbb{R}^c$,
- $y \in \mathbb{R}^{c'}$,
- Jacobian $\frac{\partial f(x)}{\partial x}$ of shape $(c' \times c)$.

**Definition (VJP).** The **vector–Jacobian product** of $f$ is the map
$$
\operatorname{vjp}_f(v)
= v^\top \frac{\partial f(x)}{\partial x},
$$
where $v \in \mathbb{R}^{c'}$.

If $f(x_1, \dots, x_n)$ has multiple arguments, we can define:

- $\operatorname{vjp}_{f, x_k}(v)$ for each argument $x_k$.

For our primitive $f(x, w)$, we use two specific VJPs:

- **Input VJP**:
  $$
  \operatorname{vjp}_{f, x}(v) = v^\top \frac{\partial f(x, w)}{\partial x}.
  $$
- **Weight VJP**:
  $$
  \operatorname{vjp}_{f, w}(v) = v^\top \frac{\partial f(x, w)}{\partial w}.
  $$

### 7.2 R-AD in terms of VJPs

Recall step (3) in R-AD:

- To get $\nabla_{w_i} y$ and update $\tilde{h}_{i-1}$, we need:
  - $\tilde{h}_i \frac{\partial h_i}{\partial w_i}$,
  - $\tilde{h}_i \frac{\partial h_i}{\partial h_{i-1}}$.

If we denote $v = \tilde{h}_i$, we can rewrite:

- Gradient w.r.t. weights:
  $$
  \nabla_{w_i} y
  = \operatorname{vjp}_{f_i, w}(v).
  $$
- Updated adjoint:
  $$
  \tilde{h}_{i-1}
  = \operatorname{vjp}_{f_i, x}(v).
  $$

Thus, to implement R-AD **for a given set of primitives**, it suffices to implement:

- Two VJP routines per primitive:
  - One for inputs, one for weights.

We never explicitly form $\frac{\partial f}{\partial x}$ or $\frac{\partial f}{\partial w}$ as full matrices.

### 7.3 Recovering full Jacobians from VJPs

Given a VJP implementation, we can in principle reconstruct the full Jacobian by applying it to basis vectors:

- Let $e_1, \dots, e_{c'}$ be the standard basis of $\mathbb{R}^{c'}$.
- Then:
  $$
  \frac{\partial f(x, w)}{\partial x}
  =
  \begin{bmatrix}
    \operatorname{vjp}_{f, x}(e_1) \\
    \operatorname{vjp}_{f, x}(e_2) \\
    \vdots \\
    \operatorname{vjp}_{f, x}(e_{c'})
  \end{bmatrix}.
  $$

This is usually too expensive in practice (requires $c'$ calls), but conceptually shows **VJPs = rows of Jacobian**.

---

## 8. VJPs for Common Primitives

### 8.1 Linear layer without bias

Let
$$
f(x, W) = Wx,
$$
with:

- $x \in \mathbb{R}^c$,
- $W \in \mathbb{R}^{c' \times c}$,
- $y = f(x, W) \in \mathbb{R}^{c'}$.

The Jacobians:

- $\frac{\partial f}{\partial x} = W$ (a matrix).
- $\frac{\partial f}{\partial W}$ is a rank-3 tensor (each entry of $W$ influences the output linearly).

But the VJPs are simple:

- **Input VJP**:
  $$
  \operatorname{vjp}_{f, x}(v)
  = v^\top W^\top
  = (Wv)^\top.
  $$
- **Weight VJP**:
  $$
  \operatorname{vjp}_{f, w}(v) = v x^\top,
  $$
  i.e. an **outer product** between $v$ and $x$.

So:

- The backward pass through a linear layer:
  - Multiplies adjoints by $W^\top$ to propagate to inputs.
  - Forms an outer product $v x^\top$ to accumulate weight gradients.
- No rank-3 tensors ever need to be explicitly formed.

### 8.2 Elementwise activation (e.g., ReLU)

Consider an activation with no parameters:
$$
f(x) = \phi(x),
$$
applied elementwise.

- The input Jacobian is diagonal:
  $$
  \left[ \frac{\partial \phi(x)}{\partial x} \right]_{ii}
  = \phi'(x_i), \quad
  \left[ \frac{\partial \phi(x)}{\partial x} \right]_{ij} = 0 \ (i \neq j).
  $$

Input VJP:
$$
\operatorname{vjp}_{f, x}(v)
= v^\top \frac{\partial \phi(x)}{\partial x}
= v \odot \phi'(x),
$$
where $\odot$ is elementwise multiplication.

Again:

- We avoid forming the diagonal matrix.
- We just multiply the adjoint vector by the derivative of the activation.

---

## 9. Implementing Reverse-Mode AD in Practice

### 9.1 Functional viewpoint (JAX-style)

Think of primitives and models as **pure functions**.

For a function:
$$
f : \mathbb{R}^c \to \mathbb{R}^{c'},
$$
a VJP can be exposed as a higher-order function:
$$
((\mathbb{R}^c \to \mathbb{R}^{c'}) \to \mathbb{R}^c \to (\mathbb{R}^{c'} \to \mathbb{R}^c)).
$$

Interpretation:

- Given:
  - a function $f$,
  - a point $x$,
- Return:
  - a function that maps $v$ to $\operatorname{vjp}_f(v) = v^\top \frac{\partial f(x)}{\partial x}$.

Similarly, a **gradient** for scalar functions:
$$
((\mathbb{R}^c \to \mathbb{R}) \to (\mathbb{R}^c \to \mathbb{R}^c)),
$$
which is:
- Input: a scalar-output function $f$,
- Output: a function computing $\nabla f(x)$.

This is how tools like `jax.grad` and `jax.jvp` are conceptually structured.

### 9.2 Object-oriented viewpoint (PyTorch-style)

In practice, models are often **objects** (e.g. `nn.Module`):

- Parameters are **encapsulated properties** of the object.
- Forward computation is a **method** (`forward`).

Two approaches:

1. **Functionalize the object**:
   - Extract parameters:
     ```python
     params = dict(model.named_parameters())
     ```
   - Call a functional version of the model:
     ```python
     y = torch.func.functional_call(model, params, x)
     ```
   - Works well with JAX-style APIs.

2. **Attach gradient metadata to tensors (PyTorch autograd)**:
   - Tensors store:
     - Their data.
     - A `grad` field for accumulated gradient.
     - A `grad_fn` pointer describing the operation that produced them.
   - When we call `y.backward()` on a scalar tensor:
     - PyTorch traverses the computational graph **backwards** via `grad_fn`.
     - Applies VJPs and accumulates gradients into `grad` for tensors with `requires_grad=True`.
   - This is essentially a runtime implementation of R-AD over a dynamic computation graph.

Both approaches rely conceptually on:

- Knowing the **VJPs** of primitives.
- Constructing and traversing a **computational graph** in reverse.

### 9.3 Additional details

- On general acyclic graphs, a **topological ordering** of nodes is used before backward:
  - Ensures correct order of accumulating adjoints.
  - Avoids double-counting or missing paths.
- Didactic libraries (e.g. `micrograd`) implement this with:
  - A small tensor-like class,
  - A graph of nodes,
  - A manual backward propagation and topological sort.

---

## 10. Activation Functions and Gradient Stability

AD analysis sheds light on **why certain activations work better**.

In backprop, each layer’s adjoint is multiplied by derivatives of activations:

- For an activation $\phi$:
  $$
  \tilde{h}_{i-1}
  = \tilde{h}_i \odot \phi'(h_{i-1}).
  $$

So across many layers, gradients are repeatedly scaled by $\phi'( \cdot )$.

### 10.1 Vanishing and exploding gradients

If for all $s$:

- $|\phi'(s)| < 1$:
  - Gradients are repeatedly shrunk.
  - Over many layers, they can **vanish** (go to 0).
- $|\phi'(s)| > 1$ (on average):
  - Gradients are repeatedly enlarged.
  - Over many layers, they can **explode** (go to $\infty$).

Given finite-precision arithmetic (e.g. 32-bit floats):

- Vanishing gradients → underflow.
- Exploding gradients → overflow / NaNs.

Both cause training difficulties.

### 10.2 Sigmoid as a case study

Sigmoid:
$$
\sigma(s) = \frac{1}{1 + e^{-s}}.
$$

Derivative:
$$
\sigma'(s) = \sigma(s)(1 - \sigma(s)).
$$

Since $\sigma(s) \in [0, 1]$:

- $\sigma'(s) \in [0, 0.25]$.

Thus:

- Sigmoid is a prime candidate for **vanishing gradients**:
  - Each layer shrinks gradients by at most a factor of $0.25$.
  - Over deep networks, this can quickly drive gradients toward zero.

### 10.3 ReLU as a better compromise

ReLU:
$$
\operatorname{ReLU}(s) = \max(0, s),
$$
with derivative (away from 0):
$$
\frac{\partial}{\partial s}\operatorname{ReLU}(s) =
\begin{cases}
0, & s < 0, \\
1, & s > 0.
\end{cases}
$$

So:

- For positive activations, gradients are **passed unchanged**.
- For negative activations, gradients are **zeroed** (sparse gradients).

Benefits:

- No systematic shrinking or blowing up of gradients on the active (positive) side.
- Induces sparsity (many zeros) which can be computationally helpful.
- Empirically more stable for deep networks than sigmoids.

### 10.4 “Linear non-linear” models

A stack of purely linear layers *should* be equivalent to one linear layer. But in finite-precision floating point:

- Tiny discontinuities and rounding errors make the system **not perfectly linear**.
- This can be exploited to **train deep “linear” networks** that still exhibit interesting behavior.

It is a subtle reminder:

- Practical implementations live in finite precision,
- Which interacts with AD in non-trivial ways.

### 10.5 Memory considerations with ReLU

Because:

- $\operatorname{ReLU}(s)$ keeps positive values unchanged,
- Only negative entries are zeroed,

we can often **overwrite** the input with the output in-place without losing information needed for gradients:

- The gradient mask depends only on the sign of activations.
- Frameworks like PyTorch allow `inplace=True` for ReLU to save memory.

This is a small, but concrete, interaction between **activation choice** and **AD implementation**.

---

## 11. Subgradients and Non-Smoothness in AD

The ReLU is **non-differentiable at 0**. How does AD work then?

### 11.1 Pragmatic answer

In typical training:

- Parameters are initialized randomly.
- SGD uses noisy updates and continuous movements.

The probability that a particular pre-activation is **exactly** zero is negligible. So:

- We rarely “hit” true non-differentiable points.
- Evaluating derivatives as if we were slightly off zero ($s = \pm \varepsilon$) is effectively fine.

### 11.2 Subgradient for convex functions

To be more formal, we use **subgradients**.

**Definition (Subgradient).**  
For a convex function $f:\mathbb{R} \to \mathbb{R}$, a vector $z$ is a **subgradient** at $x$ if for all $y$:
$$
f(y) \ge f(x) + z (y - x).
$$

- Subgradients correspond to slopes of lines that lie below $f$ and touch it at $x$.
- If $f$ is differentiable at $x$, the subgradient set is just $\{\nabla f(x)\}$.
- At a non-smooth point, there may be **many** subgradients; they form the **subdifferential**:
  $$
  \partial_x f(x) = \{ z \mid z \text{ is a subgradient at } x \}.
  $$

### 11.3 ReLU subgradient at 0

For ReLU:
$$
\operatorname{ReLU}(s) = \max(0, s),
$$
we know:

- For $s < 0$: derivative $= 0$.
- For $s > 0$: derivative $= 1$.

At $s = 0$, the subdifferential is:
$$
\partial_s \operatorname{ReLU}(s)
=
\begin{cases}
\{0\}, & s < 0, \\
\{1\}, & s > 0, \\
[0, 1], & s = 0.
\end{cases}
$$

So any value in $[0,1]$ is a valid subgradient at 0.

In practice:

- Deep learning frameworks usually pick **one** subgradient (often 0) as a convention.
- Using such subgradients in iterative methods is called **subgradient descent**.

### 11.4 Non-convexity and further complications

ReLU-based networks are **non-convex**, but the subgradient notion above is defined for convex functions. For non-convex settings:

- More general notions like the **Clarke subdifferential** extend subgradient ideas:
  - Roughly, they look at limits of gradients in neighborhoods of a point.
- AD in non-smooth, non-convex settings can be subtle:
  - Different but algebraically equivalent implementations of the same function might yield **different subgradients** at non-smooth points.
  - Formal chain rules for such generalized subgradients require care.

In practice, frameworks adopt **consistent rules** that work well empirically, even if the underlying non-smooth analysis is intricate.

---

## 12. From Theory to Practice: Building AD Systems

### 12.1 Minimal reverse-mode engines

To internalize R-AD:

- One can implement a tiny AD system, e.g.:
  - Define a scalar-valued variable class that stores:
    - Value,
    - Gradient,
    - Operation that produced it,
    - References to parents.
  - After computing a scalar output:
    - Topologically sort nodes.
    - Traverse them in reverse to propagate gradients.

Key ideas:

- Represent the computation as a **DAG**.
- Identify **VJPs** for each primitive op.
- Use a reverse traversal to accumulate gradients.

### 12.2 Extending frameworks with new primitives

In larger systems (PyTorch, JAX):

- You can define **custom primitives** by specifying:
  - Their **forward computation**,
  - Their **VJPs** (or equivalently, JVPs or custom backward functions).

Examples of exercises:

- Implement trainable activation functions as new primitives:
  - Provide the forward formula,
  - Provide the VJP rules.
- Compare with simply writing them as compositions of existing ops and letting the AD engine infer the backward pass.

### 12.3 Higher-order derivatives

By composing AD transforms:

- Apply `grad` to a function that already uses `grad`:
  - Obtain **Hessians** or higher-order derivatives.
- Using JVPs and VJPs carefully:
  - Construct Hessian-vector products efficiently.

This illustrates that:

- Once AD is available as a **primitive transform**,
- Many advanced operations (e.g., curvature, meta-learning) can be implemented as combinations of grad/JVP/VJP.

---

## 13. Key Takeaways

- **AD vs numerical vs symbolic**:
  - Numerical differentiation is too expensive and unstable for large models.
  - Symbolic differentiation doesn’t scale to arbitrary programs and yields unwieldy expressions.
  - AD operates on the **execution trace** of a program, applying the chain rule efficiently.

- **Forward-mode vs reverse-mode**:
  - F-AD pushes tangents forward; cost grows with **number of inputs**.
  - R-AD pulls adjoints backward; cost grows with **number of outputs** (ideal for scalar loss).
  - Backpropagation is precisely **reverse-mode AD** applied to neural networks.

- **VJPs are the core primitive of R-AD**:
  - Each primitive needs VJPs w.r.t. inputs and parameters.
  - For common ops (linear layers, activations), these are simple and efficient.

- **Frameworks implement AD by combining**:
  - Computational graphs (explicit or implicit).
  - Stored intermediate values (activations).
  - VJP definitions per primitive.
  - Reverse traversals (backward passes).

- **Activation functions and AD** are tightly linked:
  - Choice of $\phi$ affects gradient scaling.
  - Sigmoid can cause vanishing gradients.
  - ReLU is a useful compromise, both analytically (derivative $0$ or $1$) and computationally (sparse, in-place).

- **Non-smoothness** is handled via subgradients:
  - ReLU’s derivative at 0 is chosen from a valid subgradient set.
  - AD in non-smooth, non-convex settings is subtle, but practical rules work well.

- **Learning AD by building it**:
  - Implementing a small R-AD engine and custom primitives makes the abstract chain rule concrete.
  - Understanding these mechanics gives deeper intuition for how frameworks compute gradients—and why certain model design choices matter.