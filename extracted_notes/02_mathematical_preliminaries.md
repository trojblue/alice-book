# Chapter 2 – Mathematical Preliminaries

[TOC]

- Clarifies **notation and perspective** for tensors, vectors, matrices, and higher-order arrays used throughout the book.
- Reviews **core linear algebra operations** (inner products, norms, matrix multiplication, Hadamard products, reductions, batched operations, einsum).
- Introduces **derivatives, gradients, directional derivatives, and Jacobians**, with a geometric “local linear approximation” viewpoint.
- Presents **gradient descent** as a general method for unconstrained optimization, including notions of **local/global minima, stationary points, convexity**, and **convergence guarantees**.
- Discusses **practical acceleration methods** (momentum, adaptive step sizes like Adam) and their computational trade-offs.
- Ends with **practical guidance**: learning NumPy, then JAX/PyTorch, and an exercise to implement gradient descent (with momentum) in all three frameworks.

---

## 1. About This Chapter

- Goal: collect **all mathematical prerequisites** needed for the rest of the book, in a **compressed, notation-focused** way.
- Assumptions:
  - Reader has **prior exposure** to linear algebra, calculus, and probability.
  - Chapter focuses on **standardizing notation**, emphasizing the link between **mathematical concepts (e.g., tensors)** and their **implementation** in deep learning frameworks.
- Structure:
  1. **Linear algebra** and tensors.
  2. **Gradients / Jacobians** for multi-dimensional functions.
  3. **Optimization via gradient descent** and variants.
- Probability theory (with emphasis on **maximum likelihood**) is postponed to **Appendix A**.
- The chapter is **dense with definitions and concepts**; it sets the technical language for later chapters.

---

## 2. Linear Algebra

### 2.1 Tensors

**Definition (Tensor)**  
A tensor $X$ is an $n$-dimensional array of elements of the same type. Its **shape** is denoted by  
$$
X \sim (s_1, s_2, \dots, s_n).
$$

- Special cases by order:
  - $n = 0$: **scalar** (single value).
  - $n = 1$: **vector**.
  - $n = 2$: **matrix**.
  - $n \ge 3$: **higher-dimensional tensor** (e.g., images, feature maps, sequences of vectors, etc.).

**Notation conventions:**

- Scalars: $x$ (lowercase).
- Vectors: $\mathbf{x}$ (lowercase bold).
- Matrices: $\mathbf{X}$ (uppercase bold).
- Tensors (higher order): also uppercase bold, with shape explicit when needed.

**Why tensors matter in deep learning:**

- Tensors are the **native data structure** for modern hardware:
  - Well suited for **massively parallel** implementations (GPUs, TPUs, IPUs, etc.).
- A tensor is specified by:
  - **Element type** (e.g., floating point, integer, string).
  - **Shape** (tuple of dimension sizes).
- In deep learning, we mostly use **floating-point tensors**, but also:
  - **Integer** tensors (e.g., class indices).
  - **String** tensors (e.g., token sequences in NLP).

**Indexing:**

- The book follows conventions inspired by **NumPy**:
  - In the text, indices start from **1** (for human readability), even though NumPy uses **0-based** indexing.
- Example: $X \sim (a, b, c)$ (a 3D tensor).
  - $X_i$ — slice of shape $(b, c)$ (fix first index).
  - $X_{ijk}$ — single scalar.
  - More complex slice, e.g. $X_{i, :, j:k}$ — shape $(b, k-j)$ in the book’s 1-based convention.
- To visually separate indexing from other expressions, they sometimes write:
  $$
  [X]_{ijk}
  $$
  where the argument of $[\cdot]$ can be an expression, not just a symbol.

---

### 2.2 Common Vector Operations

We mostly work with models that are compositions of **differentiable operations**: additions, multiplications, and standard non-linearities (e.g. $\exp(x)$, $\sin(x)$, $\cos(x)$, square roots).

Let a vector $x$ be a **1D tensor** of shape:
$$
x \sim (d).
$$

#### 2.2.1 Row vs Column Vectors and Broadcasting

- In mathematics:
  - **Column vector**: $x$.
  - **Row vector**: $x^\top$.
- In code (NumPy, PyTorch, etc.):
  - A vector can be represented as:
    - Shape $(d)$: 1D tensor.
    - Shape $(d, 1)$: column vector.
    - Shape $(1, d)$: row vector.
- **Broadcasting** (NumPy-style):
  - Frameworks align shapes **from the right** and try to “repeat” dimensions to match.
  - This can lead to **unexpected results**, e.g. adding a tensor of shape $(4, 1)$ to one of shape $(4,)$ and getting a $(4, 4)$ output due to broadcasting rules.
- **Practical caution**: You must be aware of shapes when doing “vector” operations in code, as some operations that are mathematically “obvious” may produce **higher-dimensional outputs** due to implicit broadcasting.

#### 2.2.2 Vector Space, Linear Combinations, and Norms

Vectors of the same shape form a **vector space** under usual addition and scalar multiplication.

Given $x, y \sim (d)$ and scalars $a, b$:
$$
z = a x + b y \quad \Rightarrow \quad z_i = a x_i + b y_i.
$$

**Euclidean (ℓ₂) norm:**

- Interpreting $x$ as a point in $\mathbb{R}^d$, the Euclidean distance from the origin is:
  $$
  \|x\| = \sqrt{\sum_i x_i^2}.
  $$
- The **squared norm**:
  $$
  \|x\|^2 = \sum_i x_i^2
  $$
  is often more convenient (no square root), especially in optimization and code.

#### 2.2.3 Inner Product and Cosine Similarity

**Definition (Inner product / Dot product)**  
For $x, y \sim (d)$, the inner product is:
$$
\langle x, y \rangle = x^\top y = \sum_i x_i y_i.
$$

- The result is a **scalar**.

**Example:**

- $x = [0.1, 0, -0.3]$  
- $y = [-4.0, 0.05, 0.1]$  
  $$
  \langle x, y \rangle = 0.1(-4.0) + 0 \cdot 0.05 + (-0.3)(0.1)
  = -0.4 + 0 - 0.03 = -0.43.
  $$

**Geometric interpretation:**

The dot product is related to the angle $\alpha$ between $x$ and $y$:
$$
x^\top y = \|x\| \, \|y\| \cos(\alpha).
$$

If $\|x\| = \|y\| = 1$ (both **normalized**):

- $\langle x,y\rangle = \cos(\alpha)$ is the **cosine similarity**.
- Range:
  - $1$: same direction.
  - $0$: orthogonal (perpendicular).
  - $-1$: opposite direction.

Maximization viewpoint (for unit vectors):

$$
y^* = \arg\max_y \langle x, y\rangle = x.
$$

- For **fixed normalized** $x$, the dot product is maximized by choosing **$y=x$**.
- In later chapters, $x$ often denotes an **input**, while parameters (e.g. $w$) play the role of **templates**; maximizing $\langle x, w\rangle$ corresponds to **template matching** or “resonance” between input and weights.

#### 2.2.4 Sums and Distances via Inner Products

Two useful rewritings:

1. **Sum of vector entries** as a dot product with the all-ones vector:
   - Let $\mathbf{1} = [1,1,\dots,1]^\top \in \mathbb{R}^d$.
   - Then
     $$
     \langle x, \mathbf{1}\rangle = \sum_{i=1}^d x_i.
     $$

2. **Squared distance** between two vectors:
   $$
   \|x - y\|^2 = \langle x, x\rangle + \langle y, y\rangle - 2\langle x, y\rangle.
   $$
   Special case $y=0$:
   $$
   \|x\|^2 = \langle x, x\rangle.
   $$

These identities are useful both **conceptually** and for **efficient implementation** in code.

---

### 2.3 Common Matrix Operations

Let a matrix $X \sim (n, d)$ (with $n$ rows, $d$ columns) be written as:

$$
X =
\begin{bmatrix}
X_{11} & \dots & X_{1d} \\
\vdots & \ddots & \vdots \\
X_{n1} & \dots & X_{nd}
\end{bmatrix}.
$$

We can interpret $X$ as a **stack of row vectors**:
$$
X =
\begin{bmatrix}
x_1^\top \\
\vdots \\
x_n^\top
\end{bmatrix}.
$$

- In this view, $X$ encodes a **batch of data vectors** $(x_1, \dots, x_n)$.
- It is standard in deep learning to write models to operate **directly on batches**, both mathematically and in code.

#### 2.3.1 Matrix Multiplication

**Definition (Matrix multiplication)**  
Let $X \sim (a, b)$ and $Y \sim (b, c)$, with compatible inner dimension $b$. Define $Z = X Y$, where $Z \sim (a, c)$ and
$$
Z_{ij} = \langle X_i, Y_j^\top \rangle,
$$
i.e. the dot product of the $i$-th row of $X$ with the $j$-th column of $Y$.

**Special cases:**

- **Matrix–vector product**:
  $$
  z = W x
  $$
  with $W \sim (o, d)$ and $x \sim (d)$ gives $z \sim (o)$.

- **Batched matrix–vector computation**:
  - If $X$ is a batch of input vectors (rows), then:
    $$
    X W^\top
    $$
    computes **$n$ dot products** in parallel (one for each row of $X$).

- **Gram matrix / all pairwise dot products**:
  - For $X \sim (n, d)$,
    $$
    X X^\top \sim (n, n)
    $$
    collects all pairwise dot products between rows of $X$.

#### 2.3.2 Hadamard (Element-wise) Multiplication

**Definition (Hadamard product)**  
Given matrices $X, Y$ of the **same shape**, their Hadamard product is:
$$
[X \odot Y]_{ij} = X_{ij} Y_{ij}.
$$

- This is **element-wise multiplication**.
- It lacks many algebraic properties of standard matrix multiplication but is:
  - Widely used for **masking** (e.g. zeroing or scaling certain entries).
  - Important in architectures with **multiplicative interactions**.

#### 2.3.3 Element-wise vs Matrix Exponential

Sometimes one writes expressions like $\exp(X)$.

- Unless otherwise specified, deep learning frameworks interpret this as **element-wise**:
  $$
  [\exp(X)]_{ij} = \exp(X_{ij}).
  $$

By contrast, the **matrix exponential** (from linear algebra) for a **square** matrix $X$ is:
$$
\mathrm{mat\text{-}exp}(X)
= \sum_{k=0}^{\infty} \frac{1}{k!} X^k.
$$

Key points:

- Element-wise $\exp(X)$:
  - Defined for tensors of **any shape**.
  - Standard in numerical frameworks.
- Matrix exponential:
  - Defined only for **square matrices**.
  - Implemented via specialized routines (e.g., `torch.linalg.matrix_exp`).

Frameworks usually separate **general tensor ops** and **matrix-specific ops** into different modules or namespaces.

#### 2.3.4 Reduction Operations

We often sum or average along an axis **without writing indices explicitly**. For example:

- Summation over an entire axis:
  $$
  \sum_i X_i = \sum_{i=1}^n X_i.
  $$

In frameworks like PyTorch, reduction operations are methods with an axis argument:

- Example:
  - `X.sum(axis=1)` computes the sum across columns for each row.

This notation avoids writing explicit index bounds and extends naturally to **multi-dimensional tensors**.

---

### 2.4 Why Matrix Multiplication Is Defined as It Is

A conceptual justification for the definition of matrix multiplication:

- Let $f$ be a function on vectors that is **linear**, i.e.
  $$
  f(\alpha x_1 + \beta x_2) = \alpha f(x_1) + \beta f(x_2).
  $$
- Any such $f$ can be represented by a **matrix** $A$ such that:
  $$
  f(x) = A x.
  $$
- If $g$ is another linear map with matrix $B$, then their composition
  $$
  (f \circ g)(x) = f(g(x)) = A (B x)
  $$
  corresponds to **matrix multiplication**:
  $$
  f \circ g \quad \leftrightarrow \quad AB.
  $$

So the definition $Z = X Y$ as “row-by-column dot products” is precisely the one that makes matrices represent **linear maps**, and matrix multiplication represent **composition of linear maps**.

---

### 2.5 Computational Complexity and Big-$\mathcal{O}$

Using matrix multiplication as an example, consider $Z = X Y$ with:
- $X \sim (a, b)$,
- $Y \sim (b, c)$.

A naive implementation:

- Computes **$a c$ inner products** of dimension $b$.
- **Time complexity**: proportional to $a b c$.
- **Space complexity (sequential)**: dominated by storing the result $Z \sim (a, c)$.

To formalize asymptotic behavior, we use **big-$\mathcal{O}$ notation**.

**Definition (Big-$\mathcal{O}$)**  
We say $f(x) = \mathcal{O}(g(x))$ (for non-negative $f,g$) if there exist constants $c > 0$ and $x_0$ such that:
$$
f(x) \le c\,g(x) \quad \text{for all } x \ge x_0.
$$

- We ignore constant factors and lower-order terms for **sufficiently large** $x$.
- This is **asymptotic analysis**.

For naive matrix multiplication:

- Time complexity:
  $$
  T(a,b,c) = \mathcal{O}(a b c).
  $$
- For square matrices ($a = b = c = n$):
  $$
  T(n) = \mathcal{O}(n^3).
  $$

**Caveats and practical remarks:**

- There exist algorithms with better asymptotic complexity, e.g. with time
  $$
  \mathcal{O}(n^c) \quad \text{for } c < 2.4,
  $$
  but they are **hard to parallelize efficiently** and rarely used in practice.
- Parallelism:
  - Having $k$ processors theoretically gives at best a **constant** speedup of factor $\frac{1}{k}$.
  - Asymptotic $\mathcal{O}$ complexity **does not change**.
- **Memory vs compute bound**:
  - In many practical situations, the bottleneck is **data movement** (memory bandwidth), not raw arithmetic.
  - An algorithm can become **memory-bound**, meaning time is dominated by how fast data can be moved.
  - Whether code is memory- or compute-bound is typically assessed using a **profiler**.

The key message: asymptotic complexity is important, but **hardware details and parallelism** can dominate real-world performance.

---

### 2.6 Higher-Order Tensor Operations

For tensors of order $> 2$, most useful operations are either:

1. **Batched variants** of matrix operations.
2. **Reductions** (sums, means, etc.), sometimes combined with matrix operations.

#### 2.6.1 Batched Matrix Multiplication (BMM)

Let:
- $X \sim (n, a, b)$,
- $Y \sim (n, b, c)$.

Define **batched matrix multiplication**:
$$
[\mathrm{BMM}(X, Y)]_i = X_i Y_i, \quad i = 1,\dots,n,
$$
so that
$$
\mathrm{BMM}(X, Y) \sim (n, a, c).
$$

- Each pair $(X_i, Y_i)$ is multiplied as usual matrices.
- Frameworks like PyTorch and JAX implement this:
  - Often via the **same API** as standard matrix multiplication (`matmul`, `@`).
  - **Leading dimensions** are treated as batch dimensions.

#### 2.6.2 Generalized Dot Product (GDT)

Let $X, Y \sim (a, b, c)$.

The **generalized dot product** is:
$$
\mathrm{GDT}(X, Y) = \sum_{i,j,k} [X \odot Y]_{ijk}
= \sum_{i,j,k} X_{ijk} Y_{ijk}.
$$

- Equivalent to:
  - Flatten both tensors into vectors and compute a standard **dot product**.
- This pattern—**element-wise operation** followed by a **reduction**—is extremely common.

These examples cover the main tensor-level operations used later, with additional ones introduced only when needed.

---

### 2.7 Einstein Summation Notation (einsum)

This (optional) section introduces **Einstein summation notation**, which underlies functions like `einsum` in NumPy, PyTorch, JAX, etc.

The idea: many tensor operations (matrix multiplication, reductions, etc.) can be described by:

- Writing indices for each operand.
- Summing over indices that **appear on the right but not on the left**.

#### 2.7.1 Examples in Index Notation

**Batched matrix multiplication**: $X \sim (n,a,b)$, $Y \sim (n,b,c)$.

Using indices:
$$
M_{ijk} = \sum_z A_{ijz} B_{izk}.
$$

**Generalized dot product**: $X,Y \sim (a,b,c)$.
$$
M = \sum_{i,j,k} X_{ijk} Y_{ijk}.
$$

In Einstein notation, sums over **repeated indices** are implicit. For example:

- If $M_{ijk} = A_{ijz} B_{izk}$, then
  $$
  M_{ijk} = \sum_z A_{ijz} B_{izk}.
  $$
- If $M = X_{ijk}Y_{ijk}$, then
  $$
  M = \sum_{i,j,k} X_{ijk}Y_{ijk}.
  $$

Indices that appear on the **right only** are summed over; indices that appear on the **left** define the shape of the output.

#### 2.7.2 String Representation for `einsum`

The same operations can be encoded as strings:

- BMM example:
  - Indices: $A_{ijz}, B_{izk} \to M_{ijk}$.
  - Einsum string:  
    `'ijz,izk->ijk'`.
- GDT example:
  - Indices: $X_{ijk}, Y_{ijk} \to M$ (scalar).
  - Einsum string:  
    `'ijk,ijk->'`.

These strings have a **one-to-one correspondence** with the index-based definitions.

Advantages:

- Unified, framework-agnostic description of tensor operations.
- Avoids memorizing many specialized APIs (e.g., `matmul`, `bmm`, `tensordot`).
- Works identically across frameworks (e.g., `torch.einsum`, `jax.numpy.einsum`).

**Transposed axes:**  
For $A \sim (n, a, b)$ and $B \sim (n, c, b)$, a batched multiplication of $A_i$ by $B_i^\top$ (transposing the last two indices of $B$) has einsum string:

- `'ijz,ikz->ijk'`.

**Generalizations:**  
Libraries like **`einops`** build on einsum-style thinking to provide higher-level tensor reshaping and combination operations, and are widely used in modern deep learning code.

---

## 3. Gradients and Jacobians

Gradients are central to **differentiable models** and **gradient-based optimization**. We review derivatives for:

- **Scalar functions** (scalar in, scalar out).
- Functions from $\mathbb{R}^d$ to $\mathbb{R}$ (gradients, directional derivatives).
- Functions from $\mathbb{R}^d$ to $\mathbb{R}^o$ (Jacobians).

We emphasize a **geometric “local linear approximation”** viewpoint.

---

### 3.1 Derivatives of Scalar Functions

Let $y = f(x)$ with scalar input and scalar output.

**Definition (Derivative):**
$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}.
$$

**Notation:**

- Generic derivative / gradient: $\partial$.
- To emphasize variables: $\dfrac{\partial}{\partial x}$.
- For scalar-to-scalar functions, $f'(x)$ is **Lagrange’s notation**.

We assume derivatives **exist** where needed (we return to non-smooth cases later, e.g., $f(x) = |x|$ at $x=0$).

**Basic derivative rules:**

- Power:
  $$
  \frac{\partial}{\partial x} x^p = p x^{p-1}.
  $$
- Logarithm:
  $$
  \frac{\partial}{\partial x} \log(x) = \frac{1}{x}.
  $$
- Sine:
  $$
  \frac{\partial}{\partial x} \sin(x) = \cos(x).
  $$

**Geometric meaning:**

- $f'(x)$ is the **slope** of the tangent line at $x$.
- It is the **best first-order (linear) approximation** of $f$ near $x$.
- Sign of $f'(x)$:
  - Positive: locally **increasing**; function rises to the right of $x$.
  - Negative: locally **decreasing**.

**Key calculus properties (for scalar functions):**

- **Linearity:**
  $$
  \frac{d}{dx} \big(f(x) + g(x)\big) = f'(x) + g'(x).
  $$
- **Product rule:**
  $$
  \frac{d}{dx} \big(f(x)g(x)\big) = f'(x)g(x) + f(x)g'(x).
  $$
- **Chain rule (composition):**
  $$
  \frac{d}{dx} f(g(x)) = f'(g(x)) \, g'(x).
  $$

---

### 3.2 Gradients and Directional Derivatives

Now consider a function $y = f(x)$ with vector input $x \sim (d)$ and scalar output $y \in \mathbb{R}$.

#### 3.2.1 Partial Derivatives

To talk about infinitesimal perturbations, we must specify a **direction**. A basic choice is to move along a coordinate axis.

Let $e_i \sim (d)$ be the **$i$-th standard basis vector**:
$$
[e_i]_j =
\begin{cases}
1, & j = i,\\
0, & j \ne i.
\end{cases}
$$

**Partial derivative in direction $x_i$:**
$$
\frac{\partial f(x)}{\partial x_i}
= \lim_{h \to 0} \frac{f(x + h e_i) - f(x)}{h}.
$$

This measures how $f$ changes when **only the $i$-th coordinate** is perturbed.

#### 3.2.2 Gradient

**Definition (Gradient):**  
The **gradient** of $f$ at $x$ is the vector of all partial derivatives:
$$
\nabla f(x) = \partial f(x) =
\begin{bmatrix}
\frac{\partial f(x)}{\partial x_1} \\
\vdots \\
\frac{\partial f(x)}{\partial x_d}
\end{bmatrix}.
$$

- In many contexts, $\nabla f(x)$ is written simply as $\partial f(x)$.
- The gradient is a vector in $\mathbb{R}^d$.

#### 3.2.3 Directional Derivative

For a **general direction** $v \in \mathbb{R}^d$, the **directional derivative** is:
$$
D_v f(x) = \lim_{h \to 0} \frac{f(x + h v) - f(x)}{h}.
$$

Using linearity and the expansion of $v$ in the basis $\{e_i\}$, one obtains:
$$
D_v f(x) = \langle \nabla f(x), v \rangle
= \sum_i \frac{\partial f(x)}{\partial x_i} v_i.
$$

**Key takeaway:**

- Knowing the **gradient** $\nabla f(x)$ is enough to compute **all directional derivatives** via dot products.

---

### 3.3 Jacobians and Linear Approximation

Now let $f: \mathbb{R}^d \to \mathbb{R}^o$ with vector input $x \sim (d)$ and **vector output** $y \sim (o)$.

#### 3.3.1 Jacobian Matrix

**Definition (Jacobian):**  
The **Jacobian matrix** of $f$ at $x$ is:
$$
\frac{\partial f(x)}{\partial x}
=
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \dots & \frac{\partial y_1}{\partial x_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_o}{\partial x_1} & \dots & \frac{\partial y_o}{\partial x_d}
\end{pmatrix}
\in \mathbb{R}^{o \times d}.
$$

- Special cases:
  - $o = 1$: the Jacobian reduces to the **gradient** (a row or column, depending on convention).
  - $d = o = 1$: we recover the **standard scalar derivative**.

The Jacobian inherits standard derivative properties. In particular:

**Matrix chain rule:**  
For a composition $f(g(x))$:
$$
\frac{\partial [f(g(x))]}{\partial x}
= \left.\frac{\partial f(u)}{\partial u}\right|_{u = g(x)} \cdot \frac{\partial g(x)}{\partial x}.
$$

That is, the Jacobian of a composition is the **matrix product** of the Jacobians.

#### 3.3.2 First-order Taylor Approximation

Gradients and Jacobians can be interpreted as **best linear approximations** near a point.

For scalar output $f: \mathbb{R}^d \to \mathbb{R}$ and a point $x_0$:

- The **first-order (linear) Taylor approximation** of $f$ around $x_0$ is:
  $$
  \tilde f(x) = f(x_0) + \langle \nabla f(x_0), x - x_0\rangle.
  $$

- In one dimension, this becomes:
  $$
  \tilde f(x) = f(x_0) + f'(x_0)(x - x_0),
  $$
  the familiar equation of the tangent line.

Interpretation:

- $\langle \nabla f(x_0), x - x_0\rangle$ is the **slope (directional change)** scaled by the displacement $x - x_0$.
- For $x$ close to $x_0$, $\tilde f(x)$ approximates $f(x)$ well.

---

### 3.4 On the Dimensionality of Jacobians

Consider the linear function:
$$
y = W x,
$$
with:
- $W \sim (o, d)$,
- $x \sim (d)$,
- $y \sim (o)$.

**As a function of $x$ (with $W$ fixed):**

- The Jacobian is an $(o \times d)$ matrix:
  $$
  \frac{\partial [W x]}{\partial x} = W.
  $$

**As a function of $W$ (with $x$ fixed):**

- The input is now the matrix $W \sim (o, d)$.
- The “Jacobian” (in the sense of derivative of each output component $y_i$ w.r.t. each entry $W_{ij}$) is of shape $(o, o, d)$.
  - More concretely, from:
    $$
    y_i = \sum_j W_{ij} x_j,
    $$
    we get:
    $$
    \frac{\partial y_i}{\partial W_{ij}} = x_j.
    $$
- If we **vectorize** $W$ into $\mathrm{vect}(W) \sim (od)$, we can view this derivative as a traditional Jacobian matrix $\in \mathbb{R}^{o \times od}$.

Practical implication:

- Explicitly storing full Jacobians is usually **wasteful** (lots of repeated values, large memory).
- In deep learning, we rarely materialize Jacobians explicitly; instead, we work with **vector–Jacobian products** or **Jacobian–vector products**.
- This perspective underlies efficient **automatic differentiation** (discussed later).

---

## 4. Gradient Descent

We now consider **optimization**: minimizing a scalar-valued function.

Let $f: \mathbb{R}^d \to \mathbb{R}$ and consider the unconstrained problem:
$$
x^* = \arg\min_x f(x).
$$

- In the book’s context:
  - $x$ typically represents **model parameters**.
  - $f(x)$ measures some notion of **performance / loss** on the data.

Maximization vs minimization:

- Maximizing $f(x)$ is equivalent to minimizing $-f(x)$, and vice versa.

Closed-form solutions are rare (one exception: **least-squares** under certain conditions). Thus, we usually resort to **iterative methods**.

---

### 4.1 Gradient Descent as Iterative Optimization

We start from an initial guess $x_0$ (often random) and iteratively update:

$$
x_t = x_{t-1} + \eta_t p_t,
$$

where:

- $p_t$ is the **search direction** at iteration $t$.
- $\eta_t > 0$ is the **step size**, or **learning rate**.

**Descent direction:**

- A direction $p_t$ is called a **descent direction** if there exists some $\eta_t > 0$ such that:
  $$
  f(x_t) \le f(x_{t-1}) \quad \text{with } x_t = x_{t-1} + \eta_t p_t.
  $$

For differentiable functions:

- Using the directional derivative:
  $$
  D_{p_t} f(x_{t-1}) = \langle \nabla f(x_{t-1}), p_t \rangle,
  $$
  a direction $p_t$ is a **descent direction** if:
  $$
  D_{p_t} f(x_{t-1}) \le 0.
  $$

Expressing $\langle \nabla f, p_t \rangle$ via angles:

- Let $\alpha$ be the angle between $\nabla f(x_{t-1})$ and $p_t$.
- Then:
  $$
  \langle \nabla f(x_{t-1}), p_t\rangle
  = \|\nabla f(x_{t-1})\| \, \|p_t\| \cos(\alpha).
  $$
- If we restrict to **unit directions** ($\|p_t\| = 1$), the sign depends on $\cos(\alpha)$.

**Steepest descent direction:**

- Any direction with $\alpha \in (\pi/2, 3\pi/2)$ (i.e., pointing “more opposite than aligned” to the gradient) is a descent direction.
- Among them, the direction with **minimum directional derivative** (steepest descent) is:
  $$
  p_t = -\nabla f(x_{t-1}),
  $$
  corresponding to $\alpha = \pi$ (exactly opposite to the gradient).

This yields **gradient descent**.

**Definition (Steepest gradient descent):**  
Given differentiable $f(x)$, initial point $x_0$, and step sizes $\eta_t$, gradient descent updates:

$$
x_t = x_{t-1} - \eta_t \nabla f(x_{t-1}).
$$

- We assume step sizes are chosen **small enough** to decrease $f$.
- The cost of each step is governed by the cost of **computing $\nabla f$**.

---

### 4.2 Convergence: Minima, Stationary Points, Convexity

We now clarify what kind of points gradient descent can converge to.

#### 4.2.1 Local Minimum and Stationary Points

**Definition (Local minimum):**  
A point $x^+$ is a **local minimum** of $f$ if there exists $\varepsilon > 0$ such that:
$$
f(x^+) \le f(x) \quad \text{for all } x \text{ with } \|x - x^+\| < \varepsilon.
$$

- Intuitively, within some small neighborhood (ball of radius $\varepsilon$), $x^+$ has the smallest function value.

At a local minimum, the **gradient must vanish**.

**Definition (Stationary point):**  
A point $x^+$ is a stationary point of $f$ if:
$$
\nabla f(x^+) = 0.
$$

Stationary points include:

- **Local minima**.
- **Local maxima**.
- **Saddle points**, where the function has directions of both increase and decrease (curvature changes sign).

Without additional assumptions:

- Gradient descent can be proven to converge (under suitable conditions on step sizes, etc.) to **some stationary point**, but:
  - There is no guarantee that this point is a **minimum**, let alone a **global minimum**.
  - The specific stationary point found depends on **initialization**.

#### 4.2.2 Global Minimum and Convexity

**Definition (Global minimum):**  
A point $x^*$ is a global minimum of $f$ if:
$$
f(x^*) \le f(x) \quad \text{for all } x.
$$

In a simple **parabola** (e.g. $f(x) = x^2$), the unique stationary point is also a **unique global minimum**.

To generalize this idea, we use **convexity**.

**Definition (Convex function):**  
A function $f: \mathbb{R}^d \to \mathbb{R}$ is **convex** if for all $x_1, x_2$ and all $\alpha \in [0,1]$:
$$
f(\alpha x_1 + (1 - \alpha) x_2)
\le \alpha f(x_1) + (1 - \alpha) f(x_2).
$$

Geometric meaning:

- For any two points $(x_1, f(x_1))$ and $(x_2, f(x_2))$, the graph of $f$ lies **below or on** the straight line segment connecting them.
- An upward-opening parabola is a standard example.

**Strict convexity:**

- If the inequality is **strict** whenever $x_1 \ne x_2$ (and $\alpha \in (0,1)$), then $f$ is **strictly convex**.

**Key facts (for convex $f$):**

1. For a **general non-convex** function:
   - Gradient descent (under suitable conditions) converges to a **stationary point**.
   - Without further information (e.g., curvature), this stationary point might be a saddle or local maximum.

2. For a **convex** function:
   - Every local minimum is also a **global minimum**.
   - Gradient descent (with appropriate step sizes) converges to **a global minimizer**, independent of initialization.

3. For a **strictly convex** function:
   - The global minimizer is **unique**.

For **non-convex** problems (the usual case in deep learning):

- Finding a global minimum is, in general, **NP-hard**.
- A theoretical strategy (“try all initializations”) is computationally infeasible.

Historically:

- Many classical learning methods (e.g., support vector machines) are designed to yield **convex optimization problems**, guaranteeing a unique global solution.
- Modern deep learning models are **highly non-convex**, yet gradient-based methods often find **good solutions in practice**, especially with good initialization and suitable architectures.

---

### 4.3 Accelerating Gradient Descent

The negative gradient gives the **steepest descent direction locally**, but:

- In high dimensions (and especially with stochastic gradients), directions can be **noisy** and **change abruptly**.
- Naive gradient descent can be **slow** or **oscillatory**.

We want methods that:

- Accelerate convergence.
- Preferably do **not** require:
  - Higher-order derivatives (e.g., Hessians).
  - Multiple evaluations of $f$ per step.

#### 4.3.1 Momentum

View gradient descent as a ball rolling down a landscape:

- Without momentum:
  - The ball updates direction at each step purely based on the **current gradient**, which can lead to zig-zagging.
- With **momentum**:
  - We maintain a running “velocity” that smooths and accumulates gradient directions across steps.

Typical momentum scheme:

- Initialize $g_0 = 0$.
- For $t \ge 1$:
  $$
  g_t = -\eta_t \nabla f(x_{t-1}) + \lambda g_{t-1},
  $$
  $$
  x_t = x_{t-1} + g_t,
  $$
  where:
  - $\eta_t$ is the step size,
  - $\lambda \in [0,1)$ is the **momentum coefficient**.

Unrolling one step:
$$
g_t = -\eta_t \nabla f(x_{t-1})
      + \lambda \big( -\eta_t \nabla f(x_{t-2}) + \lambda g_{t-2} \big)
$$
$$
= -\eta_t \nabla f(x_{t-1}) - \lambda \eta_t \nabla f(x_{t-2}) + \lambda^2 g_{t-2}.
$$

In general:

- The contribution of the gradient from iteration $t-n$ is damped by approximately $\lambda^{n-1}$.
- Momentum therefore performs a kind of **exponential moving average** of past gradients.

Empirically:

- Momentum often **accelerates** convergence and smooths the optimization trajectory.
- It is particularly helpful when gradients oscillate along some directions.

#### 4.3.2 Adaptive Methods and Adam

Another class of methods adjusts **step sizes per parameter** based on the **history of gradients’ magnitudes**.

- Idea: coordinates with consistently large gradients get **smaller effective steps**, and vice versa.

**Adam** is a widely-used optimizer that combines:

- Momentum (first moment estimate).
- Per-coordinate adaptive step sizes (second moment estimate).
- Additional tricks for numerical stability.

Properties:

- Often quite **robust** to hyperparameter choices.
- Default settings in frameworks are a good starting point in many applications.
- Variants like **AdamW** (Adam with decoupled weight decay) are popular.

Open research questions:

- Designing optimizers that can systematically outperform Adam (and variants) across many architectures and tasks remains **ongoing research**.
- Recent work explores using **first-principles reasoning** and **learned optimizers** for neural networks.

#### 4.3.3 Memory Considerations

Acceleration techniques often increase **memory usage**.

- Momentum:
  - Requires storing the **previous update** $g_{t-1}$ (same shape as parameters).
  - Roughly doubles memory usage for the optimizer state (though the **gradient storage** itself can be more dominant in total memory).
- Adaptive methods like Adam:
  - Store additional moving averages (e.g., first and second moments).
  - Increase memory load further.

In many deep learning settings:

- The main memory cost comes from **activations and intermediate states** needed for backprop, not just from optimizer states.
- We return to these trade-offs in more detail later (e.g., in discussions of automatic differentiation and memory in Chapter 6).

---

## 5. From Theory to Practice: NumPy, JAX, PyTorch, and Exercises

The chapter concludes with **practical guidance** rather than traditional end-of-chapter exercises.

### 5.1 Learning Path: Arrays, Tensors, and Frameworks

1. **Start with NumPy:**
   - NumPy provides core functionality for **multi-dimensional arrays** (tensors):
     - Creation, indexing, reshaping, broadcasting, basic linear algebra.
   - You should become comfortable with:
     - Basic array operations.
     - **Indexing and slicing** (crucial for manipulating tensors).
   - There are curated exercise collections (e.g., “numpy-100”) that provide a **gradual challenge** and help solidify understanding.

2. **Move to a realistic deep learning framework:**
   - Limitations of NumPy:
     - Limited direct support for **GPUs** and parallel hardware (without extensions).
     - No built-in **automatic differentiation**.
   - **JAX**:
     - API closely mirrors NumPy (e.g., `jax.numpy`).
     - Adds:
       - **Hardware support** (CPU, GPU, TPU).
       - **Automatic differentiation**.
       - Transformations like **`vmap`** for vectorized computation.
   - **PyTorch**:
     - Also offers a NumPy-like interface (e.g., `torch.tensor`).
     - Adds:
       - Automatic differentiation.
       - Modules like `torch.nn` for **high-level model building**.
   - Recommended:
     - Skim documentation for `jax.numpy.array` and `torch.tensor`.
     - Notice how they **parallel** NumPy concepts.
     - For now, you can **ignore high-level model-building modules**; they come later.

### 5.2 Exercise: Implementing Gradient Descent in NumPy, JAX, and PyTorch

To practice both **mathematics** and **framework usage**, the chapter suggests implementing the same small project in **three frameworks**.

Consider a 2D function $f(x)$ with domain $[0, 10]^2$:
$$
f(x) = \sin(x_1)\cos(x_2) + \sin(0.5 x_1)\cos(0.5 x_2),
$$
where $x = (x_1, x_2)$.

Tasks (to be repeated in NumPy, JAX, and PyTorch):

1. **Vectorized implementation of $f$:**
   - Input: $X \sim (n, 2)$, a batch of $n$ points.
   - Output: $f(X) \sim (n)$, where
     $$
     [f(X)]_i = f(X_i).
     $$

2. **Manual gradient implementation:**
   - Derive and code the **gradient $\nabla f(x)$ by hand**.
   - This is done without using automatic differentiation yet.

3. **Basic gradient descent implementation:**
   - Implement gradient descent over $f$:
     - Choose several **starting points**.
     - Visualize the **optimization paths** (trajectories in 2D space).
   - You should see trajectories moving toward **stationary points** (often local minima due to the multi-modal structure of $f$).

4. **Add momentum and monitor gradient norms:**
   - Extend the algorithm with a **momentum term**.
   - Track and plot the **norm of the gradient** $\|\nabla f(x_t)\|$:
     - It should decrease and tend to **zero** as you approach a stationary point.
   - If using JAX or PyTorch, this is also a good place to experiment with **`vmap`**/**vectorized operations** to handle multiple starting points or batch evaluation.

Meta note from the author:

- This function was (deliberately) generated via an LLM to have “nice” multiple minima and maxima.
- The rest of the book is not LLM-generated; the author explicitly notes this as a kind of transparency.

---

**Overall:**  
This chapter sets up a **shared mathematical language** for the rest of the book:

- **Tensors** as the central data structure.
- **Linear algebra operations** framed in terms of batching and parallel computation.
- **Gradients and Jacobians** as tools for local linearization and directional reasoning.
- **Gradient descent and its variants** as the core optimization mechanism.
- A bridge from theory to implementation via **NumPy, JAX, and PyTorch**, preparing the ground for automatic differentiation and more advanced models in later chapters.