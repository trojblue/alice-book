# Chapter 5: Fully-Connected Models

[TOC]

---

- **Goal**: Introduce multilayer perceptrons (MLPs) / fully-connected (FC) models as differentiable models built by composing linear maps and non-linear activation functions.
- **Key ideas**:
  - Linear models cannot represent important non-linear relationships (e.g., XOR).
  - Composing functions and inserting non-linearities yields powerful approximators.
  - Fully-connected layers and activation functions define the architecture.
  - MLPs are universal approximators under mild conditions.
  - Large-scale training relies on stochastic optimization (SGD with mini-batches).
  - The choice and design of activation functions is both theoretically rich and practically constrained.
  - Modern frameworks (PyTorch, JAX, etc.) operationalize these ideas via layers, datasets, and data loaders.

---

## 1. Fully-Connected Models and MLPs

**Idea**: Build differentiable models by chaining simple blocks:
- **Linear blocks** (matrix multiplications + biases).
- **Elementwise non-linearities** (activation functions).

These chains are called:
- **Fully-connected models** or
- **Multilayer perceptrons (MLPs)**.

An MLP is essentially:
$$
f(x) = f_l \circ f_{l-1} \circ \cdots \circ f_2 \circ f_1(x),
$$
where each $f_i$ is typically:
- an affine transformation $x \mapsto Wx + b$,
- followed by an elementwise non-linearity $\phi$ (except sometimes the last layer).

---

## 2. Limitations of Linear Models

### 2.1 Linear Response to Feature Changes

Consider a linear model **without bias**:
$$
f(x) = w^\top x,
$$
and two input vectors $x$ and $x'$ which are identical except for feature $j$, with:
$$
x'_j = 2x_j, \quad x'_i = x_i \ \text{for } i \neq j.
$$

Then:
$$
f(x') = w^\top x' = w^\top x + w_j (x'_j - x_j) = f(x) + w_j x_j.
$$

So:
- A multiplicative change in a single feature induces only a **linear** change in output, controlled by $w_j$.
- Linear models cannot express context-dependent statements such as:
  - “Income of 1500 is low, **except** if age $< 30$.”
- Such relationships are **non-linear interactions across features**, which linear models cannot capture.

### 2.2 The XOR Dataset and Non-Linear Separability

**Example (XOR)**: Two-dimensional binary input $x \in \{0,1\}^2$ with labels:
- $f([0,0]) = 0$  
- $f([0,1]) = 1$  
- $f([1,0]) = 1$  
- $f([1,1]) = 0$

The output is $1$ **iff exactly one** input is $1$.

Geometrically:
- The positive and negative points are arranged so that **no single linear decision boundary** can perfectly separate the classes.
- We say the dataset is **not linearly separable**.
- Therefore, no linear model can achieve $100\%$ accuracy.

This simple example illustrates the fundamental limitation:
> **Linear models cannot represent non-linearly separable decision boundaries.**

---

## 3. Composition and Hidden Layers

### 3.1 Function Composition as Model Design

We introduce the idea of **decomposition via function composition**.

Let:
$$
f(x) = (f_2 \circ f_1)(x) = f_2(f_1(x)),
$$
where both $f_1$ and $f_2$ are trainable and have their own parameters.

Generalizing:
$$
f(x) = (f_l \circ f_{l-1} \circ \cdots \circ f_2 \circ f_1)(x).
$$

- Each $f_i$ maps vectors to vectors.
- As long as the **type** of the output matches the input of the next layer (e.g., vector $\to$ vector), we can chain **arbitrarily many** such transformations.
- Each $f_i$ contributes its own set of trainable parameters.

This is the formal view of an MLP: a **composition of layers**.

### 3.2 Collapse of Stacked Linear Layers

Suppose we chain two linear maps:
- First layer:
  $$
  h = f_1(x) = W_1 x + b_1
  $$
- Second layer:
  $$
  y = f_2(h) = w_2^\top h + b_2
  $$

Then:
$$
\begin{aligned}
y 
&= w_2^\top (W_1 x + b_1) + b_2 \\
&= (w_2^\top W_1) x + (w_2^\top b_1 + b_2) \\
&= A x + c,
\end{aligned}
$$
where:
- $A = w_2^\top W_1$,
- $c = w_2^\top b_1 + b_2$.

So two linear layers **collapse to a single linear layer**. Stacking only linear transformations does **not** increase expressive power.

---

## 4. Non-Linearities and ReLU

### 4.1 Preventing Collapse with Activation Functions

To avoid collapse, insert an **elementwise non-linearity** $\phi : \mathbb{R} \to \mathbb{R}$ between linear layers:

$$
h = f_1(x) = \phi(W_1 x + b_1),
$$
$$
y = f_2(h) = w_2^\top h + b_2,
$$

or more generally:
$$
y = W_l \, \phi\big(W_{l-1} \, \phi(\dots \phi(W_1 x + b_1) \dots) + b_{l-1}\big) + b_l.
$$

Because $\phi$ is **non-linear**, the overall model can represent non-linear functions even though each linear component is simple.

### 4.2 Rectified Linear Unit (ReLU)

**Definition (ReLU)**  
The **rectified linear unit** is defined elementwise as:
$$
\text{ReLU}(s) = \max(0, s).
$$

Properties:
- Simple and widely used.
- Non-linear, but piecewise linear.
- Zero for negative inputs, identity for positive inputs.

With $\phi = \text{ReLU}$, we can write a depth-$l$ MLP as:
$$
y = W_l \, \phi\big(W_{l-1} \, \phi(\dots \phi(W_1 x + b_1) \dots) + b_{l-1}\big) + b_l.
$$

---

## 5. Fully-Connected Layers and Terminology

### 5.1 Layers, Hidden Layers, Activations, Neurons

Standard neural network terminology:
- Each $f_i$ is a **layer**.
- $f_l$ is the **output layer**.
- $f_1, \dots, f_{l-1}$ are **hidden layers**.
- The input $x$ is sometimes referred to as the **input layer**.
- The output of a layer $f_i(x)$ is called its **activations**.
  - **Pre-activations**: values before $\phi$.
  - **Post-activations**: values after $\phi$.
- The non-linearity $\phi$ is the **activation function**.
- Each scalar output dimension of a layer is (historically) called a **neuron**.

The terminology is somewhat biologically inspired and outdated, but still standard.

### 5.2 Fully-Connected Layer in Batched Form

**Definition (Fully-Connected Layer)**  
Given a batch of $n$ vectors of dimension $c$, arranged as a matrix:
- $X \in \mathbb{R}^{n \times c}$,

a **fully-connected (FC) layer** is:
$$
\mathrm{FC}(X) = \phi(XW + b),
$$
where:
- $W \in \mathbb{R}^{c \times c'}$ is the weight matrix,
- $b \in \mathbb{R}^{c'}$ is the bias vector (broadcast across rows),
- $c'$ is the **width** (output dimension) of the layer.

**Parameters**:
- Total number of trainable parameters: $(c + 1)c'$ (weights + biases).

**Hyperparameters**:
- Output dimension $c'$ (width).
- Choice of non-linearity $\phi$.

### 5.3 Layers as Objects (PyTorch Sketch)

Conceptually, an FC layer (with ReLU) can be implemented as:

```python
class FullyConnectedLayer(nn.Module):
    def __init__(self, c: int, cprime: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(c, cprime))
        self.b = nn.Parameter(torch.randn(1, cprime))

    def forward(self, x):
        return relu(x @ self.W + self.b)