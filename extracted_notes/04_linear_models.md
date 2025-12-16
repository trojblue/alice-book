# Chapter 4 — Linear Models

[TOC]

---

### High-Level Summary

- Set up **supervised learning** with vector inputs $x\in\mathbb{R}^c$, scalar outputs $y$, and a **linear model** plus a loss.
- Introduced the **squared loss** for regression, its **Gaussian likelihood** interpretation, and **robust variants** (absolute, Huber).
- Defined **linear models** $f(x)=w^\top x + b$, wrote **least-squares** in matrix form, and derived:
  - The **gradient** and **normal equations**.
  - The **closed-form solution** via $(X^\top X)^{-1}X^\top y$ and the **pseudoinverse**.
  - A **dual view**: predictions as a weighted average of labels.
- Compared **closed-form least-squares** to **gradient descent**, including convergence and noise variance estimation.
- Discussed **computational complexity**, **conditioning**, and **ridge regression** as a form of **$\ell_2$ regularization** and Bayesian MAP.
- Extended linear models to **classification**:
  - Motivated **one-hot encoding** and the **probability simplex**.
  - Defined **softmax** and **cross-entropy**, yielding **multiclass logistic regression**.
- Specialized to **binary classification**:
  - Introduced the **sigmoid**, **binary cross-entropy**, and links to **generalized linear models** via **log-odds** (logits).
- Explained the **logsumexp trick** for numerically stable cross-entropy implementations.
- Introduced **calibration** (vs. accuracy), **cost-sensitive decisions**, **reliability diagrams**, **ECE**, and **conformal prediction** as a set-valued alternative.
- Closed with a **practical exercise**: implementing linear and logistic regression from scratch with gradient descent.

---

## 1. Supervised Learning Setup and Basic Shapes

### 1.1 Inputs, Outputs, and Tasks

We consider a **supervised learning** problem defined by:

- **Inputs**:  
  - $x \in \mathbb{R}^c$ — a vector of $c$ features (e.g. $c$ personal attributes for a bank client).  
  - The symbol $c$ (“channels”) denotes the **number of features**.

- **Outputs**:
  - **Regression**: $y \in \mathbb{R}$ (unconstrained real-valued output).
  - **Classification**: $y \in \{1,\dots,m\}$ for $m$ classes.
    - If $m=2$: **binary classification**.

- **Model**: a function $f$ (here, a **linear model**).
- **Loss**: a function $\ell(\hat y, y)$ measuring discrepancy between prediction $\hat y=f(x)$ and target $y$.

We will use the **simplest possible** choices:
- Vector inputs.
- Scalar outputs (for regression).
- A **linear** parametric model.
- **Squared loss** (and variants) for regression, and later **cross-entropy** for classification.

Basic shapes (for a dataset of size $n$):

- $n$: number of data points.
- $c$: number of features.
- $m$: number of classes (in classification).

---

## 2. Squared Loss and Variants

### 2.1 Squared Loss for Regression

For regression, the **prediction error** is
$$
e = \hat y - y = f(x) - y .
$$

A common choice of loss is the **squared loss**:

$$
\ell(\hat y, y) = (\hat y - y)^2 .
$$

Properties:

- Nonnegative, differentiable.
- Monotonically decreasing in $|e|$.
- Penalizes **large errors more strongly** (quadratically).

This loss leads to **simple gradients** that are linear in the model’s output, enabling closed-form solutions in linear regression.

---

### 2.2 Probabilistic Interpretation (Gaussian Likelihood)

Using the **maximum likelihood principle**, we can justify squared loss by assuming:

- Conditional on $x$, the target $y$ is drawn from a **Gaussian** with mean $f(x)$ and **constant variance** $\sigma^2$:
  $$
  p(y \mid f(x)) = \mathcal{N}(y \mid f(x), \sigma^2).
  $$

For one data point, the **log-likelihood** is
$$
\log p(y \mid f(x), \sigma^2)
= -\log \sigma - \tfrac{1}{2}\log(2\pi) - \frac{1}{2\sigma^2}(y - f(x))^2 .
$$

- When **minimizing in $f$**, the first two terms are constant.
- The third term is proportional to $(y-f(x))^2$.

Thus, minimizing **negative log-likelihood** is equivalent to minimizing the **squared loss** in $f$.

Once $f$ (or its parameters) is fixed, the **MLE for $\sigma^2$** is:

$$
\sigma^{2*}
= \frac{1}{n} \sum_{i=1}^n (y_i - f(x_i))^2,
$$

which is just the **average squared residual**.

---

### 2.3 Alternatives: Absolute and Huber Loss

The squared loss **over-penalizes outliers**, because large errors are squared.

Two robust alternatives:

1. **Absolute loss**:
   $$
   \ell(\hat y, y) = |\hat y - y|.
   $$
   - Linear in $|e|$: outliers have **less influence**.

2. **Huber loss** (example with threshold $1$):
   $$
   L(y,\hat y) =
   \begin{cases}
   \tfrac12 (y - \hat y)^2, & \text{if } |y - \hat y| \le 1, \\
   |y - \hat y| - \tfrac12, & \text{otherwise}.
   \end{cases}
   $$
   - **Quadratic** near zero error (like squared loss).
   - **Linear** for large errors (like absolute loss).
   - The constant $-1/2$ enforces **continuity** at the transition point.

These losses reduce the influence of **mislabeled** or **extreme** points.

---

### 2.4 Nondifferentiability and Subgradients

The **absolute loss** is not differentiable at $0$ (the cusp of $|e|$), but:

- This is not practically problematic:
  - Gradient descent with random initialization **almost never hits** the exact nondifferentiable point.
- Mathematically, we handle such points using **subgradients**:
  - A subgradient generalizes the concept of a derivative to convex but nondifferentiable functions.

Conclusion: having a **small number** of nondifferentiability points (like for $|e|$ at $0$) is acceptable in optimization.

---

## 3. Least-Squares Linear Regression

### 3.1 Definition and Geometry of Linear Models

**Definition (Linear model).**  
A linear model on input $x\in\mathbb{R}^c$ is
$$
f(x) = w^\top x + b,
$$
where:
- $w \in \mathbb{R}^c$ is the **weight vector** (trainable).
- $b \in \mathbb{R}$ is the **bias** (trainable).

Intuition:

- Each feature $x_i$ is assigned a fixed weight $w_i$.
- The model predicts a **weighted sum** of features plus a bias:
  - If $x=0$, prediction is the constant $b$.

Geometry:

- For **1D input** ($c=1$), $f(x)$ is a **line**.
- For **2D input** ($c=2$), $f(x)$ is a **plane**.
- In general, $f$ defines a **hyperplane** in $\mathbb{R}^{c+1}$.

Bias absorption trick:

- Define an **augmented feature vector**:
  $$
  \tilde x = \begin{bmatrix} x \\ 1 \end{bmatrix} \in \mathbb{R}^{c+1}, \quad
  \tilde w = \begin{bmatrix} w \\ b \end{bmatrix}.
  $$
- Then $f(x) = \tilde w^\top \tilde x$; we can treat $b$ as **part of $w$**.

---

### 3.2 Least-Squares Objective (Empirical Risk)

**Definition (Least-squares problem).**  
Given data $\{(x_i,y_i)\}_{i=1}^n$, the **least-squares** objective is:
$$
\min_{w,b}\ \frac{1}{n} \sum_{i=1}^n \bigl(y_i - w^\top x_i - b\bigr)^2.
$$

We often write the optimal solution as:
$$
(w^*, b^*) = \arg\min_{w,b} \frac{1}{n} \sum_{i=1}^n \bigl(y_i - w^\top x_i - b\bigr)^2.
$$

This is an instance of **empirical risk minimization** with squared loss and linear model.

---

### 3.3 Vectorized Form and Batched Output

Collect inputs and outputs into arrays:

- **Design matrix** $X \in \mathbb{R}^{n\times c}$:
  $$
  X =
  \begin{bmatrix}
    x_1^\top \\
    \vdots \\
    x_n^\top
  \end{bmatrix}.
  $$
- **Target vector** $y \in \mathbb{R}^n$:
  $$
  y = [y_1,\dots,y_n]^\top.
  $$

For a batch of inputs $X$, the **batched model output** is:
$$
f(X) = Xw + \mathbf{1}b,
$$
where $\mathbf{1}\in\mathbb{R}^n$ is the all-ones vector (same bias for all examples).

Vectorized **least-squares loss**:
$$
\text{LS}(w,b)
= \frac{1}{n}\, \|y - Xw - \mathbf{1}b\|_2^2 .
$$

Notes:

- Modern frameworks (NumPy, PyTorch, JAX) are optimized around **matrix operations**; equations like $Xw + \mathbf{1}b$ map almost directly to code (e.g. `X @ w + b`).
- **Permutation equivariance**:
  - If we permute the rows of $X$ and $y$ in the same way, the vector $f(X)$ is permuted accordingly.
  - The model’s behavior depends only on the **set** of data points, not their order.

---

### 3.4 Gradient, Convexity, and Normal Equations

Ignoring the bias (absorbed into $w$ via augmentation) and constants, the loss is:
$$
\text{LS}(w) = \frac{1}{n} \|y - Xw\|_2^2.
$$

**Gradient** with respect to $w$:
$$
\nabla \text{LS}(w) = \frac{2}{n} X^\top (Xw - y).
$$
(Up to constant factors, the key structure is $X^\top(Xw - y)$.)

- The objective is a **quadratic function** in $w$.
- Thus, it is **convex**; LS has no spurious local minima.

Setting the gradient to zero gives the **normal equations**:
$$
X^\top (Xw - y) = 0
\quad\Longleftrightarrow\quad
X^\top X\,w = X^\top y.
$$

This is a **linear system** in $w$:
- Let $A = X^\top X$, $b = X^\top y$.
- Then $Aw = b$.

---

### 3.5 Closed-Form Solution and Pseudoinverse

If $X^\top X$ is **invertible**, the unique solution is:
$$
w^* = (X^\top X)^{-1} X^\top y.
$$

The corresponding **bias** $b^*$ can also be solved in closed form if we keep it separate, or folded into $w^*$ via the augmented representation.

**Pseudoinverse:**

- The matrix
  $$
  X^\dagger = (X^\top X)^{-1}X^\top
  $$
  is the (Moore–Penrose) **pseudoinverse** of $X$ (in the full-rank case).
- It satisfies $X^\dagger X = I$ (on the column space).

Thus:
$$
w^* = X^\dagger y.
$$

**Collinearity and rank deficiency:**

- If some feature is a **scalar multiple** of another, $X$ does not have full column rank.
- Then $X^\top X$ is **singular**, and $(X^\top X)^{-1}$ does not exist.
- This is called **collinearity**; special handling or regularization is needed.

---

### 3.6 Dual View: Least-Squares as Weighted Average of Labels

Predictions under the LS solution:
$$
\hat y = X w^* = X X^\dagger y = M y,
$$
where $M = X X^\dagger$.

Interpretation:

- Each prediction is a **linear combination of training labels**:
  - The matrix $M$ acts like a **projection** onto the column space induced by $X$.
- This is sometimes called the **dual formulation** of least-squares:
  - You can see which **training inputs** are most influential for a prediction by inspecting the corresponding dual weights.

---

### 3.7 Gradient Descent Solution and Convergence

Instead of using the closed form, we can use **gradient descent**:

- Update rule (again ignoring constants):
  $$
  w_{t+1}
  = w_t - \eta \,\nabla \text{LS}(w_t)
  = w_t - \eta\, X^\top (Xw_t - y).
  $$
  (Sign conventions may differ depending on whether we write $y - Xw$ or $Xw - y$.)

- For a **small learning rate** $\eta$, each step gives a **stable decrease** in the loss until convergence.

Convergence check (example):

- Stop when the change in parameters is small:
  $$
  \|w_{t+1} - w_t\|_2 < \varepsilon,
  $$
  for some small threshold $\varepsilon > 0$.

Even though we **can** solve least-squares in closed form, gradient descent is:

- A natural **template** for more complex models without closed form.
- Amenable to **mini-batching**, online learning, and large-scale settings.

---

### 3.8 Estimating Noise Variance After Fitting

Returning to the Gaussian log-likelihood:
$$
\log p(y \mid f(x), \sigma^2)
= -\log \sigma - \tfrac{1}{2}\log(2\pi) - \frac{1}{2\sigma^2}(y - f(x))^2,
$$

we can **optimize $\sigma^2$** after fitting $w^*$:
$$
\sigma^{2*}
= \frac{1}{n} \sum_{i=1}^n \bigl(y_i - w^{*\top} x_i\bigr)^2.
$$

Interpretation:

- The variance $\sigma^2$ is **constant** (by model assumption).
- It is estimated as the **average squared prediction error** on the training data.

---

### 3.9 Computational Cost and Conditioning

Closed-form solution involves:

- Computing $X^\top X$ (size $c\times c$).
- Inverting $X^\top X$.
- Multiplying by $X^\top y$.

Approximate costs:

- **Matrix inversion** $(X^\top X)^{-1}$: $\mathcal{O}(c^3)$.
- Matrix products like $X^\top X$ or $(X^\top X)w$: $\mathcal{O}(c^2 n)$.

Issues:

- For large $c$ or $n$, these operations may be **prohibitively expensive**.
- The quality of inversion depends on the **condition number**:
  - For a matrix $A$,
    $$
    \kappa(A) = \|A\|\,\|A^{-1}\|
    $$
    (for some matrix norm).
  - Large $\kappa(A)$ implies **numerical instability** in inversion.

In contrast, gradient descent:

- If we compute $Xw$ first:
  - $Xw$: $\mathcal{O}(nc)$.
  - $X^\top (Xw - y)$: $\mathcal{O}(nc)$.
- Thus can be **linear** in both $n$ and $c$ if implemented carefully.

This is a small example of why **efficient ordering of matrix multiplications** matters, and an entry point to **reverse-mode automatic differentiation** (backpropagation).

---

### 3.10 Regularized Least-Squares (Ridge Regression)

To stabilize the closed-form solution when $X^\top X$ is nearly singular, we can **regularize**:

- Add a small multiple $\lambda>0$ of the identity:
  $$
  w^*_\lambda
  = (X^\top X + \lambda I)^{-1} X^\top y.
  $$

This has two interpretations:

1. **Numerical**:
   - $X^\top X + \lambda I$ is “more diagonal” and typically has a **better condition number**.

2. **Optimization / regularization**:
   - The solution is the minimizer of:
     $$
     \text{LS-Reg}(w)
       = \frac{1}{n}\|y - Xw\|_2^2
       + \frac{\lambda}{2}\|w\|_2^2.
     $$
   - The extra term is **$\ell_2$-regularization** on the weights.

3. **Bayesian (MAP) view**:
   - If we place a **Gaussian prior** on weights:
     $$
     w \sim \mathcal{N}(0, \tau^2 I),
     $$
     then the **MAP (maximum a posteriori)** solution under Gaussian noise and this prior is exactly **ridge regression**.

Regularization expresses a **preference** for small-norm weights, controlled by the **hyperparameter** $\lambda$.

---

## 4. Linear Models for Multi-Class Classification

We now switch from regression to **classification**.

- Targets: $y_i \in \{1,\dots,m\}$ for $m$ classes.
- Applications:
  - Computer vision (**image classification**).
  - NLP (**next-token prediction**), and more.

We want **linear models + differentiable transformations** for this setting.

---

### 4.1 Why Not Regress Directly on Class Indices?

A naive approach:

- Treat $y \in \{1,\dots,m\}$ as an integer and **regress** on it.
- Option 1: direct regression on $y$ (bad gradients because of integer outputs).
- Option 2: regress on a real target $\tilde y \in [1,m]$ and at inference:
  $$
  \text{Predicted class} = \operatorname{round}(\hat y).
  $$

Problems:

- Introduces an **artificial ordering** of classes:
  - Class 2 is “closer” to class 3 than to class 4 in the numeric scale,
    but there is no such natural ordering in many tasks.
- The model may exploit this spurious geometry, which is **undesirable**.

---

### 4.2 One-Hot Encoding

Instead, we represent the class as a **one-hot** vector $y^{\text{oh}}\in\{0,1\}^m$:

$$
[y^{\text{oh}}]_j =
\begin{cases}
1, & \text{if } y = j, \\
0, & \text{otherwise}.
\end{cases}
$$

Example for $m=3$:

- Class 1: $y^{\text{oh}} = [1,0,0]^\top$.
- Class 2: $y^{\text{oh}} = [0,1,0]^\top$.
- Class 3: $y^{\text{oh}} = [0,0,1]^\top$.

Properties:

- One-hot vectors are **unordered**.
- The **Euclidean distance** between any two distinct one-hot vectors is $\sqrt{2}$.
  - Same class: distance $0$.
  - Different classes: distance $\sqrt{2}$.

We could regress directly to $y^{\text{oh}}$ using squared loss (**Brier score**), but a more elegant approach is **logistic regression** with softmax + cross-entropy.

---

### 4.3 Probability Simplex

**Definition (Probability simplex).**  
The **probability simplex** $\Delta^m$ is:
$$
\Delta^m = \left\{ x \in \mathbb{R}^m \,\middle|\,
x_i \ge 0,\ \sum_{i=1}^m x_i = 1 \right\}.
$$

Interpretation:

- $\Delta^m$ is the set of all **probability distributions** over $m$ outcomes.
- Geometrically:
  - One-hot vectors are the **vertices** of a polytope.
  - $\Delta^m$ is the **convex hull** of these vertices.

Given $x\in\Delta^m$, we can recover a **predicted class** by:
$$
\arg\max_i \{x_i\},
$$
which corresponds to the **mode** of the distribution.

---

### 4.4 Linear Logits and the Softmax Function

We first build a **vector-valued linear model**:
$$
h = Wx + b,
$$
where:
- $W \in \mathbb{R}^{m\times c}$ (one linear model per class).
- $b \in \mathbb{R}^m$.
- $h \in \mathbb{R}^m$ are the **logits** (pre-normalized outputs).

This $h$ does **not** lie in the simplex. We map it into $\Delta^m$ using **softmax**.

**Definition (Softmax function).**  
For $x\in\mathbb{R}^m$,
$$
[\operatorname{softmax}(x)]_i
= \frac{\exp(x_i)}{\sum_{j=1}^m \exp(x_j)}.
$$

Decomposed:

1. Exponentiation:
   $$
   h_i = \exp(x_i).
   $$
2. Normalization factor:
   $$
   Z = \sum_{j=1}^m h_j = \sum_{j=1}^m \exp(x_j).
   $$
3. Output:
   $$
   y_i = \frac{h_i}{Z}.
   $$

Thus $\operatorname{softmax}(x)\in\Delta^m$.

---

### 4.5 Temperature and Softmax as Soft Argmax

We can define a **temperature-scaled softmax**:
$$
\operatorname{softmax}(x;\tau)
= \operatorname{softmax}\!\left(\frac{x}{\tau}\right),\quad \tau>0.
$$

Properties:

- Softmax preserves the **ordering** of $x_i$.
- As $\tau$ changes, it modifies **confidence**:

Limits:

- As $\tau \to \infty$:
  $$
  \operatorname{softmax}(x;\tau)
  \to \frac{1}{m}\mathbf{1}
  $$
  (uniform distribution).
- As $\tau \to 0$:
  $$
  \operatorname{softmax}(x;\tau) \to \text{one-hot at } \arg\max_i x_i
  $$
  (non-differentiable argmax limit).

Thus softmax is a **differentiable approximation** to the argmax; it could be called a **soft-argmax**.

---

### 4.6 Logistic Regression Objective (Multiclass)

Combining:

- Linear logits: $h = Wx + b$.
- Softmax: $\hat y = \operatorname{softmax}(h)$.

We interpret $\hat y$ as **class probabilities**:
$$
\hat y = \operatorname{softmax}(Wx + b) \in \Delta^m.
$$

We then model the one-hot targets $y^{\text{oh}}$ via a **categorical distribution** with parameter $\hat y$:
$$
p(y^{\text{oh}} \mid \hat y)
= \prod_{i=1}^{m} \hat y_i^{[y^{\text{oh}}]_i},
$$
where only one exponent is $1$ (for the true class), all others $0$.

---

### 4.7 Cross-Entropy Loss

**Definition (Cross-entropy loss).**  
For one-hot $y^{\text{oh}}$ and prediction $\hat y$:
$$
\operatorname{CE}(y^{\text{oh}}, \hat y)
= - \sum_{i=1}^{m} [y^{\text{oh}}]_i \log \hat y_i.
$$

Because exactly one entry of $y^{\text{oh}}$ equals $1$ (say at index $y$), this simplifies to:
$$
\operatorname{CE}(y, \hat y)
= -\log \hat y_y,
$$
where $\hat y_y$ is the predicted probability of the **true class** $y$.

Interpretation:

- Minimizing CE encourages the model to assign **high probability** to the true class.
- Due to softmax normalization, **increasing one class probability** necessarily **decreases** others.

**Logistic regression objective (dataset):**
$$
\operatorname{LR}(W,b)
= \frac{1}{n} \sum_{i=1}^n
\operatorname{CE}\bigl(
y_i^{\text{oh}},
\operatorname{softmax}(W x_i + b)
\bigr).
$$

No closed-form solution exists; we use **gradient descent** (or variants like SGD).

---

## 5. Binary Classification and Sigmoid Logistic Regression

Consider now the special case $m=2$:

- Classes: $y \in \{0,1\}$.
- Often called **binary classification** or **concept learning**.

---

### 5.1 Redundancy of Second Output

In softmax with two outputs $h_1,h_2$:

$$
\hat y_1
= \frac{\exp(h_1)}{\exp(h_1) + \exp(h_2)},\quad
\hat y_2
= \frac{\exp(h_2)}{\exp(h_1) + \exp(h_2)}.
$$

Since $\hat y_1 + \hat y_2 = 1$, the second output is **redundant**:
- Once we know $\hat y_1$, we have $\hat y_2 = 1 - \hat y_1$.

Thus we can use a **single scalar output** $f(x)\in[0,1]$:

- Interpretation: $f(x) \approx p(y=1 \mid x)$.
- Prediction:
  $$
  \text{Predicted class}
  = \operatorname{round}(f(x))
  =
  \begin{cases}
  0, & f(x)\le 0.5,\\
  1, & f(x) > 0.5.
  \end{cases}
  $$

---

### 5.2 Sigmoid Function

The two-class softmax can be rewritten as a **sigmoid** transformation of a single logit.

**Definition (Sigmoid function).**  
The sigmoid $\sigma:\mathbb{R}\to[0,1]$ is
$$
\sigma(s)
= \frac{1}{1 + \exp(-s)}.
$$

Properties:

- Maps any real $s$ into $(0,1)$.
- $\sigma(0) = 1/2$.
- Saturates near $0$ and $1$ for large negative/positive $s$.

**Binary logistic regression model:**
$$
f(x) = \sigma(w^\top x + b).
$$

We interpret $f(x)$ as the **predicted probability** for class $1$.

---

### 5.3 Binary Cross-Entropy and Gradient

For a binary label $y\in\{0,1\}$ and prediction $\hat y=f(x)$, the **binary cross-entropy** is:
$$
\operatorname{CE}(\hat y, y)
= -y \log \hat y - (1-y)\log(1 - \hat y).
$$

- If $y=1$: loss is $-\log \hat y$.
- If $y=0$: loss is $-\log(1-\hat y)$.

Gradient with respect to $w$:

- With $f(x) = \sigma(w^\top x + b)$, one can show:
  $$
  \nabla_w \operatorname{CE}(f(x), y)
  = (f(x) - y)\,x.
  $$

Similarity to linear regression:

- For least-squares regression with scalar output $\hat y$ and target $y$, the gradient also has the form $(\hat y - y)x$ (up to constants).
- This parallel is important: **classification and regression** share similar gradient structures when using linear models + appropriate nonlinearity.

---

### 5.4 Logits, Log-Odds, and Generalized Linear Models

The **logit** is the inverse of the sigmoid in probabilistic terms.

Let $p = f(x) = \sigma(w^\top x + b)$. Then:
$$
w^\top x + b
= \log\frac{p}{1-p}.
$$

Interpretation:

- The linear expression $w^\top x + b$ is the **log-odds** of class $1$:
  $$
  \log\frac{\Pr(y=1\mid x)}{\Pr(y=0\mid x)}.
  $$
- Thus logistic regression is a **linear model in log-odds space** plus a nonlinear mapping (sigmoid) back to probabilities.

This is a special case of **generalized linear models (GLMs)**:

- Linear predictor: $w^\top x + b$.
- Inverse link function: $\sigma^{-1}$ (logit).

The term **logits** used for $Wx+b$ in multiclass settings originates from this log-odds interpretation.

---

## 6. Numerical Stability: The Logsumexp Trick

Implementations of cross-entropy often accept **logits** directly instead of softmax-normalized outputs. Reason: **numerical stability**.

Consider the per-example cross-entropy term (for class $y$) in terms of logits $p$:
$$
-\log \frac{\exp(p_y)}{\sum_j \exp(p_j)}.
$$

Rewrite:
$$
-\log \frac{\exp(p_y)}{\sum_j \exp(p_j)}
= -p_y + \log\left(\sum_j \exp(p_j)\right).
$$

Define the **logsumexp** function:
$$
\operatorname{logsumexp}(p)
= \log\left(\sum_j \exp(p_j)\right).
$$

Issue:

- For large $|p_j|$, $\exp(p_j)$ can **overflow** or underflow.

Stability trick:

- For any scalar $c$,
  $$
  \operatorname{logsumexp}(p)
  = \operatorname{logsumexp}(p - c) + c.
  $$
- Choosing $c = \max_j p_j$ ensures all $(p_j - c)\le 0$, preventing $\exp$ from blowing up.

Important identity:

- The **gradient** of logsumexp is exactly the **softmax**:
  $$
  \nabla_p \operatorname{logsumexp}(p) = \operatorname{softmax}(p).
  $$

Therefore:

- Frameworks (PyTorch, TensorFlow, etc.) often provide:
  - Cross-entropy implementations that take **logits** and internally perform logsumexp with this trick.
  - This avoids separate explicit softmax, reducing both **numerical issues** and sometimes **computational cost**.

Conceptually, softmax may be seen either as:

- Part of the **model** (logits $\to$ probabilities), or
- Part of the **loss**, implemented via logsumexp of logits.

---

## 7. Calibration and Classification

Even with softmax outputs, there is a subtle but crucial distinction:

- **Predicted class**:
  $$
  \hat y(x) = \arg\max_i [f(x)]_i.
  $$
- **Predicted probability**:
  $$
  [f(x)]_i \overset{?}{\approx} \Pr(y=i\mid x).
  $$

Training with cross-entropy primarily focuses on **getting the class right**, not necessarily on making confidence values numerically equal to true probabilities.

---

### 7.1 Definition of Calibration

**Definition (Calibration).**  
A classifier $f(x)$ producing class probabilities is **calibrated** if, for each class $i$ and each confidence level $p$,

> Among all predictions where $[f(x)]_i \approx p$, the fraction that truly belong to class $i$ is also $\approx p$.

Formally:
$$
[f(x)]_i = \Pr(y=i \mid x),
$$
for all $x$ and $i$ (in an idealized sense).

---

### 7.2 Accuracy vs Calibration: Examples

Two thought experiments:

1. **Perfect accuracy, underconfident**:
   - A binary classifier outputs the **correct class always**, but with probability $0.8$ instead of $1.0$.
   - Accuracy: **100%**.
   - Calibration: off — the model is **underconfident**.

2. **Perfect calibration, poor accuracy**:
   - A 4-class problem with balanced classes; the model always predicts $[0.25,0.25,0.25,0.25]$.
   - Calibration: **perfect** (predictions match the empirical class frequencies).
   - Accuracy: **25%** (random guessing).

Conclusion: Accuracy and calibration are **distinct properties**.

---

### 7.3 Cost-Sensitive Decisions and Decision Theory

Calibration becomes crucial when different misclassifications have **different costs**.

Let $C\in\mathbb{R}^{m\times m}$ be a **cost matrix**:

- $C_{ij}$: cost of predicting class $j$ when the true class is $i$.

Example (binary, classes $0$ and $1$):

|        | True 0 | True 1 |
| ------ | ------ | ------ |
| Pred 0 | 0      | 10     |
| Pred 1 | 1      | 0      |

Interpretation:

- Correct predictions: zero cost.
- False positive (predict 1, true 0): cost $1$.
- False negative (predict 0, true 1): cost $10$ (much worse).

Given a calibrated probability vector $f(x)$, the **expected cost** of predicting class $i$ is:
$$
\text{EC}(i \mid x)
= \sum_{j=1}^m C_{ij} [f(x)]_j.
$$

The **rational decision** is:
$$
\hat y(x)
= \arg\min_i \text{EC}(i \mid x)
= \arg\min_i \sum_{j=1}^m C_{ij} [f(x)]_j.
$$

Special case:

- If $C_{ij} = 1$ for $i\ne j$ and $0$ for $i=j$, this reduces to predicting
  $$
  \arg\max_j [f(x)]_j,
  $$
  i.e. standard **argmax**.

This viewpoint connects classification to **decision theory**.

---

### 7.4 Measuring Calibration: Reliability Diagrams and ECE

To estimate calibration empirically:

1. Take a **validation set** of size $n$.
2. For each prediction, record:
   - The **predicted class** $\hat y_i$.
   - The **confidence** $p_i = \max_j [f(x_i)]_j$.
   - Whether the prediction is **correct**.

3. Partition the interval $[0,1]$ into $b$ **bins** of equal width.
   - For bin $k$, define:
     - $\mathcal{B}_k$: set of indices $i$ whose confidence $p_i$ falls in bin $k$.
     - Average confidence:
       $$
       p_k = \frac{1}{|\mathcal{B}_k|} \sum_{i\in\mathcal{B}_k} p_i.
       $$
     - Accuracy in bin:
       $$
       a_k = \frac{1}{|\mathcal{B}_k|} \sum_{i\in\mathcal{B}_k} \mathbb{1}[\hat y_i = y_i].
       $$

A **reliability diagram** plots $a_k$ vs. $p_k$, often with bars showing **miscalibration** $|a_k - p_k|$.

**Expected Calibration Error (ECE).**
$$
\operatorname{ECE}
= \sum_{k=1}^b \frac{|\mathcal{B}_k|}{n} \,\bigl|a_k - p_k\bigr|.
$$

- ECE is a single scalar summarizing calibration quality.
- Other calibrations measures exist (e.g. maximum bin error).

---

### 7.5 Fixing Calibration and Conformal Prediction

If a model is **uncalibrated**, we can:

- Apply **post-hoc calibration** methods, such as:
  - **Temperature scaling**: rescale logits by a scalar temperature before softmax.
  - Alternative training losses (e.g. **focal loss**) that adjust emphasis on hard/easy examples.

**Conformal prediction** is a different line of work:

- Instead of outputting a single class, output a **set of plausible classes**.

Given a threshold $\gamma$, define:
$$
\mathcal{S}(x)
= \{ i \mid [f(x)]_i > \gamma \}.
$$

- So the model returns all classes with probability above $\gamma$.

We want to choose $\gamma$ such that:
$$
\Pr(y \in \mathcal{S}(x)) \ge 1 - \alpha,
$$
for some user-specified error level $\alpha$.

- Smaller $\gamma$ $\Rightarrow$ larger sets $\mathcal{S}(x)$ and smaller error.
- In the extreme, $\gamma=0$ gives $\mathcal{S}(x)=\{1,\dots,m\}$, trivially satisfying the bound but useless.

Conformal methods provide a **data-driven** way to pick $\gamma$ (or equivalent thresholds) with **guaranteed coverage**, at the cost of **set-valued** predictions rather than single labels.

---

## 8. Practical Exercise: From Theory to Code

To connect these ideas to practice, a suggested exercise:

1. **Load a toy dataset**  
   - Use, for example, a dataset from scikit-learn’s toy datasets (classification or regression).

2. **Build a linear model**  
   - For regression: $f(x) = w^\top x + b$.  
   - For classification: logistic regression with either softmax (multiclass) or sigmoid (binary).
   - Write code in a **modular** way:
     - One function to **initialize** parameters.
     - One function to compute **predictions** given parameters and inputs.

3. **Train via gradient descent**  
   - Compute gradients **manually** (not via autograd) for practice:
     - For least-squares: gradient $X^\top(Xw - y)$.
     - For logistic regression: gradients involving $(f(x)-y)x$.
   - Make the implementation flexible:
     - E.g. easily toggle inclusion of a bias term or regularization.

4. **Evaluate and compare**  
   - Plot the **training loss** over iterations.
   - Evaluate **accuracy** (classification) or **MSE** (regression) on a test set.
   - Optionally compare with scikit-learn models such as:
     - Decision trees.
     - $k$-nearest neighbors.

This exercise reinforces:

- The **link** between algebraic derivations and actual code.
- How **linear models**, **losses**, and **gradients** interact in practice.
- The importance of **vectorization** and **good numerical practices**.

---