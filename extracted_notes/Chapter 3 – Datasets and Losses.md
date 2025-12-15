# Chapter 3 – Datasets and Losses (Summary)

[TOC]

---

- Formalizes supervised learning in terms of datasets, losses, and empirical risk minimization (ERM).  
- Explains the i.i.d. assumption, domain shift, feedback loops, and why they matter.  
- Surveys variants around supervised learning: unsupervised, self-supervised, foundation models, fine-tuning, semi-supervised, and other extensions.  
- Introduces **loss functions**, ERM, expected risk, overfitting, and the **generalization gap**.  
- Reframes learning probabilistically via **maximum likelihood**, showing how common losses arise as negative log-likelihoods.  
- Extends to **Bayesian learning**, with priors over models, posteriors, MAP estimation, and sequential updating.

---

## 1. Supervised Datasets and the i.i.d. Assumption

### 1.1 Supervised Dataset

**Definition (Dataset – D.3.1)**  
A **supervised dataset** of size $n$ is a collection
$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n,
$$
where each pair $(x_i, y_i)$ is an example of the input–output relationship we want to model.

We assume:

- Each $(x_i, y_i)$ is drawn **independently** and **identically** distributed (i.i.d.)  
- From some unknown joint distribution $p(x, y)$.

This distribution is **unknown and effectively unknowable**; we only see finite samples.

---

### 1.2 Interpreting the i.i.d. Assumption

The i.i.d. assumption encodes two key requirements:

1. **Identically distributed** (same distribution):
   - The data-generating mechanism is “stable enough” over the time and conditions under which we collect data.
   - Example: recognizing car models from photos:
     - If we collect images over a short time span in a given region, the distribution of car models is relatively stable.
     - If we mix images from many decades, the car models change substantially; training on the past may not generalize to current cars.

2. **Independently distributed** (no correlations from collection bias):
   - Data collection should not systematically favor some regions of the input space.
   - Example: collecting car photos only near a Tesla dealership:
     - The dataset over-represents Teslas and under-represents other brands.
     - The resulting model is biased and not representative of $p(x, y)$.

**Context dependence:**  
Whether data is i.i.d. depends on the **deployment context**:

- A car dataset collected in Italy might be valid for deployment in Rome or Milan.
- It may be invalid for deployment in Tokyo or Taiwan, because the car distribution is different.

**Large-scale LLMs:**  
Modern large language models are trained on enormous heterogeneous datasets:

- The boundary between what is “in-distribution” vs “out-of-distribution” becomes blurred.
- Assessing which tasks are truly covered by the training distribution and how far the model generalizes is non-trivial.

---

### 1.3 Maintaining i.i.d. Over Time: Domain Shift and Feedback Loops

Ensuring i.i.d. is **not a one-time check**. It must be monitored over the entire lifetime of a model.

- **Domain shift (distribution shift):**  
  - Example: car classification over time.  
  - If the distribution of car models gradually changes and we do nothing, performance deteriorates.
- **Feedback loops in recommender systems:**
  - A recommender system influences user behavior.
  - Users start reacting to the model’s recommendations, changing the data distribution.
  - This creates feedback loops that can amplify biases and require **continuous re-evaluation** of both the system and the application.

The i.i.d. assumption is thus both a **modeling assumption** and an **engineering responsibility**: it must be checked, monitored, and maintained.

---

## 2. Variants of Supervised Learning

### 2.1 Unsupervised Learning

If some or all targets $y_i$ are **not available**, we move outside standard supervised learning.

- **Unsupervised learning:** no explicit labels $y_i$.
  - Typical task: **clustering** – group data so that:
    - Points within a cluster are similar.
    - Points across clusters are dissimilar.
  - Used when we only want structure from $x$’s, not explicit predictions of $y$.

Another example is **retrieval**:

- Given a query (e.g., an image, a text snippet), search a large database for the **top-$k$ most similar items**.
- This is non-trivial when similarity in the raw input space (e.g., pixel space) is poorly behaved.

---

### 2.2 Representations and Embeddings from Differentiable Models

Distances in raw input space (e.g., images as pixel arrays) are often **not meaningful**:

- Tiny semantic changes can alter many pixels.
- Euclidean distance on pixels does not align with semantic similarity.

Instead, we use **differentiable models** to obtain internal representations:

- Consider a pre-trained model $g$ (e.g., an image classifier) that has been trained on a large dataset.
- For an input $x$, its internal state (some intermediate layer) is a vector $z = g_{\text{embed}}(x)$ in a **latent space**.

Empirically:

- Semantically similar inputs are mapped to nearby points in this latent space.
- The latent space is **metric-friendly**: vectors can be added, distances computed, and points ranked.

Thus we can:

- Use these embeddings $z$ as inputs to:
  - Clustering algorithms (e.g., Gaussian mixture models).
  - Retrieval systems (nearest neighbors).
- This is often much more effective than working directly in the raw input space.

---

### 2.3 Self-Supervised Learning (SSL)

When we **do not have labels** but want learned representations useful for many downstream tasks, we can construct **self-supervised** objectives.

**Self-supervised learning (SSL):**

- We automatically formulate a supervised objective from unlabeled data.
- Train a model to solve this self-generated task.
- The resulting representation is then used for many downstream tasks.

Examples:

- Large text corpora:
  - Task: predict the next token or fill in missing tokens in a sequence.
  - Model learns representations of words, phrases, sentences that are useful far beyond the pretext task.
- SSL has shown that neural networks can learn powerful **embeddings of text** (and other modalities) from purely unlabeled data.

**Foundation models and LLMs:**

- Large language models (LLMs) such as GPT or Llama are trained via a **self-supervised next-token prediction objective**.
- These models are sometimes called **foundation models** because:
  - They serve as a base for many downstream tasks.
  - They are often used “as is,” or with light adaptation, across diverse applications.

---

### 2.4 Zero-shot, Few-shot, and Fine-tuning

Given a pre-trained foundation model, we can use it in different modes:

1. **Zero-shot usage:**
   - Provide only a prompt or question; the model directly performs the new task.
   - Requires strong generalization abilities internalized during pre-training.

2. **Few-shot prompting:**
   - Provide a prompt **plus a few examples** of the desired input–output behavior.
   - The model adapts its behavior *in-context* without parameter updates (often referred to as **in-context learning**).

3. **Fine-tuning:**
   - Use a (typically smaller) labeled dataset for a specific task.
   - Continue training the model parameters via gradient descent.
   - Conceptually similar to training from scratch, but starting from a powerful initialization.

Fine-tuning is made easier by:

- Large open-source model repositories (e.g., model hubs).
- Tools and libraries that standardize model formats and training procedures.

**Parameter-Efficient Fine-Tuning (PEFT):**

- Instead of updating all parameters, update only:
  - A subset of parameters, or
  - A small set of **additional** parameters attached to the base model.
- This reduces computational and memory costs, and is especially useful for very large models.

---

### 2.5 Semi-supervised and Other Advanced Scenarios

Beyond pure supervised/unsupervised:

- **Semi-supervised learning:**
  - Only part of the dataset is labeled.
  - Combine supervised loss on labeled data with unsupervised objectives on unlabeled data.
- **Multiple datasets / multiple domains:**
  - Data coming from similar but not identical distributions.
  - Data arriving at different times.
  - Gives rise to:
    - Domain adaptation and domain generalization.
    - Meta-learning.
    - Continual / lifelong learning.
    - Metric learning, unlearning, and others.

The chapter only hints at these; many are covered in more detail in later volumes.

---

## 3. Loss Functions and Empirical Risk Minimization

### 3.1 Loss Function

**Definition (Loss function – D.3.2)**  
Given:

- Target $y$,
- Predicted value $\hat y = f(x)$ from a model $f$,

a **loss function** is a scalar, differentiable function
$$
l(y, \hat y) \in \mathbb{R}
$$
such that:

- A lower value of $l(y, \hat y)$ corresponds to **better** performance.
- For instance, if $l(y, \hat y_1) < l(y, \hat y_2)$, then $\hat y_1$ is considered a better prediction than $\hat y_2$ for the same $y$.

The loss function:

- Encodes our **understanding of the task** and our **preferences** among possible solutions.
- Translates performance into a **real-valued quantity** that can be optimized.

---

### 3.2 Empirical Risk Minimization (ERM)

Given a dataset
$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n
$$
and a loss function $l$, a natural objective is to minimize the **average loss** on the dataset:
$$
f^* = \arg\min_f \frac{1}{n}\sum_{i=1}^n l\bigl(y_i, f(x_i)\bigr).
$$
This is the **empirical risk minimization (ERM)** problem.

In practice, models are parameterized by tensors (weights) $w$:

- We write $f(x; w)$ or $f(x, w)$.
- Training becomes:
  $$
  w^* = \arg\min_w \frac{1}{n}\sum_{i=1}^n l\bigl(y_i, f(x_i, w)\bigr). 
  $$
- We usually solve this approximately using (stochastic) **gradient descent** or its variants.

Historically, “risk” is used as a synonym for “loss” in this context.

---

### 3.3 Why Differentiability Matters: 0/1 Loss vs Surrogate Losses

For optimization via gradient descent, we need a loss with **useful gradients**.

Consider a simple **binary classification** task:

- Targets: $y \in \{-1, +1\}$.
- Model output: a real number $f(x) \in \mathbb{R}$.
- Prediction is given by the **sign** of $f(x)$:
  $$
  \hat y = \text{sign}(f(x)).
  $$

A natural loss is the **0/1 loss**:
$$
l(y, \hat y) =
\begin{cases}
0, & \text{if } \text{sign}(\hat y) = y,\\[4pt]
1, & \text{otherwise}.
\end{cases}
$$

Problem:

- This loss is **piecewise constant**—its gradient is zero almost everywhere.
- Gradient-based methods cannot effectively optimize it:
  - Parameters change only when we cross the decision boundary where the sign flips.
  - This is a measure-zero event in continuous space.

Instead, we introduce a **continuous surrogate**:

- Define the **margin** $m = y \hat y = y f(x)$.
  - If $m > 0$, the prediction’s sign matches $y$.
  - If $m < 0$, the prediction is incorrect.
- A classic surrogate is the **hinge loss**:
  $$
  l(y, \hat y) = \max(0, 1 - y \hat y).
  $$
  - Used in support vector machines (SVMs).
  - Penalizes low-margin predictions even when they are correct, encouraging a margin of at least $1$.
  - Is piecewise linear and (sub)differentiable, making it amenable to gradient-based optimization.

This illustrates the tension between:

- A loss that reflects our intuitive notion of performance (e.g., 0/1 error).  
- A loss that is **smooth enough** for effective optimization (e.g., hinge, cross-entropy).

---

### 3.4 Expected Risk and Overfitting

We ultimately care about **performance on unseen data**, not just on the training set.

Let $p(x, y)$ be the true data distribution.

**Definition (Expected risk – D.3.3)**  
The **expected risk** of a function $f$ is
$$
\operatorname{ER}[f]
  = \mathbb{E}_{(x, y) \sim p(x, y)} \bigl[\, l(y, f(x)) \,\bigr].
$$

Interpretation:

- $\operatorname{ER}[f]$ is the **average loss** over all possible inputs and outputs drawn from the true distribution.
- A model with low expected risk is good on average for future data.

However:

- We cannot compute $\operatorname{ER}[f]$ exactly because $p(x, y)$ is unknown.
- We only have access to a finite sample $\mathcal{D}$.

The **empirical risk** is:
$$
\hat{R}_n[f] = \frac{1}{n} \sum_{i=1}^n l\bigl(y_i, f(x_i)\bigr),
$$
which is a **Monte Carlo approximation** of $\operatorname{ER}[f]$ under the i.i.d. assumption.

---

### 3.5 Memorization vs Learning and the Generalization Gap

The empirical risk can be minimized **trivially** by memorization.

Example: define
$$
f(x) =
\begin{cases}
y, & \text{if } (x, y) \in \mathcal{D}_{\text{train}},\\[4pt]
\bar y, & \text{otherwise},
\end{cases}
$$
for some default $\bar y$ (e.g., $\bar y = 0$ or a majority class).

- On the training set, this function can achieve **minimal possible loss** (perfect fit if the loss rewards exact matches).
- But it is **useless** for new inputs not seen during training.

This highlights the difference between:

- **Memorization**: matching training data exactly.  
- **Learning/generalization**: capturing patterns that extend to unseen data.

Define a separate **test set** $\mathcal{D}_{\text{test}}$ of size $m$:

- $\mathcal{D}_{\text{test}}$ is disjoint from $\mathcal{D}_{\text{train}}$.
- Evaluate:
  - Training loss: $\hat{R}_n[f]$ on $\mathcal{D}_{\text{train}}$.
  - Test loss: $\hat{R}_m^{\text{test}}[f]$ on $\mathcal{D}_{\text{test}}$.

The **generalization gap** is the difference between expected and empirical risk or, in practice, between test and training empirical losses:

- Small gap: good generalization.
- Large gap (training loss low, test loss high): **overfitting**.

---

### 3.6 Statistical Learning Theory (High-Level Connection)

Statistical learning theory (SLT), largely developed by Vapnik and others, studies:

- How well $\hat{R}_n[f]$ approximates $\operatorname{ER}[f]$ for a given **function class** (hypothesis space).
- How **complexity measures** (e.g., VC dimension, Rademacher complexity, norms) control generalization.
- Under what conditions empirical risk minimization leads to small expected risk.

Modern deep neural networks exhibit surprising phenomena (e.g., good generalization even in regimes where classical theory would predict overfitting), motivating ongoing research in SLT.

---

## 4. Choosing Losses via Probabilistic Modeling

### 4.1 From Joint to Conditional Distributions

We assumed data comes from a joint distribution $p(x, y)$. Using the product rule:
$$
p(x, y) = p(x)\,p(y \mid x).
$$

Interpretation:

- $p(x)$: probability of encountering input $x$.
- $p(y \mid x)$: probability of observing output $y$ given $x$.

Two broad modeling philosophies:

- **Generative** models: estimate $p(x)$ or $p(x \mid y)$ and combine with $p(y)$.
- **Discriminative** models: estimate $p(y \mid x)$ directly.

Here we focus on the discriminative perspective.

Earlier we implicitly assumed $p(y \mid x)$ is sharply peaked near a single $y$, so a **deterministic predictor** $f(x)$ is reasonable. But we can relax that:

- Let $f(x)$ not be the prediction itself.
- Instead, use $f(x)$ to **parameterize a conditional distribution** $p(y \mid f(x))$.

This allows the model to express **uncertainty** and **multi-modality**.

---

### 4.2 Modeling $p(y \mid x)$ via $f(x)$

We let the network output parameters of a chosen distribution.

#### 4.2.1 Multi-class Classification (Categorical)

Let there be $K$ possible classes, $y \in \{1, \dots, K\}$.

- Represent the label as a **one-hot** vector:
  $$
  y = (y_1, \dots, y_K), \quad y_i \in \{0,1\}, \quad \sum_{i=1}^K y_i = 1.
  $$
- Let the model output a probability vector:
  $$
  f(x) = (f_1(x), \dots, f_K(x)) \in \Delta^{K},
  $$
  where each $f_i(x) \ge 0$ and $\sum_i f_i(x) = 1$ (typically via softmax).

Define a **categorical distribution** parameterized by $f(x)$:
$$
p(y \mid f(x)) = \prod_{i=1}^K f_i(x)^{y_i}.
$$

#### 4.2.2 Regression (Gaussian)

For scalar regression $y \in \mathbb{R}$:

- Let
  $$
  f(x) = \bigl(f_1(x), f_2(x)\bigr),
  $$
  where:
  - $f_1(x)$ is the mean,
  - $f_2(x)$ is (an unconstrained) scale; we enforce positivity by using $f_2(x)^2$ as the variance.
- Define:
  $$
  p(y \mid f(x)) = \mathcal{N}\bigl(y \,\big|\, f_1(x),\, f_2(x)^2\bigr).
  $$

This framework:

- Generalizes the deterministic view.
- Forces us to **choose a family of distributions** for $p(y \mid x)$.
- Gives a natural way to represent uncertainty (e.g., predictive variance).

---

### 4.3 Maximum Likelihood (ML) and Losses as Negative Log-Likelihood

Assume:

- Dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$.
- Samples are i.i.d. from $p(x, y)$.
- Model $f$ parameterizes $p(y \mid f(x))$.

The **likelihood** of the dataset under model $f$ is:
$$
p(\mathcal{D} \mid f)
 = \prod_{i=1}^n p\bigl(y_i \mid f(x_i)\bigr).
$$

**Definition (Maximum Likelihood – D.3.4)**  
The **maximum likelihood estimate** is:
$$
f^* = \arg\max_f \prod_{i=1}^n p\bigl(y_i \mid f(x_i)\bigr).
$$

This is equivalent to minimizing the **negative log-likelihood**:

1. Take logs:
   $$
   \log p(\mathcal{D} \mid f)
   = \sum_{i=1}^n \log p\bigl(y_i \mid f(x_i)\bigr).
   $$
2. Turn maximization into minimization:
   $$
   f^* = \arg\min_f \sum_{i=1}^n \bigl(-\log p\bigl(y_i \mid f(x_i)\bigr)\bigr).
   $$

Thus we can define a **pseudo-loss**:
$$
l(y, f(x)) = -\log p\bigl(y \mid f(x)\bigr),
$$
and train by minimizing its empirical average.

Key point:

- Many commonly used loss functions (e.g., squared error, cross-entropy) can be derived as **negative log-likelihoods** under suitable distributional assumptions.
- This gives a principled way to **select loss functions** based on probabilistic modeling.

---

## 5. Bayesian Learning

### 5.1 From a Single Best Model to a Distribution over Models

Maximum likelihood (or ERM) selects a **single** function $f^*$.

However:

- There may be many parameterizations that explain the data almost equally well.
- It can be advantageous to keep track of **uncertainty over models**, not just over predictions.

In **Bayesian learning**, we:

1. Place a **prior** distribution over functions (or equivalently, parameters):
   $$
   p(f).
   $$
   - In practice, $f$ is parameterized by weights $w$, so we use a prior $p(w)$.
   - Example: prefer small-norm functions (or parameters) by defining a prior that favors smaller $\|f\|$ or $\|w\|$.

2. Observe data $\mathcal{D}$ and form the **posterior** via Bayes’ theorem:
   $$
   p(f \mid \mathcal{D})
   = \frac{p(\mathcal{D} \mid f)\, p(f)}{p(\mathcal{D})},
   $$
   where:
   $$
   p(\mathcal{D}) = \int p(\mathcal{D} \mid f)\, p(f)\, df
   $$
   is the **evidence**, ensuring normalization.

This posterior encodes:

- Which functions are more credible given both the prior and the data.

---

### 5.2 Bayesian Prediction

Given the posterior $p(f \mid \mathcal{D})$, we make predictions by **averaging over models**:

$$
p(y \mid x, \mathcal{D}) = \int p\bigl(y \mid f(x)\bigr)\, p(f \mid \mathcal{D})\, df.
$$

This is the **Bayesian predictive distribution**.

Since the integral is generally intractable, we approximate it:

- Draw $k$ samples $f_1, \dots, f_k \sim p(f \mid \mathcal{D})$.
- Use the Monte Carlo approximation:
  $$
  p(y \mid x, \mathcal{D}) 
  \approx \frac{1}{k} \sum_{j=1}^k p\bigl(y \mid f_j(x)\bigr).
  $$

This averaging:

- Combines multiple plausible models.
- Often yields better-calibrated uncertainty than a single point estimate.

**Practical difficulty:**

- Computing the posterior $p(f \mid \mathcal{D})$ exactly is generally impossible for neural networks.
- Requires approximation methods:
  - Markov chain Monte Carlo (MCMC).
  - Variational inference.
  - Approximations like Monte Carlo dropout (treated later).

---

### 5.3 MAP Estimation and Regularization

If we only want the **most probable** function under the posterior (instead of the full distribution), we seek the **maximum a posteriori (MAP)** estimate:

$$
f^* = \arg\max_f p(\mathcal{D} \mid f)\, p(f).
$$

Take logs:
$$
f^* = \arg\max_f \Bigl[\log p(\mathcal{D} \mid f) + \log p(f)\Bigr].
$$

Equivalently, as a minimization:
$$
f^* = \arg\min_f \Bigl[-\log p(\mathcal{D} \mid f) - \log p(f)\Bigr].
$$

Interpretation:

- $-\log p(\mathcal{D} \mid f)$ is the **data term** (negative log-likelihood).
- $-\log p(f)$ is a **regularization term** derived from the prior.
  - It penalizes functions that are unlikely under the prior.
  - Encourages solutions in regions of high prior density.

Special case:

- If the prior is **uniform** over the function space, $\log p(f)$ is constant and:
  - MAP reduces to maximum likelihood.
  - No explicit regularization term remains.

Connection to regularization in practice:

- Many common regularizers (e.g., $\ell_2$ weight decay) can be interpreted as imposing a specific prior (e.g., Gaussian) on parameters.

---

### 5.4 Sequential Bayesian Updating and Catastrophic Forgetting

A key advantage of the Bayesian formalism is **sequential updating**:

- Suppose we first observe dataset $\mathcal{D}_1$.
  - We compute the posterior $p(f \mid \mathcal{D}_1)$.
- Later we observe additional data $\mathcal{D}_2$ from the same (or related) distribution.

We can treat $p(f \mid \mathcal{D}_1)$ as the **new prior** and update with $\mathcal{D}_2$:

$$
p(f \mid \mathcal{D}_1 \cup \mathcal{D}_2)
\propto p(\mathcal{D}_2 \mid f)\, p(f \mid \mathcal{D}_1).
$$

Conceptually:

- The original prior represents our **beliefs before any data**.  
- The posterior after seeing $\mathcal{D}_1$ becomes our **starting belief** before seeing $\mathcal{D}_2$.

This sequential update:

- Offers a principled way to handle **online learning** and **streaming data**.
- Helps mitigate **catastrophic forgetting**:
  - Standard training that only optimizes on new data may “forget” old tasks.
  - A Bayesian treatment, at least in principle, retains information from earlier data via the posterior.

---

### 5.5 Frequentist vs Bayesian View (High-Level)

At a high level:

- **Maximum likelihood / ERM** (frequentist flavor):
  - Treats the parameters as fixed but unknown.
  - Views data as random draws from $p(x, y)$.
  - Focuses on point estimates like $f^*$.

- **Bayesian learning**:
  - Treats parameters/functions as random variables with a prior.
  - After observing data, maintains a posterior distribution over models.
  - Uses the posterior to make predictions and quantify uncertainty.

Both views are useful:

- ML/ERM connects directly to optimization and is standard in deep learning practice.
- Bayesian methods provide a rich framework for uncertainty, regularization, and sequential learning, albeit at higher computational and conceptual cost.

---

**End of Summary for Chapter 3: Datasets and Losses**