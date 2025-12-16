# Chapter 1 – Introduction to Differentiable Models and Neural Networks

[TOC]

---

## High-Level Summary

- Neural networks are now core infrastructure for many technologies (LLMs, vision, recommender systems, molecular design, etc.), all built from a **small set of shared principles and components**.
- **Neural scaling laws**: increasing data, compute, and model size together tends to yield **predictable, steady improvements** in performance; modeling details matter less at very large scales.
- Modern large models (often called **foundation models**) show **strong generalization** and surprisingly low dependence on pure memorization, even though they are large enough to memorize their datasets.
- The ecosystem has shifted toward:
  - treating large pre-trained models as **black-box tools** accessed via prompting;
  - and, in parallel, a rich **open-source and experimentation culture** where smaller teams can fine-tune, merge, quantize, and probe models using commodity hardware.
- The book’s central viewpoint: neural networks are best seen as **differentiable models** or **differentiable programs** – compositions of differentiable primitives trained via gradient-based optimization.
- **Tensors** (multidimensional arrays) are the fundamental data structure; much of understanding a model reduces to reasoning about **tensor shapes** and constraints.
- The book is organized into:
  - **Part I – Compass and Needle**: optimization, gradients, and the basic machinery of differentiable models.
  - **Part II – Convolutions and Deep Architectures**: convolutional operators and sequence modeling.
  - **Part III – Down the Rabbit Hole**: attention/transformers, graphs, and recurrent models.
  - Plus online chapters and labs on more advanced or specialized topics.
- A brief historical and terminological discussion positions modern deep learning within decades of work on neural networks, optimization, and software frameworks.
- The book is **concept-focused but not exhaustive**, explicitly limited in scope and recency; it aims to provide a conceptual toolkit rather than a complete catalog of tricks.
- It is released under a **CC BY-SA** license, allowing reuse and modification under the same terms.

---

## 1. Neural Networks, Scaling Laws, and the “Bitter Lesson”

### 1.1 Neural Networks in Modern AI

**Key idea**: Neural networks have become a **general-purpose modeling tool** used across many domains:

- Drones and autonomous cars.
- Search engines and recommender systems.
- Molecular and drug design.
- Large language models (LLMs) and other generative systems.

Despite this variety, these systems are built from a **small, shared toolkit** of:

- Parameterized differentiable components.
- Standard optimization algorithms.
- General design patterns for model architectures.

Current research often focuses less on inventing entirely new mechanisms and more on **scaling up** these existing tools.

---

### 1.2 Neural Scaling Laws

**Concept: Neural scaling laws**

Informally:

- For a wide range of tasks, if we **simultaneously** increase:
  - the **amount of data**,
  - the **available compute**, and
  - the **model size** (number of parameters),
- then model performance (e.g., accuracy, loss) tends to improve in a **smooth and predictable way**, often following approximate power laws.

Equivalently:

- To achieve a given target accuracy on a task, the **required compute** decreases by a roughly **constant multiplicative factor over time**, as models and training procedures improve.

The **training cost** of notable AI models (e.g., since 2016) illustrates this:

- Training cost (in US dollars) is strongly tied to:
  - dataset size,
  - compute power,
  - model size.
- As models become larger and better trained, **architectural variations** often matter **less** in the asymptotic regime than sheer scale.

This perspective matches Sutton’s **“bitter lesson”**:

> Progress in AI has largely come from leveraging simple, general-purpose methods that scale with compute, rather than from complex, domain-specific hand-engineering.

---

### 1.3 Generalization vs Memorization at Scale

Modern neural networks are often:

- So large that they **could** memorize their entire training dataset (i.e., behave like a giant look-up table).
- Yet empirically, **well-trained large models**:
  - Generalize well to new inputs.
  - Can perform tasks **not explicitly represented** in the training data (e.g., in-context learning, compositional reasoning).
  - Show **limited reliance** on pure memorization, even when overparameterized.

As datasets become huge:

- The line between **“in-distribution”** and **“out-of-distribution”** becomes blurry.
- Many “new” tasks or prompts lie within the broader distribution induced by the training corpus.
- Large-scale models exhibit **broad generalization** rather than brittle overfitting.

This behavior is one of the empirical foundations for viewing neural networks as powerful **approximators of probability distributions**:

- Given data sampled from an underlying distribution, a neural network is trained to approximate that distribution (e.g., model $p(\text{output} \mid \text{input})$).
- In principle, approximation could fail or reduce to memorization.
- In practice, with appropriate training and regularization, **generalization emerges**.

---

## 2. Foundation Models, Black-Box Use, and Open-Source Experimentation

### 2.1 Foundation Models and Prompting

The rise of **foundation models**:

- Large models trained on broad corpora (text, images, audio, multimodal data).
- Can be **adapted or prompted** to perform many downstream tasks without task-specific retraining.

User interaction is often via:

- **Prompting**: giving instructions, examples, or queries in natural language or visual form.
- The **internal parameters** and structure of the model are usually **hidden**, especially in commercial systems.

High-level analogy:

- Historically: you might have written your own numerical or ML libraries in C++.
- Now: you often call pre-built, possibly closed-source libraries or services (e.g., cloud APIs) whose source you never see.

---

### 2.2 Open-Source Ecosystem and “Looking Under the Hood”

In parallel with large proprietary models:

- A **vibrant open-source community** provides:
  - Model checkpoints (e.g., LLMs, diffusion models).
  - Code, training scripts, and infrastructure.
- This allows:

  - **Fine-tuning** models for specific tasks using relatively modest hardware.
  - **Model merging**—combining weights or behaviors from different models.
  - **Quantization** for deployment on low-power devices (e.g., edge or mobile).
  - **Robustness testing**, interpretability analyses, and safety evaluations.
  - **Design of new architectures and variants**.

Key message:

> Even if only a few organizations can train the largest models from scratch, many researchers and practitioners can still **experiment, adapt, and innovate** by understanding how these models work internally.

This motivates the book’s goal:

- To encourage the reader to **“look under the hood”**:
  - Understand how differentiable models transform data.
  - Learn practical “tricks” and “idiosyncrasies” that arise from real-world training and debugging.

---

## 3. Supervised Learning and Differentiable Models

### 3.1 Assumed Background and View of Supervised Learning

The book assumes familiarity with basic **machine learning (ML)** and especially **supervised learning (SL)**.

**Supervised learning viewpoint**:

- You are given data $\{(x_i, y_i)\}$ representing examples of a desired behavior (input–output pairs).
- You choose a parameterized model $f_\theta$.
- You **optimize** $\theta$ so that $f_\theta(x_i)$ approximates $y_i$ well, for all training examples and ideally for unseen examples.

Many complex tasks can be rephrased as supervised learning:

- **Image generation**:
  - Collect a large dataset of images and their captions.
  - Learn to generate an image given a text description.
- **Natural language modeling**:
  - Collect a large corpus of text.
  - Learn to predict the next token/sentence from previous text.
- **Medical diagnosis from X-rays**:
  - Collect a dataset of images paired with expert labels (e.g., diagnoses).
  - Learn to map images to diagnoses.

So, **complex “intelligent” tasks** can often be reduced to:

> Approximate a mapping from input to output based on many labeled examples.

---

### 3.2 Learning as Search Over Parameters

**Key idea: Learning as search**

- A model is defined by a set of **parameters** (degrees of freedom), often denoted $\theta$.
- We treat learning as a **search** over possible parameter configurations.
- Goal: find $\theta^\star$ such that some performance measure (e.g., loss function) is acceptably low.

Challenges:

- Modern models can have **millions, billions, or even trillions** of parameters.
- Exhaustive search is impossible; we need **efficient, structured search**.

---

### 3.3 Differentiable Models, Gradients, and Optimization

**Definition: Differentiable model**

- A model is differentiable if its mapping from parameters $\theta$ and inputs $x$ to outputs (or losses) is **differentiable** with respect to $\theta$.
- More precisely, we require that each component of the computation graph is differentiable.

Consequences:

- We can compute **gradients**:
  $$
  \nabla_\theta L(\theta) = \frac{\partial L(\theta)}{\partial \theta},
  $$
  where $L(\theta)$ is a loss function (e.g., average training loss).
- Knowing the gradient tells us how $L$ changes when we slightly perturb $\theta$.
- This enables **gradient-based optimization** methods such as (stochastic) gradient descent.

In practice:

- We compose many **differentiable primitives**, such as:
  - Linear transformations.
  - Elementwise nonlinearities (e.g., ReLU, sigmoid).
  - Convolutions, attention, normalization layers, etc.
- Using tools like **automatic differentiation**, we can efficiently compute gradients for **high-dimensional** parameter spaces.

This setup—**differentiable components + gradient-based optimization**—is the conceptual core of the book’s approach.

---

## 4. Data Types, Tensors, and Differentiable Primitives

### 4.1 From Data Types to Tensors

A central view in the book:

> Neural networks are **sequences of differentiable primitives** acting on structured arrays (tensors).

Two key questions:

1. **What input and output data types can we handle?**
2. **What differentiable primitives can we use and how can we compose them?**

Differentiability imposes constraints:

- Many standard data types are **discrete**, e.g.:

  - Characters, tokens.
  - Integers and symbolic structures.

- These are not directly differentiable.
- Instead, differentiable models typically operate on:
  - **Real-valued arrays** (tensors), which can represent:
    - Images (grids of pixels).
    - Audio (sequences of numerical samples).
    - Text (encoded as sequences of embeddings).
    - Graph features, video frames, etc.

---

### 4.2 Tensors and Their Rank

**Definition: Tensor**

- In this book, a tensor is a **multidimensional array** of objects, usually real numbers.
- Formally, we treat it as an **$n$-dimensional array**, where $n$ is called the **rank** (in this text).

The basic cases:

1. **Scalars (rank 0)**  
   - A single number, e.g. $x$ or $y$.
   - Notation: lowercase letters, e.g. $x$.

2. **Vectors (rank 1)**  
   - A column of values, e.g. $x \in \mathbb{R}^d$.
   - Notation: lowercase bold, e.g. $\mathbf{x}$.
   - The corresponding row vector is denoted $\mathbf{x}^\top$ when needed.

3. **Matrices (rank 2)**  
   - A rectangular array of values, e.g. $X \in \mathbb{R}^{n \times d}$.
   - Notation: uppercase bold, e.g. $\mathbf{X}$, $\mathbf{Y}$.

4. **Higher-rank tensors ($n > 2$)**  
   - No special letter style; just think of them as generic multidimensional arrays.
   - We **avoid** calligraphic letters for tensors, reserving those for sets or distributions.

All these are collectively referred to as **tensors**.

---

### 4.3 Shapes and Shape Notation

Understanding operations often reduces to understanding the **shape** of each tensor.

**Notation for shapes**:

- We write:
  $$
  X \sim (b, h, w, 3)
  $$
  to indicate that $X$ is a rank-4 tensor with shape $(b, h, w, 3)$, where:
  - $b$ might represent batch size,
  - $h$ and $w$ spatial dimensions (height, width),
  - $3$ the number of channels (e.g., RGB).

More generally:

- We use $x \sim (d)$ instead of $x \in \mathbb{R}^d$.
- We use $X \sim (n, d)$ instead of $X \in \mathbb{R}^{n \times d}$.

The author also reuses the symbol $\sim$ to denote sampling from a distribution:

- For example:
  $$
  \epsilon \sim \mathcal{N}(0, 1)
  $$
  indicates that $\epsilon$ is sampled from a standard normal distribution.

Although this is potentially ambiguous, the **context** (shape vs distribution) is usually clear.

---

### 4.4 Constrained Tensors: Binary and Simplex

Sometimes we need to constrain tensor elements.

**Example 1 – Binary tensors**

- Notation:
  $$
  x \sim \text{Binary}(c)
  $$
  means:

  - $x$ is a tensor of shape $(c)$ (or some other specified shape),
  - Each element of $x$ belongs to $\{0, 1\}$.

**Example 2 – Simplex-constrained vectors**

- Notation:
  $$
  x \sim \Delta(a)
  $$
  means:

  - $x$ is a vector of length $a$,
  - It belongs to the **probability simplex**, i.e.
    $$
    x_i \ge 0, \quad \sum_i x_i = 1.
    $$

- For higher-rank tensors, e.g. $X \sim \Delta(n, c)$, the convention is:

  - The simplex constraint applies along the **last dimension**.
  - So each row $X_i$ is a probability vector in $\Delta(c)$.

These conventions clarify both **shape** and **constraints** simultaneously.

---

## 5. Structure of the Book

### 5.1 Part I – Compass and Needle (Chapters 2–6)

Focus:

- The fundamental setup for **differentiable models**:
  - Representation of data as tensors.
  - Loss functions and optimization.
  - Gradients, automatic differentiation.
  - Gradient descent and variants for training models.

Part I builds the **conceptual and mathematical compass** for navigating the rest of the book.

---

### 5.2 Part II – Convolutions and Deep Architectures (Chapters 7–9)

Focus on **convolutional operators** as a prototypical differentiable primitive.

Key ideas:

- Convolutions apply whenever data can be represented as an **ordered sequence** (or grid) of elements:
  - Audio (1D sequences).
  - Images (2D grids).
  - Text (sequences of tokens).
  - Video (spatio-temporal sequences).
- Discussion of **deep architectures**:
  - Models composed of many layers in sequence.
  - How depth, nonlinearity, and convolution interact.

Additional concepts introduced:

- **Text tokenization**:
  - Converting raw text into discrete tokens (words, subwords, characters).
  - Then embedding tokens into continuous vectors.
- **Autoregressive sequence generation**:
  - Modeling $p(x_t \mid x_{<t})$ and generating sequences step-by-step.
- **Causal modeling**:
  - Ensuring that predictions at time $t$ depend only on past (or allowed) information.
  - Important for language modeling and time series.

These concepts underpin modern **LLMs** and many state-of-the-art generative models.

---

### 5.3 Part III – Down the Rabbit Hole (Chapters 10–13)

Further exploration of differentiable architectures for different data structures:

1. **Sets and Attention/Transformers (Chapters 10–11)**  
   - Attention layers as differentiable primitives that operate on **sets or sequences**.
   - Transformer architectures:
     - Built upon attention.
     - Have revolutionized natural language processing and beyond.

2. **Graphs (Chapter 12)**  
   - Differentiable models for graph-structured data.
   - Graph neural networks and related architectures.

3. **Recurrent Layers for Temporal Sequences (Chapter 13)**  
   - Recurrent neural networks (RNNs) and variants.
   - Suitable for sequential/temporal data with explicit recurrence.

Together, these chapters extend the core ideas to more complex structure types: sets, sequences, graphs, and time.

---

### 5.4 Online Material and Labs

The book is complemented by online resources:

- Additional chapters on topics such as:
  - **Generative modeling**.
  - **Conditional computation**.
  - **Transfer learning**.
  - **Explainability**.
- These chapters are:
  - More **research-oriented**.
  - Not tied to a specific data type.
  - Largely independent of each other (can be read in any order).

There are also **guided lab sessions** (in notebook form):

- Cover many core topics from the book.
- Include more advanced areas like:
  - **Contrastive learning**.
  - **Model merging**.
- Aim to connect **conceptual understanding** with **practical experimentation**.

---

## 6. Historical Context and Terminology

### 6.1 From Neurons to Differentiable Models

**Terminology evolution**:

- Early work used biological metaphors:
  - **Neurons**, **synapses**, **layers**, **activations**, **connectionism**.
- The phrase **deep learning** later emphasized:
  - Depth (many layers).
  - Larger scale and complexity compared to earlier models.

However:

- Modern artificial neural networks have **little resemblance** to real biological neurons or brain circuits.
- The terminology persists mostly for **historical and conventional reasons**.

The author argues that, from a modern perspective, the **most faithful description** of these models is:

> They are **differentiable models**, or more broadly, components in **differentiable programs**.

This viewpoint naturally connects to the emerging field of **differentiable programming**:

- Studying computer programs in which many (or all) operations are differentiable.
- Combining insights from:
  - Optimization.
  - Programming languages.
  - Software systems.

---

### 6.2 Milestones in Neural Network History

Key historical points highlighted:

- **Perceptrons (1950s)**:
  - Early neural network models.
  - Generated media hype and optimism about “electronic brains.”

- **Gradient methods for linear models (19th century)**:
  - Basic ideas of numerical optimization and gradient descent predate modern ML by over a century.

- **Fully-connected networks (1980s)**:
  - Models similar in structure to today’s multilayer perceptrons were already in use.

- **Convolutional networks (late 1990s)**:
  - Convolutional architectures were known and developed for vision and related tasks.

- **2012–2017: Deep Learning Boom**:
  - Large-scale benchmarks, especially the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)**, drove innovation.
  - Rapid growth in model size and dataset size.
  - Convolutional models became dominant in computer vision.

- **2017 and After: Transformers and beyond**:
  - Introduction of **transformers** dramatically reshaped natural language processing.
  - Soon extended to images, video, graphs, audio, and multimodal tasks.
  - Led to the current focus on:
    - LLMs,
    - Multimodal models,
    - Generative models.

Despite the long history, only in recent decades did we obtain:

- Sufficient data.
- Sufficient compute.
- Mature software frameworks.

to fully realize the potential of these architectures.

---

## 7. Software Frameworks and Differentiable Programming in Practice

There is a tight coupling between:

- **Conceptual development** of models.
- **Software tools** used to implement them.

Major steps in framework evolution:

- Early tools like **Theano** enabled:
  - Symbolic computation graphs.
  - Automatic differentiation.
- Later frameworks:
  - **Caffe**, **Chainer**, then **TensorFlow**, **PyTorch**, **JAX**, etc.
- These frameworks made it easier to:
  - Express complex models as compositions of differentiable primitives.
  - Run large-scale experiments efficiently on GPUs and other accelerators.

The book:

- Frequently connects theory to constructs from these frameworks.
- Focuses mostly on **PyTorch** and **JAX** for concrete examples.

However:

- It is **not a programming manual**.
- For detailed usage of APIs and syntax, readers are referred to the official documentation.

---

## 8. Notation, Icons, and Reading Guidance

### 8.1 Notation Summary

- **Scalars**: lowercase letters, e.g. $x$.
- **Vectors**: bold lowercase, e.g. $\mathbf{x}$, often treated as column vectors.
- **Row vectors**: $\mathbf{x}^\top$ when necessary.
- **Matrices**: bold uppercase, e.g. $\mathbf{X}$.
- **Tensors (rank > 2)**: general arrays; no special calligraphic letters.

**Shapes**:

- $x \sim (d)$: vector of length $d$ (instead of $x \in \mathbb{R}^d$).
- $X \sim (n, d)$: matrix with $n$ rows and $d$ columns.
- $X \sim (b, h, w, 3)$: 4D tensor (e.g. batch of RGB images).

**Constraints**:

- $x \sim \text{Binary}(c)$: binary-valued tensor.
- $x \sim \Delta(a)$: vector on the simplex (nonnegative entries summing to $1$).
- $X \sim \Delta(n, c)$: each row of $X$ lies on the simplex.

**Distributions**:

- $\epsilon \sim \mathcal{N}(0, 1)$: sampling from a probability distribution.
- The symbol $\sim$ is reused, with context clarifying whether it refers to **shape** or **sampling**.

---

### 8.2 Icons in the Book

The text uses margins with small icons to guide reading:

- **Bottle**: highlights particularly important **definitions**.
- **Clock**: marks sections that are **crucial** for understanding later material (do not skip).
- **Teacup**: indicates **more relaxed, discursive** sections that are relatively **optional**.

These are visual cues rather than mathematical content, but they structure how to navigate the material.

---

## 9. Scope, Limitations, and Reading Philosophy

### 9.1 What the Book Aims to Be

Goals:

- Present a **coherent, conceptually unified** introduction to differentiable models.
- Cover ideas that are:
  1. **Common in current practice**, and
  2. **Likely to remain useful** in the near future.

The content draws heavily from the author’s experience:

- Teaching **Neural Networks for Data Science Applications**.
- Other courses and tutorials on neural networks and ML.
- Research interests in optimization, architectures, and applications.

The book overlaps with other modern deep learning texts in early chapters, but:

- The exposition,
- Emphasis,
- And advanced topics

reflect the author’s personal viewpoint and choices.

---

### 9.2 What the Book Is Not

The author explicitly notes several **non-goals**:

1. **Not fully up-to-date or exhaustive**  
   - The field progresses quickly; some material may be outdated by the time of reading.
   - The book does not attempt to catalog every architecture or trick.

2. **Not a full survey of all model variants**  
   - For each concept (e.g., normalization) only a few variants are discussed (e.g., batch normalization vs layer normalization).
   - Many more exist in the broader literature (e.g., as tracked on sites like Papers With Code).

3. **Not a hardware or engineering handbook**  
   - Large-scale deployments require sophisticated engineering:
     - Distributed training.
     - Memory and compute optimization.
     - System-level design.
   - These topics are only lightly touched upon.

4. **Not a substitute for hands-on experience**  
   - Practical intuition about what “works” comes from:
     - Implementing models.
     - Debugging.
     - Reading code and blog posts.
   - The book encourages this, but cannot replace it.

Readers are encouraged to complement the text with:

- **Opinionated blog posts and tutorials**.
- **Hands-on experimentation** with open-source code.

---

## 10. Acknowledgments and License

### 10.1 Acknowledgments

The author credits:

- A LaTeX package for **equation coloring**, which improves readability.
- Licensed artwork:
  - Color illustrations of *Alice in Wonderland* and margin symbols from Shutterstock.
  - John Tenniel’s original *Alice* illustrations from public sources (e.g., Wikimedia) for figures.
- Colleagues and readers who:
  - Provided feedback on early drafts.
  - Suggested corrections and improvements.
  - Encouraged the book’s publication.
- Various readers who sent feedback via email.

---

### 10.2 License

The book is released under a **Creative Commons Attribution–ShareAlike (CC BY-SA)** license.

**Implications**:

- You may:
  - Distribute the material.
  - Remix, adapt, and build upon it.
  - Use it for commercial purposes.
- Conditions:
  - You must provide **proper attribution** to the original creator.
  - Any derivative works must be released under the **same CC BY-SA license**.

This license encourages:

- Sharing and adaptation.
- Open, collaborative development of educational resources around differentiable models and neural networks.

---

## 11. Conceptual Takeaways

To close the introduction, the central conceptual messages are:

- Modern neural networks are best seen as **differentiable models**, or building blocks in differentiable programs.
- They operate on **tensors**, and much of their behavior can be understood in terms of:
  - Shapes,
  - Differentiable primitives,
  - Gradients and optimization.
- **Scaling laws** and the **bitter lesson** explain why simple, general methods plus massive compute and data have been so successful.
- While few can train the largest models from scratch, many can still:
  - Understand their internals.
  - Adapt and extend them via fine-tuning, merging, and analysis.
- The book aims to provide the **“compass and needle”** needed to navigate this landscape: enough theory, notation, and practical framing to engage with current and future work on differentiable models.