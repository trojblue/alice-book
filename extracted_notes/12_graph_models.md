# Chapter 12 – Graph Models

[TOC]

---

## Brief Overview of the Chapter

- Introduces graphs as a unifying structure for sets, sequences, images, and more general relational data.
- Defines graph representations: adjacency and incidence matrices, node/edge/graph features, degrees, and normalized adjacencies.
- Develops diffusion and Laplacian-based notions of smoothness, leading to **manifold regularization**.
- Defines **graph convolutional (GC) layers**, their locality and permutation equivariance, and builds **graph convolutional networks (GCNs)** for node, edge, and graph-level tasks.
- Discusses sparse implementations, mini-batching with block-diagonal adjacency, and scatter-based pooling.
- Generalizes to **graph attention networks (GAT)** and **message-passing neural networks (MPNNs)**, including edge features and spatio-temporal extensions.
- Introduces **graph transformers**, where graph connectivity is encoded in structural embeddings and fed to a standard transformer.
- Connects theory to practice with common libraries and a didactic “GNN for vision” exercise.

---

## 1. Graph-Structured Data

### 1.1 Motivation and Scope

Many real-world domains are naturally **graph-structured**:

- **Molecules**: atoms (nodes) sparsely connected by chemical bonds (edges).
- **Transportation networks**: intersections or stations connected by roads or routes.
- **Social/recommender networks**: people or products connected by interactions.
- **Energy/traffic networks**: components linked by physical or informational flows.

In earlier chapters, data was:

- **Unstructured**: tabular vectors.
- **Simply structured**: sets, sequences, grids (images).

Graphs extend these by explicitly encoding **irregular, sparse dependencies** between entities.

The goal of this chapter is to introduce **differentiable models** that operate directly on such graph-defined data.

---

### 1.2 Basic Graph Definition

We adopt a standard formalization:

- **Definition (Graph)**  
  A graph is
  $$
  \mathcal{G} = (\mathcal{V}, \mathcal{E}),
  $$
  where:
  - $\mathcal{V} = \{1, \dots, n\}$ is the **node (vertex) set**,
  - $\mathcal{E} \subseteq \{(i,j) \mid i,j \in \mathcal{V}\}$ is the **edge set**.

The number of nodes is $n = |\mathcal{V}|$ and the number of edges is $m = |\mathcal{E}|$. In typical datasets, both $n$ and $m$ may vary from graph to graph.

Graphs generalize familiar data types:

- **Set**  
  Only self-loops $(i,i)$; no edges between distinct nodes. Equivalent to a collection of independent objects.
- **Sequence**  
  Nodes connected in a line: $(1,2), (2,3), \dots, (n-1,n)$.
- **Image**  
  Nodes are pixels; edges connect spatial neighbors on a 2D grid.
- **Fully connected structure**  
  All $(i,j)$ present; related to fully-connected attention layers.

---

### 1.3 Adjacency and Incidence Matrices

#### 1.3.1 Adjacency Matrix

- **Definition (Adjacency matrix)**  
  For a graph with $n$ nodes, the adjacency matrix is
  $$
  A \in \{0,1\}^{n \times n}, \qquad
  A_{ij} =
  \begin{cases}
  1, & (i,j) \in \mathcal{E},\\
  0, & \text{otherwise}.
  \end{cases}
  $$

Special cases:

- **Set**: $A = I_n$ (identity) if we only include self-loops.
- **Fully connected graph**: $A$ is all ones (possibly excluding the diagonal).
- **Image**: $A$ has a regular, Toeplitz-like structure encoding the grid neighbors.

If every edge is bidirectional, i.e. $(i,j)\in\mathcal{E}$ iff $(j,i)\in\mathcal{E}$, the graph is **undirected**, and:
$$
A^\top = A.
$$

In this chapter we mainly treat undirected graphs; extensions to directed graphs are straightforward.

We typically assume **self-loops** are present: $A_{ii} = 1$. If they are not, we add them by:
$$
A \leftarrow A + I_n.
$$

#### 1.3.2 Incidence Matrix

A second representation is the **incidence matrix**.

- **Definition (Incidence matrix)**  
  Let $m = |\mathcal{E}|$ be the number of edges. The incidence matrix is
  $$
  B \in \{0,1\}^{n \times m}, \qquad
  B_{i e} =
  \begin{cases}
  1, & \text{node } i \text{ is an endpoint of edge } e,\\
  0, & \text{otherwise}.
  \end{cases}
  $$

For simple undirected graphs, each edge connects exactly two nodes, so each column of $B$ has **exactly two ones**. Summing over rows yields $2$ per column.

Adjacency and incidence matrices encode the same connectivity in different ways; choice depends on the operations we want to perform.

---

### 1.4 Node, Edge, and Graph Features

Graphs usually come with **features**:

- **Node features**: $x_i \in \mathbb{R}^c$ for node $i$.
  - Examples: atom type, user attributes, pixel intensities (for image-as-graph).
- **Edge features**: $e_{ij}$ for edge $(i,j)$.
  - Examples: bond type, road capacity, time since friendship started.
- **Graph features**: a global vector per graph.
  - Examples: overall charge of a molecule, global network statistics.

In the basic setting of this section, we assume **only node features**:

- **Feature matrix**
  $$
  X =
  \begin{bmatrix}
  x_1^\top\\
  \vdots\\
  x_n^\top
  \end{bmatrix}
  \in \mathbb{R}^{n \times c}.
  $$

A graph with node features is represented as the pair $(X, A)$.

---

### 1.5 Permutation Invariance

The numerical ordering of nodes is arbitrary. A graph model must not depend on how we label nodes.

Let $P \in \{0,1\}^{n \times n}$ be a **permutation matrix**. Then:

- Applying $P$ to $X$ permutes rows (nodes):
  $$
  X \mapsto P X.
  $$
- Applying $P$ to $A$ gives:
  $$
  A \mapsto P A P^\top,
  $$
  simultaneously permuting rows and columns (relabelling nodes).

Thus, $(X, A)$ and $(PX, PAP^\top)$ represent the **same graph**.  

Graph layers should be **permutation equivariant**:
$$
f(PX, P A P^\top) = P f(X, A).
$$

---

## 2. Degrees, Normalization, and Diffusion

### 2.1 Degrees and Degree Matrix

- **Definition (Degree of a node)**  
  The degree of node $i$ is:
  $$
  d_i = \sum_{j=1}^n A_{ij}.
  $$

- **Definition (Degree matrix)**  
  The degree matrix is the diagonal matrix:
  $$
  D = \operatorname{diag}(d_1, \dots, d_n)
  =
  \begin{bmatrix}
  d_1 & 0 & \dots & 0\\
  0 & d_2 & \dots & 0\\
  \vdots & \vdots & \ddots & \vdots\\
  0 & 0 & \dots & d_n
  \end{bmatrix}.
  $$

The **degree distribution** (histogram of $d_i$) is a key structural signature; e.g., preferential attachment graphs yield heavy-tailed distributions with hubs.

---

### 2.2 Normalized Adjacency and Random Walk Interpretation

We can normalize $A$ to interpret it as a **random-walk transition matrix**.

- **Row-normalized adjacency**
  $$
  A_{\text{row}}' = D^{-1} A,
  \qquad
  A'_{ij} = \frac{1}{d_i} A_{ij}.
  $$
  Each row sums to 1:
  $$
  \sum_j A'_{ij} = 1.
  $$
  Interpreted as: from node $i$, move to neighbor $j$ with probability $A'_{ij}$.

- **Column-normalized adjacency**
  $$
  A_{\text{col}}' = A D^{-1}.
  $$

- **Symmetric normalization**
  $$
  \tilde{A} = D^{-1/2} A D^{-1/2},
  \qquad
  \tilde{A}_{ij} = \frac{A_{ij}}{\sqrt{d_i d_j}}.
  $$

All these matrices share the **sparsity pattern** of $A$: if $(i,j) \notin \mathcal{E}$, then $A_{ij} = \tilde{A}_{ij} = 0$. In graph signal processing, such matrices (including Laplacians) are called **graph-shift matrices**.

---

### 2.3 Sparsity in Graph Matrices

Real-world graphs are usually sparse: $m = |\mathcal{E}| \ll n^2$.

Example adjacency for a 6-node graph:
$$
A =
\begin{bmatrix}
0 & 1 & 1 & 1 & 1 & 1\\
1 & 0 & 0 & 0 & 1 & 0\\
1 & 0 & 0 & 0 & 1 & 1\\
1 & 0 & 0 & 0 & 0 & 1\\
1 & 1 & 1 & 0 & 0 & 0\\
1 & 0 & 1 & 1 & 0 & 0
\end{bmatrix}.
$$

Most entries are zero. This matters because:

- **Sparse storage** (e.g., coordinate list, CSR) stores only non-zero entries.
- Specialized sparse operations can be asymptotically more efficient than dense ones.
- Frameworks like JAX and PyTorch provide sparse linear algebra, and graph libraries build GNN layers on top of such primitives.

---

### 2.4 Diffusion on Graphs

Let $x \in \mathbb{R}^n$ be a scalar feature on nodes (e.g., temperature, label score). Diffusion via adjacency:

- **Matrix form**
  $$
  x' = A x,
  $$
  where $A$ can be raw or normalized adjacency (or another graph-shift).

- **Node-wise form**
  $$
  x'_i = \sum_{j \in \mathcal{N}(i)} A_{ij} x_j,
  \qquad
  \mathcal{N}(i) = \{ j \mid (i,j)\in\mathcal{E} \}.
  $$

Interpretation: $x'_i$ is a weighted average of neighbors’ values; this is a **smoothing / diffusion** process on the graph.

---

### 2.5 Graph Laplacian and Gradient-Like Behavior

- **Definition (Graph Laplacian)**  
  $$
  L = D - A.
  $$

For a scalar signal $x \in \mathbb{R}^n$:

- **Node-wise expression**
  $$
  (Lx)_i = \sum_{(i,j)\in\mathcal{E}} A_{ij} (x_i - x_j).
  $$

Interpretation:

- $(Lx)_i$ measures how different $x_i$ is from its neighbors.
- It plays the role of a discrete gradient/divergence operator.

Key spectral property:

- The constant vector $\mathbf{1}$ is always an eigenvector of $L$ with eigenvalue $0$.
- Under repeated Laplacian diffusion, signals tend to converge toward a constant over each connected component.

---

### 2.6 Laplacian Quadratic Form

- **Definition (Laplacian quadratic form)**  
  For $x \in \mathbb{R}^n$,
  $$
  x^\top L x
  = \sum_{(i,j)\in\mathcal{E}} A_{ij} (x_i - x_j)^2.
  $$

This scalar measures how **non-smooth** $x$ is along edges:

- If $x_i \approx x_j$ whenever $A_{ij}$ is large, the quadratic form is small.
- If $x$ changes sharply across high-weight edges, the form is large.

Thus, $x^\top L x$ is a **smoothness penalty** on signals over the graph.

---

### 2.7 Building Graphs from Tabular Data

Given a standard supervised dataset
$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n,
$$
we can **construct a graph** where:

- Nodes: individual samples $x_i$.
- Edges: connect similar samples.

A common construction uses a Gaussian (heat-kernel) affinity with threshold:

- **Distance-based adjacency**
  $$
  A_{ij} =
  \begin{cases}
  \exp\big(-\|x_i - x_j\|_2^2\big), & \|x_i - x_j\|_2^2 < \tau,\\
  0, & \text{otherwise},
  \end{cases}
  $$
  where $\tau > 0$ is a hyperparameter controlling sparsity.

This creates a **similarity graph** reflecting the geometry of the data manifold in feature space.

---

### 2.8 Manifold Regularization

Let $f:\mathbb{R}^d \to \mathbb{R}$ be a scalar-output model. Evaluate $f$ on all inputs:

$$
f =
\begin{bmatrix}
f(x_1)\\
\vdots\\
f(x_n)
\end{bmatrix}
\in \mathbb{R}^n.
$$

The Laplacian quadratic form applied to $f$ is:

$$
f^\top L f
= \sum_{i,j} A_{ij} \big(f(x_i) - f(x_j)\big)^2.
$$

This penalizes predictions that vary sharply between similar points (large $A_{ij}$).

We can combine this with standard supervised loss:

- **Manifold regularization objective**
  $$
  f^\star
  = \arg\min_f
  \left[
    \sum_{i=1}^n \mathcal{L}(y_i, f(x_i))
    + \lambda\, f^\top L f
  \right],
  $$
  where:

  - $\mathcal{L}$ is a generic loss (e.g., cross-entropy, squared loss),
  - $\lambda > 0$ trades off data fit and smoothness.

Important properties:

- The regularization term uses **only** $(x_i)$ and the graph; it does not need labels.
- Particularly useful in **semi-supervised** settings with many unlabeled points: unlabeled points still constrain $f$ via the smoothness term.
- However, at inference time the model depends only on $x$; the graph is typically discarded after training.

This motivates architectures that **explicitly embed** graph connectivity into their computation, which we now develop.

---

## 3. Graph Convolutional Layers

### 3.1 Graph Layers and Graph-Shift Matrices

We aim to define layers that:

- Take node features and graph structure as input.
- Are **permutation equivariant**.
- Are **local** in the graph sense.

- **Graph layer**
  $$
  H = f(X, A),
  $$
  where:

  - $X \in \mathbb{R}^{n\times c}$: input node features,
  - $A \in \mathbb{R}^{n\times n}$: graph-shift matrix (e.g., adjacency, normalized adjacency, Laplacian),
  - $H \in \mathbb{R}^{n\times c'}$: output node embeddings.

We require permutation equivariance:
$$
f(PX, P A P^\top) = P f(X, A)
$$
for all permutation matrices $P$.

---

### 3.2 Locality via Induced Subgraphs

For a subset $\mathcal{S} \subseteq \mathcal{V}$:

- **Induced subgraph**
  $$
  \mathcal{G}_\mathcal{S} = (X_\mathcal{S}, A_\mathcal{S}),
  $$
  where $X_\mathcal{S}$ and $A_\mathcal{S}$ are the rows/columns of $X,A$ restricted to $\mathcal{S}$.

- **1-hop neighborhood**
  $$
  \mathcal{N}(i) = \{ j \mid (i,j)\in\mathcal{E} \}.
  $$

- **Definition (Local graph layer)**  
  A layer $H = f(X,A)$ is **local** if
  $$
  H_i = f\big( X_{\mathcal{N}(i)}, A_{\mathcal{N}(i)} \big)
  $$
  for each node $i$.

Differences from standard image convolutions:

- $\mathcal{N}(i)$ has **no natural ordering**.
- $|\mathcal{N}(i)|$ varies with $i$.

Hence, we cannot reuse the standard convolution definition (with a fixed, ordered kernel) directly.

We can extend locality to higher-hop neighborhoods:

- **2-hop neighborhood**
  $$
  \mathcal{N}^2(i) = \bigcup_{j \in \mathcal{N}(i)} \mathcal{N}(j).
  $$

More generally, $\mathcal{N}^k(i)$ is all nodes within distance at most $k$ of $i$.

---

### 3.3 Fully-Connected Layer as a Starting Point

Consider a simple per-node fully-connected layer:

$$
\tilde{H} = \phi(X W + b),
$$
where:

- $W \in \mathbb{R}^{c \times c'}$,
- $b \in \mathbb{R}^{c'}$,
- $\phi$ is an activation (e.g., ReLU).

This layer is **permutation equivariant** (rows are processed independently), but it does **not use** $A$.

We want to extend it to be both permutation equivariant and **graph-aware**.

---

### 3.4 Definition of a Graph Convolutional (GC) Layer

We interleave per-node transformations with a diffusion step.

- **Definition (Graph convolutional layer)**  
  Given $(X, A)$, a **graph convolutional (GC) layer** is:
  $$
  H = f(X, A) = \phi\big( A (X W + b) \big),
  $$
  with parameters:

  - $W \in \mathbb{R}^{c\times c'}$,
  - $b \in \mathbb{R}^{c'}$.

Node-wise (ignoring the bias for clarity):

$$
H_i = \phi\left(
  \sum_{j\in\mathcal{N}(i)} A_{ij}\, X_j W
\right).
$$

Two-phase interpretation:

1. **Node-wise update**: $X \mapsto XW + b$ updates each node’s features independently (“channel mixing”).
2. **Neighborhood aggregation**: Multiply by $A$ to mix updated features across neighbors (“node mixing”).

Because $\mathcal{N}(i)$ sizes vary, using normalized adjacency (e.g., symmetric or row-normalized) often improves training stability.

Permutation equivariance holds:
$$
\begin{aligned}
f(PX, P A P^\top)
&= \phi\big( P A P^\top (PXW + b) \big) \\
&= \phi\big( P A (XW + b) \big) \\
&= P \phi\big( A (XW + b) \big) \\
&= P f(X, A).
\end{aligned}
$$

---

### 3.5 Receptive Fields of Stacked GC Layers

Consider a 2-layer GC network:

- **Two-layer GCN**
  $$
  f(X, A) = \phi\big( A \, \phi( A X W_1 ) W_2 \big),
  $$
  with trainable $W_1, W_2$.

We define:

- **Definition (Graph receptive field)**  
  For a GNN $H = f(X,A)$, the **receptive field** of node $i$ is the smallest set $\mathcal{R}(i) \subseteq \mathcal{V}$ such that:
  $$
  H_i = f\big( X_{\mathcal{R}(i)}, A_{\mathcal{R}(i)} \big).
  $$

For GC layers:

- **Single GC layer**: $\mathcal{R}(i) = \mathcal{N}(i)$.
- **Two GC layers**: $\mathcal{R}(i) = \mathcal{N}^2(i)$.
- **$k$ GC layers**: $\mathcal{R}(i) = \mathcal{N}^k(i)$.

The **graph diameter** $D$ is the minimum integer such that every pair of nodes is connected by a path of length at most $D$. To obtain a globally-aware node representation purely via GC layers, we generally need at least $D$ layers.

---

### 3.6 Polynomial GC Layers

We can increase receptive field **within a single layer** by using powers of $A$.

Assume $A$ has no self-loops (or handle them separately). Define:

- **Polynomial GC layer**
  $$
  H = \phi\big(
    X W_0 + A X W_1 + A^2 X W_2
  \big),
  $$
  where $W_0, W_1, W_2$ are learned.

Interpretation:

- $X W_0$: self-information.
- $A X W_1$: 1-hop neighbors.
- $A^2 X W_2$: 2-hop neighbors.

Generalizing:

$$
H = \phi\left(
  \sum_{k=0}^K A^k X W_k
\right).
$$

These are **polynomial filters** in the graph-shift matrix. More advanced designs use **rational filters** (ratios of polynomials) for more flexible spectral shaping.

---

### 3.7 Architectural Components and Pooling

A generic GCN architecture:

1. **Backbone** (stack of GC/polynomial/message-passing layers):
   $$
   H = f(X, A) \in \mathbb{R}^{n\times c'}.
   $$
2. **Head** $g$ for the task:
   $$
   y = (g \circ f)(X, A).
   $$

We can combine GC layers with:

- Normalization layers (batch norm, layer norm).
- Residual connections.
- Dropout and other regularization methods.

Pooling is more subtle than in images:

- There is no canonical way to “downsample” a graph (no grid).
- Graph pooling layers use either:
  - Predefined coarsening (graph clustering, hierarchical clustering).
  - Learnable pooling (attention-pooling, node selection, etc.).

However, in many practical GNN architectures, pooling is used only at the **final stage** (graph-level tasks), while intermediate representations remain at the original node resolution.

---

### 3.8 Node, Edge, and Graph Heads

Let $H = f(X, A)$ be the output of a GCN backbone.

#### 3.8.1 Node Classification

Given:

- Labels on a subset of nodes $\mathcal{V}_\ell \subseteq \mathcal{V}$.

We define a node-level head:

- **Node head**
  $$
  \hat{y}_i = g(H_i) = \operatorname{softmax}\big( \mathrm{MLP}(H_i) \big).
  $$

Training objective:

$$
\min_{f,g}
\frac{1}{|\mathcal{V}_\ell|}
\sum_{i\in\mathcal{V}_\ell}
\operatorname{CE}(\hat{y}_i, y_i),
$$
where CE is cross-entropy.

This is often **semi-supervised**: only a subset of nodes is labeled, but unlabeled nodes influence the learning through message passing.

#### 3.8.2 Edge Classification / Link Prediction

Given:

- Labels on a subset of edges $\mathcal{E}_\ell \subseteq \mathcal{E}$.

We define an edge-level head using pairs of node embeddings.

- **Edge head via concatenation**
  $$
  \hat{y}_{ij}
  = g(H_i, H_j)
  = \mathrm{MLP}\big( [H_i \,\|\, H_j] \big).
  $$

For scalar (e.g., binary) edge prediction, a common choice is:

- **Dot-product score**
  $$
  \hat{y}_{ij}
  = \sigma\big( H_i^\top H_j \big),
  $$
  where $\sigma$ is a sigmoid.

The loss is then a sum over known edges (and possibly negative samples).

#### 3.8.3 Graph Classification / Regression

For graph-level tasks, each graph has a label $y$.

Given multiple graphs $(X^{(g)}, A^{(g)})$ with corresponding $H^{(g)}$, we:

1. **Pool node embeddings** to get a fixed-size graph embedding:
   $$
   h^{(g)} = \frac{1}{n_g} \sum_{i=1}^{n_g} H^{(g)}_i
   \quad\text{or}\quad
   h^{(g)} = \sum_{i=1}^{n_g} H^{(g)}_i
   $$
   (mean or sum pooling; other options include max or attention-based pooling).

2. **Apply a head**:
   $$
   \hat{y}^{(g)} = \mathrm{MLP}\big(h^{(g)}\big).
   $$

Permutation invariance is guaranteed by the permutation-invariant pooling operation.

---

## 4. Implementation: Sparse Data Structures and Mini-Batching

### 4.1 Sparse Representations

Consider:
$$
A =
\begin{bmatrix}
0 & 0 & 1\\
0 & 0 & 0\\
1 & 0 & 0
\end{bmatrix},
$$
a 3-node graph with a single undirected edge between nodes 1 and 3.

Instead of storing the full matrix, we can store non-zero coordinates:

- **Coordinate list (COO)**
  $$
  \text{edges} = \{(0,2), (2,0)\}.
  $$

Sparse formats reduce memory and accelerate operations like:

- Sparse–dense matrix multiplication ($A X$).
- Sparse–sparse operations.

Libraries:

- **pytorch-sparse** (efficient sparse operations in PyTorch).
- **PyTorch Geometric** (GNN layers parameterized by feature matrices and edge index lists).
- Sparse functionality in JAX and other frameworks.

---

### 4.2 Mini-Batching Multiple Graphs

Suppose we have $b$ graphs $(X_i, A_i)$, $i=1,\dots,b$, with:

- $X_i \in \mathbb{R}^{n_i \times c}$,
- $A_i \in \{0,1\}^{n_i \times n_i}$.

#### 4.2.1 Naive Padding

One option: pad all graphs to size $n = \max_i n_i$:

- $X \in \mathbb{R}^{b \times n \times c}$,
- $A \in \{0,1\}^{b \times n \times n}$.

This wastes space and ignores natural sparsity.

#### 4.2.2 Block-Diagonal Union (Better Approach)

We can instead form a single **disjoint union** graph:

- Concatenate node features:
  $$
  X =
  \begin{bmatrix}
  X_1\\
  \vdots\\
  X_b
  \end{bmatrix}
  \in \mathbb{R}^{(\sum_i n_i)\times c}.
  $$
- Construct a block-diagonal adjacency:
  $$
  A =
  \begin{bmatrix}
  A_1 & 0 & \dots & 0\\
  0 & A_2 & \dots & 0\\
  \vdots & \vdots & \ddots & \vdots\\
  0 & 0 & \dots & A_b
  \end{bmatrix}
  \in \{0,1\}^{(\sum_i n_i)\times (\sum_i n_i)}.
  $$

Nodes from different graphs are disconnected; message passing within one connected component cannot reach another.

This representation:

- Preserves sparsity (in fact, often increases the sparsity ratio).
- Lets us treat batched graphs as a **single big graph**.

To keep track of which node belongs to which graph, define:

- **Graph index vector**
  $$
  g \in \{1,\dots,b\}^{\sum_i n_i},
  $$
  where $g_k$ is the ID of the graph to which node $k$ belongs.

---

### 4.3 Scatter-Based Pooling

Given:

- Node embeddings $H\in\mathbb{R}^{N\times c'}$, $N = \sum_i n_i$,
- Graph index vector $g$,

we define **scatter pooling**.

- **Scatter sum**
  $$
  Y = \operatorname{scatter\_sum}(H, g) \in \mathbb{R}^{b\times c'},
  $$
  where row $Y_k$ is:
  $$
  Y_k = \sum_{i : g_i = k} H_i.
  $$

This yields a graph embedding for each of the $b$ graphs in the batch. Similarly, we can define:

- $\operatorname{scatter\_mean}$ (average).
- $\operatorname{scatter\_max}$ (elementwise maximum).

These are standard primitives in GNN libraries and are crucial for efficient **graph-level** prediction in mini-batches.

---

### 4.4 Sampling Subgraphs from a Large Graph

When we have a **single, very large graph** that does not fit in memory:

- We sample smaller **induced subgraphs** to build mini-batches.
- Nodes in a batch are selected, and the subgraph induced by those nodes is used.
- Specialized sampling schemes (e.g., neighbor sampling, random walks, clustering) manage the trade-off between locality, coverage, and computational cost.

This is essential for large-scale applications such as web-scale social networks or huge knowledge graphs.

---

## 5. Beyond GC Layers: Attention and Message Passing

### 5.1 Homophily, Heterophily, and the Limits of Fixed Weights

GC layers use weights derived from $A$ (or its normalization). This assumes:

- All neighbors are **equally informative**, modulo normalization.
- Node labels/features tend to be **similar** across edges (homophily).

For **homophilic graphs** (e.g., citation networks where co-cited papers are related), this works well.

For **heterophilic graphs** (e.g., dating networks where edges may connect dissimilar users), uniform aggregation may **wash out** informative signals.

We thus seek **adaptive edge weights** that depend on features, not only on fixed connectivity.

---

### 5.2 Graph Attention Networks (GAT)

GAT introduces **attention** between neighboring nodes.

Let $x_i$ be the input features of node $i$.

- **Graph attention layer**
  $$
  h_i = \phi\left(
    \sum_{j\in\mathcal{N}(i)}
      \alpha_{ij} \, W^\top x_j
  \right),
  $$
  where:

  - $W$ is a linear projection of node features,
  - $\alpha_{ij}$ are learned attention weights,
  - $\phi$ is an activation function.

Attention scores:

- Compute raw scores:
  $$
  e_{ij} = \alpha(x_i, x_j),
  $$
  for some learnable function $\alpha$.
- Normalize via softmax over neighbors of $i$:
  $$
  \alpha_{ij} =
  \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}.
  $$

This preserves locality (only neighbors are considered) and permutation equivariance, but allows weights to be **feature-adaptive**.

#### 5.2.1 Original GAT Scoring

In the original GAT layer:

- **Attention score**
  $$
  e_{ij}
  = \operatorname{LeakyReLU}\big(
      a^\top [ V x_i \,\|\, V x_j ]
    \big),
  $$
  where:

  - $V$ is a learned linear projection,
  - $a$ is a learned vector,
  - $[\cdot \,\|\, \cdot]$ denotes concatenation.

Multiple attention heads can be used in parallel, similar to multi-head attention in transformers.

This parameterization has some expressiveness limitations in how it distinguishes the “central” node from its neighbors, which motivated improvements.

#### 5.2.2 GATv2

A variant called **GATv2** modifies the scoring to increase expressiveness, for example:

- **GATv2-style score**
  $$
  e_{ij} = a^\top \operatorname{LeakyReLU}\big(
    V [ x_i \,\|\, x_j ]
  \big),
  $$
  which allows more flexible dependence on the pair $(x_i,x_j)$.

Both GAT and GATv2 are widely used baselines that often outperform fixed-weight GCNs on heterophilic or more complex graphs.

---

### 5.3 Message-Passing Neural Networks (MPNNs)

GAT and GC layers can be seen as special cases of a general message-passing framework.

Assume:

- Node features $x_i$,
- (Optional) edge features $e_{ij}$.

- **Definition (Message-passing layer)**
  $$
  h_i = \psi\left(
    x_i,\;
    \operatorname{Aggr}\big(
      \{\, M(x_i, x_j, e_{ij}) : j \in \mathcal{N}(i) \,\}
    \big)
  \right),
  $$
  where:

  1. $M$: **message function** producing a vector from node and edge features.
  2. $\operatorname{Aggr}$: **permutation-invariant** aggregator (sum, mean, max, attention-weighted sum).
  3. $\psi$: **update function** combining $x_i$ with the aggregated message.

This template encompasses many GNN architectures and is commonly referred to as a **message-passing neural network (MPNN)**.

#### 5.3.1 GC Layer as an MPNN

The GC layer fits into this template:

- Messages:
  $$
  M(x_i, x_j, e_{ij}) = A_{ij} W^\top x_j.
  $$
- Aggregation:
  $$
  \operatorname{Aggr} = \sum_{j \in \mathcal{N}(i)} (\cdot).
  $$
- Update:
  $$
  \psi(x_i, m) = \phi(m).
  $$

Here, messages are **linear** in the neighbor’s features and scaled by $A_{ij}$.

#### 5.3.2 Emphasizing Central Node Features (GraphConv)

To more strongly incorporate $x_i$ in the update, we can use:

- **GraphConv-style update**
  $$
  \psi(x_i, m) = \phi(V x_i + m),
  $$
  where $V$ is a trainable matrix.

This is used in layers like `GraphConv` in PyTorch Geometric, improving expressiveness by blending central node information and aggregated messages.

#### 5.3.3 Spatio-Temporal Message Passing

If each node carries a **time series** $x_i$ (e.g., sensor measurements), we can use a temporal convolution inside the message:

- Let $x_i \in \mathbb{R}^{T \times c}$ be a sequence.
- Define:
  $$
  h_i =
    \sum_{j \in \mathcal{N}(i)}
      A_{ij} \, \mathrm{Conv1d}(x_j),
  $$
  where $\mathrm{Conv1d}$ operates along the temporal dimension.

This yields **spatio-temporal GNNs**, mixing temporal modeling with spatial message passing.

#### 5.3.4 Edge Feature Updates

To also update edge features, we can add an edge-update step:

- **Edge update**
  $$
  e_{ij} \leftarrow \mathrm{MLP}(e_{ij}, h_i, h_j).
  $$

Interpretation:

- Each edge collects messages from its incident nodes and updates its own feature.

This leads to more expressive architectures where both node and edge states are iteratively refined.

#### 5.3.5 Higher-Order and Equivariant GNNs (Overview)

Further generalizations (not detailed in these notes) include:

- **Higher-order / hypergraph message passing**: edges (hyperedges) connect more than two nodes; messages can be exchanged between nodes and higher-order structures.
- **Geometric / equivariant GNNs for point clouds**: message functions depend on relative positions and are constrained to be equivariant/invariant under translations and rotations, important for 3D data such as molecules or physical systems.

These constructions are extensions of the MPNN framework with additional symmetry constraints.

---

## 6. Graph Transformers

We have seen two ways to use graph structure:

1. As a **regularizer** (manifold regularization with Laplacian).
2. As a **constraint on computation** (GC/GAT/MPNN layers that follow adjacency).

Transformers, however, provide a flexible architecture that is **data-type agnostic**, using tokens and positional embeddings.

**Graph transformers** aim to:

- Use transformer architectures for graph data.
- Inject graph structure via **structural embeddings**.

---

### 6.1 Naive Transformer on Graph Nodes

Treat each node as a token. Ignoring edges, we could apply:

$$
H = \mathrm{Transformer}(X),
$$
where $X$ contains node features.

This is:

- Permutation equivariant if we treat node order as arbitrary.
- Completely **blind to adjacency**: any node can attend to any other, with no knowledge of which are connected in the graph.

We need a way to encode graph structure analogous to positional embeddings in sequences.

---

### 6.2 Structural Embeddings from Connectivity

Idea:

- Build **structural embeddings** from $A$ that encode structural position and roles of nodes.
- Add these to $X$ before feeding into the transformer.

- **Graph transformer with structural embedding**
  $$
  H = \mathrm{Transformer}\big( X + \mathrm{Embedding}(A) \big),
  $$
  where $\mathrm{Embedding}(A) \in \mathbb{R}^{n\times d}$ is a function of the graph connectivity only.

Each row of $\mathrm{Embedding}(A)$ is a vector summarizing node structure (e.g., centrality, local neighborhood patterns).

#### 6.2.1 Random-Walk-Based Structural Embeddings

Random walks provide a way to embed structural information:

- Define a (column-normalized) random-walk matrix:
  $$
  R = A D^{-1}.
  $$
  Under a suitable convention, $R_{ij}$ is the probability of transitioning from node $j$ to node $i$.

- Powers of $R$:
  $$
  R, R^2, \dots, R^k
  $$
  encode $t$-step transition probabilities.

Focus on the **return probabilities**:

- For each $t$, the diagonal $\operatorname{diag}(R^t)$ gives the probability that a random walk starting at a node returns to it after $t$ steps.

Stack these diagonals and project to a fixed-dimensional embedding:

- **Random-walk structural embedding**
  $$
  \mathrm{Embedding}(A)
  =
  \left[
  \begin{array}{c}
    \operatorname{diag}(R)\\
    \operatorname{diag}(R^2)\\
    \vdots\\
    \operatorname{diag}(R^k)
  \end{array}
  \right]^\top W,
  $$
  where $W$ is a learned projection.

Under certain conditions, such embeddings can uniquely represent nodes’ structural roles.

#### 6.2.2 Spectral Structural Embeddings

Another approach uses the eigen-decomposition of the Laplacian $L$:

- Use leading eigenvectors as a **spectral embedding**.
- These embeddings capture global connectivity patterns and can serve as positional encodings.

Both random-walk and spectral embeddings can be combined or extended with other structural features (e.g., shortest-path distances to a set of anchor nodes).

---

### 6.3 Graph Transformers and Foundation Models

With structural embeddings, we can apply standard transformer layers to graph data:

- Self-attention operates over all nodes, modulated by structural embeddings.
- This allows **long-range interactions** beyond local neighborhoods.
- It opens the possibility of **graph foundation models**: large pre-trained graph transformers, analogous to GPT-style models for text.

Graph transformers also facilitate **multimodal integration**:

- Graphs can be treated as one modality, integrated with text, images, etc., through a shared transformer backbone.

---

## 7. From Theory to Practice

### 7.1 Frameworks and Libraries

Efficient implementation of GNNs relies on:

- Sparse matrix operations (for adjacency, Laplacian).
- Graph-specific batching and pooling.
- Reusable layer primitives (GCN, GAT, MPNN variants).

Common libraries:

- **PyTorch Geometric (PyG)**:
  - Provides GC, GAT, GraphConv, and many other MPNN layers.
  - Uses edge index lists for sparse adjacency.
  - Includes datasets and tutorials (e.g., node classification on citation networks).

- **Jraph** (for JAX):
  - Provides graph data structures and GNN utilities.
  - Integrates with JAX’s JIT and functional programming style.

These libraries expose sparse-friendly API designs and handle many implementation details (e.g., scatter operations, batch handling, etc.).

---

### 7.2 A Didactic Exercise: GNNs for Vision

To connect convolutional networks, transformers, and GNNs:

1. **Tokenize an image into patches** (as in a Vision Transformer). Each patch becomes a node, with a patch embedding as its feature.
2. Build an adjacency matrix $A \in \{0,1\}^{p\times p}$ over the $p$ patches:
   - **Definition (Patch adjacency for a grid)**
     $$
     A_{ij} =
     \begin{cases}
     1, & \text{if patches } i \text{ and } j \text{ share a border in the image},\\
     0, & \text{otherwise}.
     \end{cases}
     $$
3. Treat $(X, A)$ as a graph:
   - $X$: patch embeddings,
   - $A$: grid connectivity of patches.
4. Apply a graph model (GCN, GAT, or message-passing network) to obtain updated patch embeddings.
5. Pool node embeddings to get an image-level representation and perform classification.

This recasts image classification as a **graph classification** problem on a regular image graph and illustrates how GNNs can be adapted to classical computer vision tasks.

---

## 8. Consolidated Key Points

- Graphs generalize sets, sequences, and images by encoding arbitrary sparse relations between entities.
- Adjacency, degree, and incidence matrices are foundational representations; normalized adjacency and Laplacian operators define diffusion and smoothness.
- The Laplacian quadratic form $x^\top L x$ measures smoothness of signals over graphs and underpins **manifold regularization**, which is especially useful in semi-supervised learning.
- **Graph convolutional layers** combine per-node transformations and neighborhood aggregation via a graph-shift matrix, yielding local, permutation-equivariant operations.
- Stacking GC layers or using polynomial filters expands the receptive field; task-specific heads enable node-, edge-, and graph-level predictions.
- Sparse representations, block-diagonal batching, and scatter pooling are crucial implementation techniques for scalable GNNs.
- **Graph attention networks (GAT)** and general **message-passing neural networks (MPNNs)** extend GCNs with adaptive, feature-dependent weights and flexible message and update functions, including edge features and spatio-temporal structure.
- **Graph transformers** embed structural information into node features and apply transformer architectures, paving the way for graph foundation models and multimodal integration.