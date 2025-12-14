12 | Graph models

About this chapter

In this chapter we consider graph-structured data, i.e.,
nodes connected by a set of (known) relations. Graph
are pervasive in the real world, ranging from proteins
to traffic networks, social networks, and recommender
systems. We introduce specialized layers to work on
graphs, broadly categorized as either message-passing
layers or graph transformers architectures.

12.1 Learning on graph-based data

12.1.1 Graphs and features on graphs

Up to now we have considered data which is either
completely unstructured (tabular data represented as a
vector) or structured in simple ways,
including sets,
sequences, and grids such as images. However, many
types of data are defined by more sophisticated
dependencies between its constituents.
For example,
molecules are composed by atoms which are only sparsely
connected via chemical bonds. Networks of many kinds
energy
(social networks,

transportation networks,

283


284

Learning on graph-based data

Figure F.12.1: Graphs generalize many types of data: sets can
be seen as empty graphs (or graphs having only self-loops), images
as regular graphs, and sequences as linear graphs. In this chapter
we look at more general graph structures.

networks) are composed of millions of units (people,
products, users) which interact only through a small set of
connections, e.g., roads, feedbacks, or friendships. These
are more naturally defined in the language of graph
theory.
this chapter is to introduce
differentiable models to work with data defined in such a
way.

The aim of

In its simplest form, a graph can be described by a pair of
sets (cid:71) = ((cid:86) , (cid:69) ), where (cid:86) = {1, . . . , n} is the set of nodes
(vertices), while:

(cid:69) = (cid:166)(i, j) |

i, j ∈ (cid:78) (cid:169)

Two nodes of the graph

is the set of edges present in the graph. In most datasets,
the number of nodes n and the number of edges m = |(cid:69) |
can vary from graph to graph.

Graph generalize many concepts we have already seen:
for example, graphs containing only self-loops of the form
(i, i) represent sets of objects, while graphs containing all
possible edges (fully-connected graphs) are connected to
Images can be
attention layers, as we show next.

284

ImageRegular graphSequenceLinear graphGeneric graphSetEmpty graph
Chapter 12: Graph models

285

represented as a graph by associating each pixel to a node
of the graph and connecting close pixels based on a
regular grid-like structure - see Figure F.12.1.1

Connections in a graph can be equivalently represented by
a matrix representation called the adjacency matrix. This
is a binary square matrix A ∼ Binary(n, n) such that:

=

Ai j

(cid:168)

1

0

if (i, j) ∈ (cid:69)
otherwise

In this format, a set is represented by the identity matrix
A = I, a fully-connected graph by a matrix of all ones, and
an image by a Toeplitz matrix. A graph where connections
are always bidirectional (i.e., (i, j) and ( j, i) are always
present as pairs among the edges) is called undirected,
and we have A⊤ = A. We will deal with undirected graphs
for simplicity, but the methods can be easily extended to
the directed case. We note that there are also alternative
the incidence matrix
matrix representations,
e.g.,
B ∼ Binary(n, |(cid:69) |) is such that Bi j
= 1 if node i
participates in edge j, and we have B1⊤ = 2 because each
edge connects exactly two nodes. See Figure F.12.2 for an
example.

= 1.
We will assume our graphs to have self-loops, i.e., Aii
If the adjacency matrix does not have self-loops, we can
add them by re-assigning it as:

A ← A + I

1There are many variants of this basic setup, including heterogenous
graphs (graphs with different types of nodes), directed graphs, signed
graphs, etc. Most of them can be handled by variations of the
techniques we describe next.

285


286

Learning on graph-based data

Figure F.12.2: We can represent the graph connectivity in three
ways: as a set (cid:69) of pairs (second column); as an (n, n) adjacency
matrix (third column); or as an (n, |(cid:69) |) incidence matrix (fourth
column).

12.1.2 Graph features

Graphs come with a variety of possible features describing
them. For example, atoms and bonds in a molecule can
be described by categorical features denoting their types;
roads in a transportation network can have a capacity and
a traffic flow; and two friends in a social networks can be
described by how many years they have known each other.

In general, these features can be of three types: node
features associated to each node, edge features associated
to each edge, and graph features associated to the entire
graph. We will begin with the simplest case of having access
to only unstructured node features, i.e., each node i has
∼ (c). The complete graph can then
associated a vector xi
be described by two matrices X ∼ (n, c), that we call the
feature matrix, and the adjacency matrix A ∼ (n, n).

In most cases, the ordering of the nodes is irrelevant, i.e.,
if we consider a permutation matrix P ∼ Binary(n, n) (see
Section 10.2), a graph and its permuted version are
fundamentally identical, in other words:

286

Neighbors ofnode 1 Nodes connectedby edge 2GraphSet formatAdjacency matrixIncidence matrix
Chapter 12: Graph models

287

(X, A) is the same graph as (PX, PAP⊤)

Note that the permutation matrix acts by swapping the
rows in X, while it swaps both rows and columns in the
adjacency matrix.

Some features can also be extracted directly from the
topology of the graph. For example, we can associate to
each node a scalar value di, called the degree, which
describes how many nodes it is connected to:

= (cid:88)

Ai j

di

j

The distribution of the degrees across the graph is an
important characteristic of the graph itself, as shown in
Figure F.12.3. We can collect the degrees into a single
diagonal matrix called the degree matrix:

D =





d1
. . . 0
...
...
...
0 . . . dn





We can use the degree matrix to define several types of
weighted adjacency matrices.
the
row-normalized adjacency matrix is defined as:

For example,

A′ ← D−1Ai j

→ A′
i j

= 1
di

Ai j

This is normalized in the sense that (cid:80)
= 1. We can
also define a column-normalized adjacency matrix as

i A′

i j

287


288

Learning on graph-based data

(a) Erd˝os–Rényi

(b) Degree

(c) Barabasi-Albert

(d) Degree

Figure F.12.3: (a) Random graph generated by drawing each
edge independently from a Bernoulli distribution (Erd˝os–Rényi
model). (b) These graphs show a Gaussian-like degree distribution.
(c) Random graph generated by adding nodes sequentially, and for
each of them drawing 3 connections towards existing nodes with a
probability proportional to their degree (preferential attachment
process or Barabasi-Albert model). (d) These graphs have a few
nodes with many connections acting as hubs for the graph.

A′ = AD−1. Both matrices can be interpreted as “random
walks” over the graph, in the sense that, given a node i,
the corresponding row or column of the normalized
adjacency matrix represents a probability distribution of
moving at random towards any of its neighbours. A more
general symmetrically normalized adjacency matrix is
given by:

A′ = D−1/2AD−1/2

This is defined by A′
i j

= Ai j(cid:112)

, giving a weight to each

di d j
connection based on the degree of both nodes it connects
to. Both the adjacency matrix and its weighted variants
= 0 whenever (i, j) /∈ (cid:69) . In signal
have the property that Ai j
processing terms, these are called graph-shift matrices.

288

24681012Degree024681012Nodes5101520Degree05101520Nodes
Chapter 12: Graph models

289

Sparsity in matrices

Consider a generic adjacency matrix for a 6-nodes graph
(try drawing the graph as an exercise):

A =










0 1 1 1 1 1
1 0 0 0 1 0
1 0 0 0 1 1
1 0 0 0 0 1
1 1 1 0 0 0
1 0 1 1 0 0










This
The adjacency is very sparse (many zeros).
is an important property, because sparse matrices
have customized implementations and techniques
for manipulating them, with better computational
complexity than their dense counterparts.a

aAs an example, in JAX: https://jax.readthedocs.io/en/

latest/jax.experimental.sparse.html.

12.1.3 Diffusion operations over graphs

The fundamental graph operation we are interested into is
something called diffusion, which corresponds to a
smoothing of the node features with respect to the graph
topology. To understand it, consider a scalar feature on
each node, that we collect in a vector x ∼ (n), and the
following operation over the features:

x′ = Ax

where A can be the adjacency matrix, a normalized variant,
or any weighted adjacency matrix. We can re-write this

289


290

Learning on graph-based data

operation node-wise as:

x ′
i

= (cid:88)
j∈(cid:78) (i)

Ai j x j

where we have defined the 1-hop neighborhood:

All edges with node i as a vertex

(cid:78) (i) = (cid:166)

j | (i, j) ∈ (cid:69) (cid:169)

If we interpret the node feature as a physical quantity,
projection by the adjacency matrix can be seen as a
“diffusion” process which replaces the quantity at each
node by a weighted average of the quantity in its
neighborhood.

Another fundamental matrix in the context of graph
analysis is the Laplacian matrix:

L = D − A

where the degree matrix is computed as Dii
j Ai j
irrespective of whether the adjacency matrix is normalized
or not. One step of diffusion by the Laplacian can be
written as:

= (cid:80)

[Lx]

i

= (cid:88)
(i, j)∈(cid:69)

Ai j

(x i

− x j

)

(E.12.1)

We can see from here that the Laplacian is intimately linked
to the idea of a gradient over a graph, and its analysis is
at the core of the field of spectral graph theory. As an

290


Chapter 12: Graph models

291

(a)
graph

Initial

(b) 10 steps

(c) 20 steps

(d) 30 steps

Figure F.12.4: (a) A random graph with 15 nodes and a scalar
feature on each node (denoted with variable colors). (b)-(d) The
result after 10, 20, and 30 steps of diffusion with the Laplacian
matrix. The features converge to a stable state.

example, in (E.12.1) 1 is always an eigenvector of the
Laplacian associated to a zero eigenvalue (in particular, the
smallest one). We show an example of diffusion with the
Laplacian matrix in Figure F.12.4.

12.1.4 Manifold regularization

From (E.12.1) we can also derive a quadratic form built
on the Laplacian:

x⊤Lx = (cid:88)

Ai j

(x i

− x j

)2

(i, j)∈(cid:69)

(E.12.2)

Informally, this is a scalar value that measures how
“smooth” the signal over the graph is, i.e., how quickly it
changes for pairs of nodes that are connected in the graph.
To see a simple application of this concept, consider a
tabular classification dataset (cid:83)
)}. Suppose we
build a graph over this dataset, where each node is an
element of the dataset, and the adjacency matrix is built
based on the distance between features:

= {(xi, yi

n

291


292

Learning on graph-based data

=

Ai j

(cid:168)

exp(−∥xi
0

− x j

∥2)

∥2 < τ

− x j
if ∥xi
otherwise

(E.12.3)

where τ is a user-defined hyper-parameter. Given a
classification model f (x), we may want to constrain its
output to be similar for similar inputs, where similarity is
defined proportionally to (E.12.3). To this end, we can
define the features of the graph as the outputs of our
model:

f =





f (x1
...
f (xn

)


 ∼ (n)
)

The quadratic form (E.12.2) tells us exactly how much
similar inputs vary in terms of their predictions:

f⊤Lf = (cid:88)

i, j

Ai j

( f (xi

) − f (x j

))2

(E.12.4)

The optimal model can be found by a regularized
optimization problem, where the regularizer is given by
(E.12.4) :

f ∗(x) = arg min

L( yi, f (x)) + λ f⊤Lf

(cid:171)

(cid:168) n

(cid:88)

i=1

where L is a generic loss function and λ is a scalar hyper-
parameter:

This is called manifold regularization [BNS06] and it can
be used as a generic regularization tool to force the model

292


Chapter 12: Graph models

293

to be smooth over a graph, where the adjacency is either
given or is built by the user as in (E.12.3). This is especially
helpful in a semi-supervised scenario where we have a
small labeled dataset and a large unlabeled one from the
same distribution, since the regularizer in (E.12.4) does
not require labels [BNS06]. However, the prediction of
the model depends only on a single element xi, and the
graph is thrown away after training. In the next section,
we will introduce more natural ways of embedding the
connectivity inside the model itself.

12.2 Graph convolutional layers

12.2.1 Properties of a graph layer

In order to design models whose predictions are conditional
on the connectivity, we can augment standard layers f (X)
with knowledge of the adjacency matrix, i.e., we consider
layers of the form:

H = f (X, A)

where as before X ∼ (n, c) (with n the number of nodes
and c the features at each node) and H ∼ (n, c′), i.e., the
operation does not change the connectivity of the graph,
∼ (c′) for each
and it returns an updated embedding Hi
node i in the graph. For what follows, A can be the
adjacency or any matrix with the same sparsity pattern (a
including a weighted adjacency
graph-shift matrix),
matrix, the Laplacian matrix, and so on.

Since permuting the nodes in a graph should have no
impact on the final predictions, the layer should not
depend on the specific ordering of the nodes, i.e., for any
permutation matrix P the output of the layer should be

293


294

Graph convolutional layers

permutation equivariant:

f (PX, PAP⊤) = P · f (X, A)

We can define a notion of “locality” for a graph layer,
similar to the image case. To this end, we first introduce
the concept of a subgraph. Given a subset of nodes (cid:84) ∈ (cid:86)
from the full graph, we define the subgraph induced by (cid:84)
as:

(cid:71)(cid:84) = (X(cid:84) , A(cid:84) )
where X(cid:84) is a (|(cid:84) |, c) matrix collecting the features of the
nodes in (cid:84) , and A ∼ (|(cid:84) |, |(cid:84) |) is the corresponding block
of the full adjacency matrix.

Definition D.12.1 (Graph locality)

A graph layer H = f (X, A) is local if for every node, Hi
=
f (X(cid:78) (i), A(cid:78) (i)), where (cid:78) (i) is the 1-hop neighborhood
of node i.

This is similar to considering all pixels at distance 1 in the
image case, except that (a) nodes in (cid:78) (i) have no specific
ordering in this case, and (b) the size of (cid:78) (i) can vary a lot
depending on i. Hence, we cannot define a convolution like
we did in the image case, as its definition requires these two
properties (think of the weight tensor in a convolutional
layer).

For what follows, note that we can extend our definition of
locality beyond 1-hop neighbors. For example, the 2-hop
neighborhood (cid:78) 2(i) is defined as all nodes at distance at
most 2:

(cid:78) 2(i) = (cid:91)
j∈(cid:78) (i)

(cid:78) ( j)

where ∪ is the set union operator. We can extend the

294


Chapter 12: Graph models

295

definition of locality to take higher-order neighborhoods
into consideration and design the equivalent of 3 × 3 filters,
5 × 5 filters, and so on.

12.2.2 The graph convolutional layer

layer, we need it

In order to define a graph layer that mimicks the
convolutional
to be permutation
equivariant (instead of translation equivariant) and local.
The MHA layer is naturally permutation equivariant, but it
is not local and it does not depend explicitly on the
adjacency matrix A. We will see possible extensions to this
end in the next section. For now, let us focus on a simpler
fully-connected layer:

f (X, _) = φ(XW + b)

where W ∼ (c, c′) and b ∼ (c′). This is also naturally
permutation equivariant, but it does not depend on the
connectivity of the graph, which is ignored. To build an
appropriate differentiable layer, we can alternate the layer’s
operation with a diffusion step.

Definition D.12.2 (Graph convolution)

Given a graph represented by a node feature matrix X ∼
(n, c) and a generic graph-shift matrix A ∼ (n, n) (the
adjacency, the Laplacian, ...), a graph convolutional
(GC) layer is given by [KW17]:

f (X, A) = φ(A(XW + b))

where the trainable parameters are W ∼ (c, c′) and

295


296

Graph convolutional layers

Figure F.12.5: Two stages of a GC layer: each node updates
its embedding in parallel to all other nodes; the output is given
by a weighted average of all updated embeddings in the node’s
neighbourhood.

b ∼ (c′), with c′ an hyper-parameter. φ is a standard
activation function, such as a ReLU.

Note the similarity with a standard convolutional layer: we
are performing a “channel mixing” operation via the matrix
W, and a “node mixing” operation via the matrix A, the
difference being that the former is untrainable in this case
(due to, once again, variable degrees between nodes and
the need to make the layer permutation equivariant). See
Figure F.12.5 for a visualization. The analogy can also be
justified more formally by leveraging concepts from graph
signal processing, which is beyond the scope of this book
[BBL+17].

Ignoring the bias, we can rewrite this for a single node i
as:

(cid:32)

(cid:88)

(cid:33)

Ai jX jW

= φ

Hi

j∈(cid:78) (i)

296

Node-wise update:Aggregation:
Chapter 12: Graph models

297

Hence, we first perform a simultaneous update of all node
embeddings (given by the right multiplication by W).
Then, each node computes a weighted average of the
updated node embeddings from itself and its neighbors.
Since the number of neighbors can vary from node to
node, working with the normalized variants of the
adjacency matrix can help significantly in training. It is
trivial to show permutation equivariance for the layer:

f (PX, PAP⊤) = φ (cid:0)PAP⊤PXW(cid:1) = P · f (X, A)

12.2.3 Building a graph convolutional

network

A single GC layer is local, but the stack of multiple layers
is not. For example, consider a two-layered GC model:

f (X, A) = φ(A φ (AXW1

) W2

)

(E.12.5)

First GC layer

with two trainable weight matrices W1 and W2. Similarly
to the image case, we can define a notion of receptive field.

Definition D.12.3 (Graph receptive field)

Given a generic graph neural network H = f (X, A), the
receptive field of node i is the smallest set of nodes (cid:86) (i) ∈
(cid:86) such that Hi

= f (X(cid:86) (i), A(cid:86) (i)).

For a single GC layer, the receptive field is (cid:86) (i) = (cid:78) (i).
For a two-layer network as in (E.12.5), we need to
consider neighbors of neighbors, and the receptive field

297


298

Graph convolutional layers

becomes (cid:86) (i) = (cid:78) 2(i). In general, for a stack of k layers
we will have a receptive field of (cid:86) (i) = (cid:78) k(i). The
smallest number of steps which is needed to move from
any two nodes in the graph is called the diameter of the
graph. The diameter defines the smallest number of layers
which is required to achieve a global receptive field for all
the nodes.

Polynomial GC layers

Alternatively, we can increase the receptive field of a
single GC layer. For example, if we remove the self-
loops from the adjacency matrix, we can make the layer
local with respect to (cid:78) 2(i) instead of (cid:78) (i) by also
considering the square of the adjacency matrix:

H = φ (cid:0)XW0

+ AXW1

+ A2XW2

(cid:1)

where we have three sets of parameters W0, W1, and
W2 to handle self-loops, neighbors, and neighbors of
neighbors respectively. This is called a polynomial
GC layer. Larger receptive fields can be obtained with
higher powers. More complex layers can be designed by
considering ratios of polynomials [BGLA21].

We can combine GC layers with standard normalization
residual connections, dropout, or any other
layers,
operation that is permutation equivariant. Differently
from the image case, pooling is harder because there is no
immediate way to subsample a graph connectivity. Pooling
layers can still be defined by leveraging tools from graph
theory or adding additional trainable components, but
they are less common [GZBA22].

Denote by H = f (X, A) a generic combination of layers
providing an updated embedding for each node (without

298


Chapter 12: Graph models

299

Figure F.12.6: Different types of graph heads: (a) node tasks
need to process the features of a single node; (b) edge tasks require
heads that are conditioned on two nodes simultaneously; (c) graph
tasks can be achieved by pooling all node representations into a
fixed-dimensional vector.

modifying the connectivity). In analogy with CNNs, we call
it the backbone network. We can complete the design of a
generic graph convolutional network (GCN) by adding a
small head of top of these representations:

y = (g ◦ f )(X, A)

The design of the head depends on the task we are trying
to solve. The most common tasks fall into one of three
basic categories: node-level tasks (e.g., node classification),
edge-level task (e.g., edge classification), or graph-level
tasks (e.g., graph classification). We briefly consider an
example for each of them in turn, see Figure F.12.6.

Node classification

First, suppose the input graph describes some kind of social
network, where each user is associated to a node. For a
given subset of users, (cid:84) ⊆ (cid:86) , we know a label yi, i ∈ (cid:84)
(e.g., whether the user if a real user, a bot, or another
kind of automated profile). We are interested in predicting
the label for all other nodes. In this case, we can obtain
a node-wise prediction by processing each updated node
embedding, e.g.:

299

GCNBackboneNode headEdge headGraph headAverageNodepredictionEdgepredictionGraphprediction
300

Graph convolutional layers

ˆyi

= g(Hi

) = softmax(MLP(Hi

))

Running this operation over the entire matrix H gives us a
prediction for all nodes, but we only know the true labels
for a small subset. We can train the GCN by discarding the
nodes outside of the training set:

arg min

1
|(cid:84) |

(cid:88)

i∈(cid:84)

CE( ˆyi, yi

)

where CE is the cross-entropy loss. Importantly, even if we
are discarding the output predictions for nodes outside our
training set, their input features are still involved in the
training process due to the diffusion steps inside the GCN.
The rest of the nodes can then be classified by running
the GCN a final time after training. This scenario, where
only a subset of the training data is labeled, is called a
semi-supervised problem.

Edge classification

E

As a second example, suppose we have a label for a subset
of edges, i.e., (cid:84)
⊆ (cid:69) . As an example, our graph could be a
traffic network, of which we know the traffic flow only on
a subset of roads. In this case, we can obtain an edge-wise
prediction by adding an head that depends on the features
of the two connected nodes, e.g., by concatenating them:

ˆyi j

= g(Hi, H j

) = MLP (cid:0)(cid:2)Hi

∥ H j

(cid:3)(cid:1)

For binary classification (e.g., predicting the affinity of
two users with a scalar value between 0 and 1) we can
simplify this by considering the dot product between the

300


Chapter 12: Graph models

301

two features:

ˆyi j

= σ(H⊤

i H j

)

Like before, we can train the network by minimizing a loss
over the known edges.

Graph classification

Finally, suppose we are interested in classifying (or
regressing) the entire graph. As an example, the graph
could be a molecule of which we want to predict some
chemical property, such as reactivity against a given
compound. We can achieve this by pooling the node
representations (e.g., via a sum), and processing the
resulting fixed-dimensional embedding:

y = MLP

(cid:130)

(cid:140)

1
n

n
(cid:88)

i=1

Hi

The final pooling layer makes the network invariant to the
permutation of the nodes. In this case, our dataset will
be composed of multiple graphs (e.g., several molecules),
making it similar to a standard image classification task.
For node and edge tasks, instead, some datasets may be
composed of a single graph (e.g., a large social network),
while other datasets can have more than a single graph
(e.g., several unconnected road networks from different
towns). This opens up the question of how to efficiently
build mini-batches of graphs.

301


302

Graph convolutional layers

12.2.4 On the implementation of graph

neural networks

As mentioned, the peculiarity of working with graphs is that
several matrices can be very sparse. For example, consider
the following adjacency matrix:

A =





0 0 1
0 0 0
1 0 0





This corresponds to a three-node graph with a single
bidirectional edge between nodes 1 and 3. We can store
this more efficiently by only storing the indices of the
non-zero values, e.g., in code:

A = [[0,2], [2,0]]

This is called a coordinate list format. For very sparse
matrices, specialized formats like this one can reduce
storage but also significantly improve the runtime of
operating on sparse matrices or on combinations of sparse
and dense matrices. As an example, pytorch-sparse2
supports highly-efficient implementations of transposition
and several types of matrix multiplications in PyTorch.
This is also reflected on the layers’ implementation. The
forward pass of the layers in PyTorch Geometric3 (one of
the most common libraries for working with graph neural
networks in PyTorch) is parameterized by providing as
inputs the features of the graph and the connectivity as a
list of edge coordinates.

2https://github.com/rusty1s/pytorch_sparse
3https://pytorch-geometric.readthedocs.io/en/latest/get_

started/introduction.html#learning-methods-on-graphs

302


Chapter 12: Graph models

303

Figure F.12.7: Two graphs
in a mini-batch can be seen
as a single graph with two
disconnected components. In
order to distinguish them,
we need to introduce an
additional vector containing
the mapping between nodes
and graph IDs.

Working with sparse matrices has another interesting
consequence in terms of mini-batches. Suppose we have b
graphs (Xi, Ai
)b
i=1. For each graph we have the same
number of node features c but a different number of nodes
). In order to
ni, so that Xi
build a mini-batch, we can create two rank-3 tensors:

∼ (ni, c) and Ai

∼ Binary(ni, ni

X ∼ (b, n, c)
A ∼ Binary(b, n, n)

(E.12.6)

(E.12.7)

where n = max(n1, . . . , nb
), and both matrices are padded
with zeros to fill up the two tensors. However, a more
elegant alternative can be obtained by noting that in a GC
layer, two nodes that are not connected by any path (a
sequence of edges) will never communicate. Hence, we
can build a single graph describing the entire mini-batch
by simply merging all the nodes:

X =









X1
...
Xb
A1
. . . 0
...
...
...
0 . . . Ab





A =

303

(E.12.8)

(E.12.9)





1324567
304

Graph convolutional layers

i ni

i ni, (cid:80)

where X ∼ ((cid:80)
i ni, c) and A ∼ Binary((cid:80)
). The
adjacency matrix of the mini-batch has a block-diagonal
structure, where all elements outside the diagonal blocks
are zero (nodes from different graphs are not connected).
While seemingly wasteful, this actually increases the
sparsity ratio of the graph, making better use of the sparse
matrix operations. Hence, for graph datasets in many
cases there is no real difference between working with a
single graph or a mini-batch of graphs.

To keep track of the correspondence between nodes and
graphs, we can augment the representation with an
additional vector b ∼ ((cid:80)
) such that bi is an index in
i ni
[0, . . . , b − 1] identifying one of the b input graphs - see
Figure F.12.7. For graph classification, we can exploit b to
perform pooling separately on groups of nodes
corresponding to different graphs. Suppose H ∼ (n, c′) is
the output of the GCN backbone, then:

scatter_sum (H, b) = Y ∼ (b, c′)

(E.12.10)

is called a scattered sum operation, and it is defined such
= i, as shown
that Yi is the sum of all rows of H where b j
in Figure F.12.8. Similar operations can be defined for
other types of pooling operations, including averages and
maximums.

As a separate problem, sometimes we may have a single
graph that does not fit into memory: in this case, mini-
batches should be formed by sampling subgraphs from the
original graph [HYL17]. This is a relatively complex task
that goes beyond the scope of this chapter.

304


Chapter 12: Graph models

305

Figure F.12.8: Example of scattered sum on the graph of Figure
F.12.7. In this example nodes (1,2,3,4) belong to graph 1, and
nodes (5,6,7) to graph 2. After pooling, we obtain a pooled
representation for each of the two graphs.

12.3 Beyond graph convolutional

layers

With the GC layer as a template, we now overview a few
extensions, either in terms of adaptivity or graph features
that can be handled. We close by discussing graph
transformers, a different family of layers in which the
graph is embedded into a structural embedding which is
summed to the node features.

12.3.1 Graph attention layers

One issue with GC layers is that the weights that are used
to sum up contributions from the neighborhoods are fixed
and are given by the adjacency matrix (or a proper
normalized variant). This is equivalent to the assumption
that, apart from the relative number of connections, all
neighbors are similarly important. A graph where nodes
are connected mostly with similar nodes is called
homophilic: empirically, homophily is a good predictor of
the performance of graph convolutional layers [LLLG22].

305

SumSum
306

Beyond graph convolutional layers

Not all graphs are homophilic: for example, in a dating
network, most people will be connected with people from
the opposite sex. Hence, in these scenarios we need
techniques that can adapt the weights across nodes.

For sufficiently small graphs, we can let the non-zero
elements of the weight matrix A adapt from their starting
value through gradient descent. However, the number of
trainable parameters in this case increases quadratically
with the number of nodes, and this solution does not apply
to a scenario with more than a single graph. If we assume
that an edge depends only on the features of the two
nodes it connects, we can generalize the GC layer with an
attention-like operation:

= φ

hi

(cid:32)

(cid:88)

j∈(cid:78) (i)

softmax(α(xi, x j

))W⊤x j

(cid:33)

where α is some generic MLP block having two inputs and a
scalar output, and the softmax is applied, for each node, to
the set of outputs of α with respect to (cid:78) (i), to normalize
the weights irrespective of the size of the neighborhood.
Due to the similarity to the attention layer, these are called
graph attention (GAT) layers [VCC+18]. Seen from the
perspective of the entire graph, this is very similar to a
MHA layer, where the attention operation is restricted only
on nodes having an edge that connects them.

The choice of α is relatively free. Instead of a dot product,
the original GAT formulation considered an MLP applied
on a concatenation of features:

α(xi, x j

) = LeakyReLU(a⊤ (cid:2)Vxi

∥ Vx j

(cid:3))

306


Chapter 12: Graph models

307

in the sense that

where V and a are trainable. This was later found to be
the ordering between
restrictive,
elements does not depend on the central node [BAY22]. A
less restrictive variant, called GATv2 [BAY22] is obtained
as:

α(xi, x j

) = a⊤LeakyReLU(V (cid:2)xi

(cid:3))

∥ x j

Both GAT and GATv2 are very popular baselines nowadays.

12.3.2 Message-passing neural networks

Suppose we have available additional edge features ei j,
e.g., in a molecular dataset we may know a one-hot
encoded representation of the type of each molecular
bond. We can generalize the GAT layer to include these
features by properly modifying the attention function:

α(xi, x j

) = a⊤LeakyReLU(V (cid:2)xi

∥ x j

∥ ei j

(cid:3))

We can further generalize all the layers seen up to now
(GC, GAT, GATv2, GAT with edge features) by abstracting
away their basic components. Consider a very general layer
formulation:

= ψ (cid:128)

hi

xi, Aggr

(cid:128)(cid:8)M (xi, x j, ei j

)(cid:9)

(cid:138)(cid:138)

(cid:78) (i)

(E.12.11)

where:

1. M builds a feature vector (which we call a message)
relative to the edge between node i and node j.
Contrary to GC and GAT layers, we are not

307


308

Beyond graph convolutional layers

restricting the message to be scalar-valued.

2. Aggr is a generic permutation invariant function (e.g.,
a sum) to aggregate the messages from all nodes
connected to node i.

3. ψ is a final block that combines the aggregated
message with the node features xi. In this way, two
nodes with the same neighborhood can still be
distinguished.

As an example, in a GC layer the message is built as
M (_, x j, _) = Ai jW⊤x j, the aggregation is a simple sum,
and ψ(_, x) = φ(x). The general layer (E.12.11) was
[GSR+17] with
introduced
of
message-passing layer, and it has become a very popular
way to categorize (and generalize) layers operating on
graphs [Vel22].

name

the

in

Let us
consider a few examples of using this
message-passing framework. First, we may want to give
more
in the
the
message-passing phase. We can do this by modifying the
ψ function:

central node

importance

to

ψ(x, m) = φ(Vx + m)

where V is a generically trainable matrix (this was
introduced in [MRF+19] and popularized in PyTorch
Geometric as the GraphConv4 layer). Second, suppose
nodes have available more complex features such as a
time series per node (e.g., a distributed set of sensors).
Note that in the message-passing framework, node-wise

4https://pytorch-geometric.readthedocs.io/en/latest/generated/

torch_geometric.nn.conv.GraphConv.html

308


Chapter 12: Graph models

309

operations are decoupled from the way messages are
aggregated and processed. Denoting by x i the time-series
at node i, we can generalize the GC layer by simply
modifying the message function with a layer working on
time series, e.g., a Conv1d layer:

hi

= (cid:88)
j∈(cid:78) (i)

Ai jConv1d(xi

)

This is an example of a spatio-temporal GC layer [YYZ17].
Furthermore, up to now we have assumed that only node
features should be updated. However, it is easy to also
update edge features by an additional edge update layer:

ei j

← MLP(ei j, hi, h j

)

This can also be seen as a message-passing iteration, in
which the edge aggregates messages from its neighbors
(the two connected nodes). This line of reasoning allows to
further generalize these layers to consider more extended
neighborhoods and graph features [BHB+18].

This is a very brief overview that provides a gist of many
possible message-passing variants. There are many topics
we are not able to cover in detail due to space: among
these, we single out building MP layers for higher-order
graphs (in which edges connect more that a pair of nodes)
[CPPM22] and MP layers for point cloud data, in which
we are interested in satisfying additional symmetries
(rotational
symmetries)
[SHW21, EHB23].

translational

and

309


310

Beyond graph convolutional layers

12.3.3 Graph transformers

We have seen two techniques to employ the graph structure:
the first one is to add a regularization term that forces the
network’s outputs to be smooth relative to the graph; the
second one is to constrain the operations of the graph to
follow the graph connectivity. In particular, in the GAT layer
we have used a standard attention operation by properly
masking the pairwise comparisons. However, we have
also seen in the previous chapter that transformers have
become popular because they provide an architecture that
is completely agnostic from the type of data. Can we design
the equivalent of a graph transformer [MGMR24]?

Recall that the two basic steps for building a transformer
are tokenization of the input data and definition of the
positional embedding. Tokenization for a graph is simple:
for example, we can consider each node as a token, or
(if edge features are given) each node and each edge as
separate tokens after embedding them in a shared space.
Let us ignore for now edge features. Consider the generic
architecture taking as input the node features:

H = Transformer(X)

This is permutation equivariant but completely agnostic to
the connectivity. We can partially solve this by augmenting
the node features with some graph-based features, such
as the degree of the node, or the shortest path distance to
some pre-selected nodes (anchors) [RGD+22, MGMR24].
More in general, however, we can consider an embedding
of the graph connectivity into what we call a structural
embedding:

310


Chapter 12: Graph models

311

Figure F.12.9: General idea of a graph transformer:
the
connectivity is embedded into a set of positional embeddings, which
are added to the collected features. The result is then processed by
a standard transformer network.

H = Transformer(X + Embedding(A))

Each row of Embedding(A) provides a vectorial
embedding of the connectivity of the graph relative to a
single node,
ignoring all features (see Figure F.12.9).
Luckily, embedding the structure of a graph into a vector
space is a broad field. As an example, we describe here a
common embedding procedure based on random walks
[DLL+22]. Recall that the following matrix:

R = AD−1

can be interpreted as a “random walk”, in which Ri j is the
probability of moving from node i to node j. We can iterate
the random walk multiple times, for a fixed k set a priori
by the user:

R, R2, . . . , Rk

Random walk embeddings are built by collecting all the
walk probabilities of a node returning on itself, and
projecting them to a fixed-dimensional embedding:

311

PositionalembeddingsTransformer
312

Beyond graph convolutional layers

Embedding(A) =













diag(R)
diag(R2)
...
diag(Rk)

W

Under specific conditions on the graph structure, this can
be shown to provide a unique representation for each
node [DLL+22]. Alternative types of embeddings can be
obtained by considering eigen-decompositions of the
Laplacian matrix [LRZ+23]. For a fuller exposition of
graph transformers, we refer to [MGMR24]. Building
graph transformers opens up the possibility of GPT-like
foundation models for the graph domain, and also of
adding graph-based data as an additional modality to
existing language models [MCT+].

From theory to practice

Handling efficiently graph data requires
extensions of the basic frameworks,
due to the problems described in this
chapter (e.g., sparsity).
Common
libraries include PyTorch Geometric for
PyTorch, and Jraph for JAX. Both have ample sets of
tutorials, for example for node classification in small
citation networks.5

If you implemented a Vision Transformer in Chapter 11, I
suggest a funny exercise which has (mostly) didactic value,
as shown in Figure F.12.10. Suppose we tokenize the image
into patches, but instead of adding positional embeddings,

5Recommended example in PyTorch Geometric: https://pytorch-

geometric.readthedocs.io/en/latest/get_started/introduction.html.

312


Chapter 12: Graph models

313

Figure F.12.10: A GNN for computer vision:
the image is
tokenized into patches, an adjacency matrix is built over the patches,
and the two are propagated through a graph model.

we construct an adjacency matrix A ∼ (p, p) (where p is
the number of patches) as:

=

Ai j

(cid:168)

1

0

if the two patches share a border in the image

otherwise

(E.12.12)

We now have a graph classification dataset, where the
node features are given by the patch embedding, and the
adjacency matrix by (E.12.12). Thus, we can perform
image classification by adapting the GNN from the
previously-mentioned tutorials.

313

Image tokenizerGraph model
314

Beyond graph convolutional layers

314
