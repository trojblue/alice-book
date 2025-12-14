10 | Transformer
models

About this chapter

Convolutional models are strong baselines, especially
for images and sequences where local relations prevail,
but they are limited in handling very long sequences or
non-local dependencies between elements of a sequence.
In this chapter we introduce another class of models,
called transformers, which are designed to overcome
such challenges.

10.1 Long convolutions and
non-local models

After the key developments in the period 2012-2016,
discussed in the previous chapter, the next important
breakthrough in the design of differentiable models came
in 2016-2017 with the popularization of the transformer
[VSP+17], an architecture designed to handle efficiently
long-range dependencies in natural language processing.
Due to its strong scaling laws, the architecture was then

239


240

Long convolutions and non-local models

extended to other types of data, from images to time-series
and graphs, and it is today a state-of-the-art model in
many fields due to its very good scaling laws when trained
on large amounts of data [KMH+20, BPA+24].

As we will see, an interesting aspect of the transformer is a
decoupling between the data type (through the use of
appropriate tokenizers) and the architecture, which for
the most part remains data-agnostic. This opens up
several interesting directions, such as simple multimodal
architectures and transfer learning strategies. We begin by
motivating the core component of the transformer, called
the multi-head attention (MHA) layer. We will defer a
discussion on the original
from
[VSP+17] to the next chapter.

transformer model

A bit of history

in 2015,
Historically, this chapter is out of order:
the most common alternative to CNNs for text were
recurrent neural networks (RNNs). As an isolated
component, MHA was introduced for RNNs [BCB15],
before being used as the core component in the
transformer model. We cover RNNs and their
modern incarnation, linearized RNNs, in Chapter 13.
Recently, RNNs have become an attractive competitor to
transformers for language modeling.

240


Chapter 10: Transformer models

241

10.1.1 Handling long-range and sparse

dependencies

Consider these two sentences:

“The cat is on the table”

and a longer one:

“The cat, who belongs to my mother, is on the
table”.

In order to be processed by a differentiable model, the
sentences must be tokenized and the tokens embedded as
vectors (Chapter 8). From a semantic point of view, the
tokens belonging to the red word (cat) and to the green
word (table) share a similar dependency in both sentences.
However, their relative offset varies in the two cases, and
their distance can become arbitrarily large. Hence,
dependencies in text can be both long-range and
input-dependent.

Denote by X ∼ (n, e) a sentence of n tokens embedded in
e-dimensional vectors and denote by xi the ith token. We
can rewrite a 1D convolution with kernel size k on token i
as follows:

2k+1
(cid:88)

=

hi

W jxi+k+1− j

(E.10.1)

j=1

Each token inside the receptive field is processed with a
fixed weight matrix Wi that only depends on the specific
offset i. Modeling long-range dependencies inside the
layer requires us to increase the receptive field of the layer,
increasing the number of parameters linearly in the
receptive field.

One possibility to solve this is the following: instead of

241


242

Long convolutions and non-local models

Figure F.10.1: Comparison between different types of convolution
for a 1D sequence. We show how one output token (in red )
interacts with two tokens, one inside the receptive field of the
convolution (in green ), and one outside (in blue ). (a) In a
standard convolution, the blue token is ignored because it is outside
of the receptive field of the filter. (b) For a continuous convolution,
both tokens are considered, and the resulting weight matrices are
given by g(−1) and g(2) respectively. (c) In the non-local case,
the weight matrices depend on a pairwise comparison between the
tokens themselves.

explicitly learning the matrices W1, W2, . . ., we can define
them implicitly by defining a separate neural block g(i) :
(cid:82) → (cid:82)e×e that outputs all weight matrices based on the
relative offset i. Hence, we rewrite (E.10.1) as:

=

hi

n
(cid:88)

j=1

g(i − j)x j

The sum is now on all tokens

This is called a long convolution, as the convolution
spans the entire input matrix X.
It is also called a
continuous convolution [RKG+22], because we can use
g(•) to parameterize intermediate positions or variable
resolutions [RKG+22]. The number of parameters in this
case only depends on the parameters of g, while it does

242

InputtokensOutputtokens(a) Conv1D, kernel size 3(b) Continuous convolution(c) Non-local modelOutput tokenInput token inside theConv1D receptive fieldInput token outside theConv1D receptive field
Chapter 10: Transformer models

243

not depend on n, the length of the sequence. Defining g is
non-trivial because it needs to output an entire weight
matrix. We can recover a standard convolution easily:

g(i, j) =

(cid:168)

Wi− j
0

if |i − j| ≤ k
otherwise

(E.10.2)

solves

the problem of

long-range
This partially
dependencies, but it does not solve the problem of
dependencies which are conditional on the input, since
the weight given to a token depends only on the relative
offset with respect to the index i.
this
formulation provides a simple way to tackle this problem
by letting the trained function g depend on the content of
the tokens instead of their positions:

However,

=

hi

n
(cid:88)

j=1

g(xi, x j

)x j

(E.10.3)

In the context of computer vision, these models are also
called non-local networks [WGGH18]. We provide a
comparison of
continuous
convolutions, and non-local convolutions in Figure F.10.1.

standard convolutions,

10.1.2 The attention layer

The MHA layer is a simplification of (E.10.3).
First,
working with functions having matrix outputs is difficult,
so we restrict the layer to work with scalar weights. In
particular, a simple measure of similarity between tokens
is their inner (dot) product:

243


244

Long convolutions and non-local models

g(xi, x j

) = x⊤

i x j

As we will see, this results in an easily parallelizable
algorithm for the entire sequence. For the following we
consider a normalized version of the dot-product:

g(xi, x j

) = 1
(cid:112)
e

x⊤
i x j

∼ (cid:78) (0, σ2I), the variance of each element of x⊤

if we assume
This can be motivated as follows:
i x j is σ4,
xi
hence the elements can easily grow very large in
magnitude. The scaling factor ensures that the variance of
the dot product remains bounded at σ2.

Because we are summing over a potentially variable
number of tokens n,
it is also helpful to include a
normalization operation, such as a softmax:1

=

hi

n
(cid:88)

j=1

softmax j

(g(xi, x j

))x j

(E.10.4)

In this context, we refer to g(•, •) as the attention
scoring function, and to the output of the softmax as the
attention scores. Because of the normalization properties
of the softmax, we can imagine that each token i has a
certain amount of “attention” it can allocate across the
other tokens: by increasing the budget on a token, the
attention over the other tokens will necessarily decrease
due to the denominator in the softmax.

1The notation softmax j in (E.10.4) means we are applying the
j=1, independently for each

softmax normalization to the set (cid:8)g(xi, x j
i. This is easier to see in the vectorized case, described below.

)(cid:9)n

244


Chapter 10: Transformer models

245

If we use a “dot-product attention”, our g does not have
trainable parameters. The idea of an attention layer is to
recover them by adding trainable projections to the input
before computing the previous equation. To this end, we
∼ (e, v),
define three trainable matrices Wk
∼ (e, k), where k and v are hyper-parameters. Each
Wq
token is projected using these three matrices, obtaining 3n
tokens in total:

∼ (e, k), Wv

Key tokens: ki
Value tokens: vi
Query tokens: qi

= W⊤
= W⊤
= W⊤

k xi
v xi
q xi

(E.10.5)

(E.10.6)

(E.10.7)

These processed tokens are called the keys, the values, and
the queries (you can ignore the choice of terminology for
now; we will return on this point at the end of the section).
The self-attention (SA) layer is obtained by combining the
three projections (E.10.5)-(E.10.6)-(E.10.7) with (E.10.4):

=

hi

n
(cid:88)

j=1

softmax j

(g(qi, k j

))vj

Hence, we compute the updated representation of token i
by comparing its query to all possible keys, and we use the
normalized weights to combine the corresponding value
tokens. Note that the dimensionality of keys and queries
must be identical, while the dimensionality of the values
can be different.

If we use the dot product, we can rewrite the operation
of the SA layer compactly for all tokens. To this end, we

245


246

Long convolutions and non-local models

define three matrices with the stack of all possible keys,
queries, and values:

K = XWk
V = XWv
Q = XWq

(E.10.8)

(E.10.9)

(E.10.10)

The three derived matrices K, V, Q have shapes (n, k),
(n, v), and (n, k) respectively. As a side note, we can also
implement them as a single matrix multiplication whose
output is chunked in three parts:

[K ∥ V ∥ Q] = X (cid:2)Wk

∥ Wv

∥ Wq

(cid:3)

where ∥ denotes concatenation. The SA layer is then
written as:

SA(X) = softmax

(cid:18) QK⊤
(cid:112)
k

(cid:19)

V

where we assume the softmax is applied row-wise. We can
also make the projections explicit, as follows.

Definition D.10.1 (Self-attention layer)

The self-attention (SA) layer is defined for an input
X ∼ (n, e) as:

SA(X) = softmax

(cid:19)

k X⊤
(cid:18) XWqW⊤
k

(cid:112)

XWv

(E.10.11)

246


Chapter 10: Transformer models

247

Figure F.10.2: Visualization of the main operations of the SA
layer (excluding projections).

∼ (k, e)
∼ (v, e), where k and v are hyper-parameters.
there are 2ke + ve trainable parameters,

The trainable parameters are Wq
and Wv
Hence,
independent of n.

∼ (k, e), Wk

We show the operation of the layer visually in Figure F.10.2.

10.1.3 Multi-head attention

The previous layer is also called a single-head attention
operation. It allows to model pairwise dependencies across
tokens with high flexibility. However, in some cases we
may have multiple sets of dependencies to consider: taking
again the example of “the cat, which belongs to my mother,
is on the table”, the dependencies between “cat” and “table”
are different with respect to the dependencies between
“cat” and “mother”, and we may want the layer to be able
to model them separately.2

A multi-head layer achieves this by running multiple
attention operations in parallel, each with its own set of

2And everything depends on the cat, of course.

247

QueriesKeysValuesSoftmaxRow-normalized
248

Long convolutions and non-local models

trainable parameters, before aggregating the results with
some pooling operation. To this end, we define a new
hyper-parameter h, that we call the number of heads of
the layer. We instantiate h separate projections for the
tokens, for a total of 3hn tokens (3n for each “head”):

Ke
Ve
Qe

= XWk,e
= XWv,e
= XWq,e

(E.10.12)

(E.10.13)

(E.10.14)

Wk,e represents the key projection for the e-th head, and
The multi-head
similarly for the other quantities.
attention (MHA) layer performs h separate SA operations,
stacks the resulting output embeddings, and projects them
a final time to the desired dimensionality:

Individual SA layer

MHA(X) = (cid:148)

SA1

(X) ∥ . . . ∥ SAh

(X)(cid:151)

Wo

(E.10.15)

Output projection

where:

SAi

(X) = softmax

(cid:19)

(cid:18) QiK⊤
i(cid:112)
k

Vi

Each SA operation returns a matrix of shape (n, v). These
h matrices are concatenated across the second dimension
to obtain a matrix (n, hv), which is then projected with
∼ (hv, o), where o is an additional hyper-
a matrix Wo
parameter allowing flexibility in the choice of the output
dimensionality.

248


Chapter 10: Transformer models

249

Heads and circuits

We will see shortly that the MHA layer is always
combined with a residual connection (Section 9.4). In
this case we can write its output for the i-th token as:

Sum over heads

xi

← xi

+ (cid:88)
e

(cid:88)

j

α

(xi, x j

e

⊤
)W
e x j

(E.10.16)

Sum over tokens

e

(xi, x j

) is the attention score between tokens i
where α
and j in head e, and We combines the value projection of
the e-th head with the e-th block of the output projection
in (E.10.15). The token embeddings are sometimes
called the residual stream of the model.a Hence, the
heads can be understood as “reading” from the residual
stream (via the projection by We and the selection via
the attention scores), and linearly “writing” back on the
streams.

aThis has been popularized in the context of mechanistic
interpretability, which tries to retro-engineer the layers’
behaviour to find interpretable components called circuits:
https://transformer-circuits.pub. The linearity of the stream
is fundamental for the analisys.

An explanation of the terminology

In order to understand why the three tokens are called
queries, keys, and values, we consider the analogy of a SA
layer with a standard Python dictionary, which is shown in
Box C.10.1.

Formally, a dictionary is a set of pairs of the form (key,

249


250

Long convolutions and non-local models

d = dict()
d["Alice"] = 2
d["Alice"]
d["Alce"]

# Returns 2
# Returns an error

Box C.10.1: A dictionary in Python: a value is returned only if a
perfect key-query match is found. Otherwise, we get an error.

value), where the key acts as an univocal ID to retrieve
the corresponding value. For example, in the third and
fourth line of Box C.10.1 we query the dictionary with
two different strings (“Alice” and “Alce”): the dictionary
compares the query string to all keys which are stored
inside, returning the corresponding value if a perfect match
is found, an error otherwise.

Given a measure of similarity over pair of keys, we can
consider a variant of a standard dictionary which always
returns the value corresponding to the closest key found in
the dictionary. If the keys, queries, and values are vectors,
this dictionary variant is equivalent to our SA layer if we
replace the softmax operation with an argmax over the
tokens, as shown in Figure F.10.3.

This “hard” variant of attention is difficult to implement
because the gradients of the argmax operation are zero
almost everywhere (we will cover discrete sampling and
approximating the argmax operation with a discrete
relaxation in the next volume). Hence, we can interpret
the SA layer as a soft approximation in which each token
is updated with a weighted combination of all values
based on the corresponding key/query similarities.

250


Chapter 10: Transformer models

251

Figure F.10.3: SA with a “hard” attention is equivalent to a
vector-valued dictionary.

10.2 Positional embeddings

With the MHA layer in hand, we consider the design of
the complete transformer model, which requires another
component, positional embeddings.

10.2.1 Permutation equivariance

It is interesting to consider what happens to the output of
a MHA layer when the order of the tokens is re-arranged
(permuted). To formalize this, we introduce the concept of
permutation matrices.

Definition D.10.2 (Permutation matrix)

A permutation matrix of size n is a square binary matrix
P ∼ Binary(n, n) such that only a single 1 is present on
each row or column:

1⊤P = 1, P1 = 1

If we remove the requirement for the matrix to have
binary entries and we only constrain the entries to be
non-negative, we obtain the set of doubly stochastic
matrices (matrices whose rows and columns sum to one).

251

     1,      0Argmax
252

Positional embeddings

Figure F.10.4: The output of a MHA layer after permuting the
ordering of the tokens is trivially the permutation of the original
outputs.

The effect of applying a permutation matrix is to rearrange
the corresponding rows / columns of a matrix. For example,
consider the following permutation:

P =









1 0 0
0 0 1
0 1 0

Looking at the rows, we see that the second and third
elements are swapped by its application:



P



x1
x2
x3


 =









x1
x3
x2

Interestingly, the only effect of applying a permutation
matrix to the inputs of a MHA layer is to rearrange the
outputs of the layer in an equivalent way:

MHA(PX) = P · MHA(X)

252

MHAMHA
Chapter 10: Transformer models

253

This is immediate to prove. We focus on the single headed
variant as the multi-headed variant proceeds similarly. First,
the softmax renormalizes the elements over the columns
of a matrix, so it is trivially permutation equivariant across
both rows and columns:

softmax(PXP⊤) = P [softmax(X)] P⊤

From this we can immediately deduce the positional
equivariance of SA:

SA(PX) = softmax

(cid:18)

P

XWqW⊤
k X⊤
(cid:112)
k

(cid:19)

P⊤

PXWv

(E.10.17)

= P · softmax

(cid:19)

k X⊤
(cid:18) XWqW⊤
k

(cid:112)

XWv

= P · SA(X) (E.10.18)

where we make use of the fact that P⊤P = I for any
permutation matrix. This can also be seen by reasoning on
the SA layer for each token: the output is given by a sum
of elements, each weighted by a pairwise comparison.
Hence, for a given token the operation is permutation
invariant.
Instead, for the entire input matrix, the
operation is permutation equivariant.

Translational equivariance was a desirable property for a
convolutional
layer, but permutation equivariance is
undesirable (at least here), because it discards the valuable
ordering of the input sequence. As an example, the only
effect of processing a text whose tokens have been
reversed would be to reverse the output of the layer,
despite the fact that the resulting reversed input is
probably invalid. Formally, the SA and MHA layers are set

253


254

Positional embeddings

functions, not sequence functions.3

Instead of modifying the layer or adding layers that are
not permutation equivariant, the transformer operates by
introducing a new concept of positional embeddings,
which are auxiliary tokens that depend only on the
position of a token in a sequence (absolute positional
embeddings) or the offset of two tokens (relative
positional embeddings). We describe the two in turn.

10.2.2 Absolute positional embeddings

Each token in the input matrix X ∼ (n, e) represents the
content of the specific piece of text (e.g., a subword).
Suppose we fix the maximum length of any sequence to m
To overcome positional equivariance, we
tokens.
introduce an additional set of positional embeddings
S ∼ (m, e), where the vector Si uniquely encodes the
concept of “being in position i”. Hence, the sum of the
input matrix with the first rows of S:

X′ = X + S1:n

is such that [X′]
in position i”.
i represents “token Xi
Because it does not make sense to permute the positional
embeddings (as they only depend on the position), the
resulting layer is not permutation equivariant anymore:

MHA(PX + S) ̸= P · MHA(X + S)

See Figure F.10.5 for a visualization of this idea.

How should we build positional embeddings? The easiest
strategy is to consider S as part of the model’s parameters,

3To be even more pedantic, they are multiset functions since tokens

can be repeated.

254


Chapter 10: Transformer models

255

Figure F.10.5: Positional embeddings ( green ) added to the

tokens’ embeddings ( red ). The same token in different positions

has different outputs ( blue ).

and train it together with the rest of the trainable
parameters, similarly to the token embeddings. This
strategy works well when the number of tokens is
relatively stable; we will see an example in the next
chapter in the context of computer vision.

Alternatively, we can define some deterministic function
from the set of tokens’ positions to a given vector that
uniquely identifies the position. Some strategies are clearly
poor choices, for example:

1. We can associate to each position a scalar p = i/m
which is linearly increasing with the position.
However, adding a single scalar to the token
embeddings has a minor effect.

2. We can one-hot encode the position into a binary
vector of size m, but the resulting vector would be
extremely sparse and high-dimensional.

A possibility, introduced in the original transformer paper
[VSP+17],
To
understand them, consider a sine function:

is that of sinusoidal embeddings.

y = sin(x)

255

"Tut Tut Child"TokenizationTutTutChild+++123===Child/3Tut/2Tut/1
256

Positional embeddings

The sine assigns a unique value to any input x inside the
range [0, 2π]. We can also vary the frequency of the sine:

y = sin(ωx)

This oscillates more or less rapidly based on the frequency
ω, and it assigns a unique value to any input in the range
[0, 2π

ω ].

There is an analogy with an (analogical) clock:
the
seconds’ hand makes a full rotation with a frequency of 1
60
Hz (once every minute). Hence, every “point in time”
inside a minute can be distinguished by looking at the
hand, but two time instants in general can only be
identified modulo 60 seconds. We overcome this in a clock
by adding a separate hand (the minute hand) that rotates
1
with a much slower frequency of
3600 Hz. Hence, by
looking at the pair of coordinates (second, minute) (the
“embedding” of time) we can distinguish any point inside
an hour. Adding yet another hand with an even slower
frequency (the hour hand) we can distinguish any point
inside a day. This can be generalized: we could design
clocks with lower or higher frequencies to distinguish
months, years, or milliseconds.

A similar strategy can be applied here: we can distinguish
each position i by encoding it through a set of e sines (with
e an hyper-parameter) of increasing frequencies:

Si

= [sin(ω

1i), sin(ω

2i), . . . , sin(ω

ei)]

In practice, the original proposal from [VSP+17] uses only
e/2 possible frequencies, but adds both sines and cosines:

= (cid:2)sin(ω

Si

1i), cos(ω

1i), . . . , sin(ω

e/2i), cos(ω

e/2i)(cid:3)

256


Chapter 10: Transformer models

257

Figure F.10.6:
We show three
sin functions with
ω = 0.1, ω = 1,
and ω = 10. The
embedding for
position x = 6
is given by the
corresponding values
(red circles).

This can be justified by noting that in this embedding, two
positions are related via a simple linear transformation, a
rotation, that depends only on the relative offset of the two
positions.4 Any choice of frequency is valid provided they
are sufficiently large and increasing at a super-linear rate.
The choice from [VSP+17] was a geometric progression:

ω

i

=

1
10000i/e

that varies from ω
for a visualization.

0

= 1 to ω

e

= 1

10000 . See Figure F.10.6

10.2.3 Relative positional embeddings

Trainable positional embeddings and sinuisodal positional
embeddings are examples of absolute embeddings,
because they encode a specific position in the sequence.
An alternative that has become common with very long
sequences are relative positional embeddings. In this
case, instead of adding a positional encoding to a token,
we modify the attention function to make it dependent on

4See

https://kazemnejad.com/blog/transformer_architecture_

positional_encoding/ for a worked-out computation.

257

−101sin(0.1x)−101sin(1x)0246810−101sin(10x)
258

Building the transformer model

the offset between any two tokens:

g(xi, x j

) → g(xi, x j, i − j)

This is a combination of the two ideas we introduced at
the beginning of this chapter (Figure F.10.1). Note that
while absolute embeddings are added only once (at the
input), relative embeddings must be added every time an
MHA layer is used. As an example, we can add a trainable
bias matrix B ∼ (m, m) and rewrite the dot product with
an offset-dependent bias:

g(xi, x j

) = x⊤

i x j

+ Bi j

A simpler variant, attention with linear biases (ALiBi)
[PSL22], considers a single trainable scalar in each head
which is multiplied by a matrix of offsets. More advanced
strategies, such as rotary positional embeddings (RoPE),
are also possible [SAL+24].

10.3 Building the transformer model

10.3.1 The transformer block and model

A model could be built, in principle, from a stack of
multiple MHA layers (with the softmax providing the
non-linearity necessary to avoid the collapse of multiple
linear projections). Empirically, however, it is found that
the MHA works best when interleaved with a separate
fully-connected block that operates on each token
independently. These two operations can be understood as
mixing the tokens (MHA), and mixing the channels (MLP),
similarly to the depthwise-separable convolution model.

In particular, for the MLP block it is common to choose a

258


Chapter 10: Transformer models

259

Figure F.10.7: Schematic
view of pre-normalized and
post-normalized transformer
blocks. In the post-normalized
variant the LN block is
applied after the MHA or MLP
operation, while in the pre-
normalized one before each
layer.

bottleneck architecture composed of two fully-connected
layers of the form:

MLP(x) = W2

φ (W1x)

∼ (p, e), with p selected
where x ∼ (e) is a token, W1
as an integer multiple of e (e.g., p = 3e or p = 4e), and
∼ (e, p) reprojecting back to the original embedding
W2
dimension. Biases are generally removed as the increased
hidden dimension provides sufficient degrees of freedom.

To ensure efficient training of deep models we also need
a few additional regularization strategies. In particular, it
is common to include two layer normalization steps and
two residual connections, respectively for the MHA and
MLP blocks. Depending on where the layer normalization
is applied, we obtain two variants of the basic transformer
block, sometimes denoted as pre-normalized and post-
normalized. These are shown in Figure F.10.7.

While the post-normalized version corresponds to the
original transformer block, the pre-normalized variant is
generally found to be more stable and faster to train
[XYH+20]. The design of the block in Figure F.10.7 is,
fundamentally, an empirical choice, and many variants
have been proposed and tested in the literature. We

259

MHALNInputsMLPLN(a) Post-normalized blockMHALNInputsMLPLN(b) Pre-normalized block
260

Building the transformer model

review some of these later on in Section 11.3.

We can now complete the description of a basic transformer
model:

1. Tokenize and embed the original input sequence in a

matrix X ∼ (n, e).

2. If using absolute positional embeddings, add them

to the input matrix.

3. Apply 1 or more blocks of the form discussed above.

4. Include a final head depending on the task.

The output of step (3) is a set of processed tokens
H ∼ (n, e), where neither n nor e are changed by the
transformer model (the former because we do not have
local pooling operations on sets, the latter because of the
residual connections in the block).
Considering for
example a classification task, we can apply a standard
classification head by pooling over the tokens and
proceeding with a fully-connected block:

(cid:130)

(cid:130)

y = softmax

MLP

(cid:140)(cid:140)

1
n

(cid:88)

Hi

i

This part is identical to its corresponding CNN design.
However, the transformer has a number of interesting
properties, mostly stemming by the fact
it
manipulates its input as a (multi)set, without modifying
its dimensionality throughout the architecture. We
investigate one simple example next.

that

260


Chapter 10: Transformer models

261

10.3.2 Class tokens and register tokens

While up to now we have assumed that each token
corresponds to one part of our input sequence, nothing
prevents us from adding additional tokens to the input of
the transformer. This is strictly dependent on its specific
architecture: a CNN, for example, requires its input to be
precisely ordered, and it is not clear how we could add
additional tokens to an image or to a sequence. This is a
very powerful idea, and we only consider two specific
implementations here.

First, we consider the use of a class token [DBK+21], an
additional token which is added explicitly for classification
in order to replace the global pooling operation above.
Suppose we initialize a single trainable token c ∼ (e), which
is added to the input matrix:

X ←

(cid:152)

(cid:149) X
c⊤

The new matrix has shape (n + 1, e). The class token is
identical for all sequences in a mini-batch. After step (3)
above, the transformer outputs a matrix H ∼ (n + 1, e) of
updated representations for all tokens, including the class
one. The idea is that, instead of pooling over the tokens,
the model should be able to “compress” all information
related to the classification task inside the class token, and
we can rewrite the classification head by simply discarding
all other tokens:5

y = softmax (MLP (Hn+1

))

5In the language of circuits and heads from Section 10.1.3, we could
say equivalently that the model must learn to move all information
related to the task in the residual stream of the class token.

261


262

Building the transformer model

Additional trainable tokens can be useful even if not
explicitly used. For example, [DOMB24] has shown that
adding a few additional tokens (called registers in this
case) can improve the quality of the attention maps by
providing the model with the possibility of using the
registers to “store” auxiliary information that does not
depend explicitly on a given position.

From theory to practice

We will
introduce many important
concepts related to transformers in the
next chapter. Thus, for this chapter I
am suggesting a slightly unconventional
exercise which combines a convolutional backbone with a
transformer-like head – as depicted in Figure F.10.8.

The convolutional models you developed in Chapters 7
and 9 were applied to a single image. However, sometimes
we have available a set of images of the same object to
be recognized – for example, in a monitoring system, we
may have multiple screenshots of a suspicious person. This
is called a multi-view system in the literature, and each
image is called a view of the object. A multi-view model
should provide a single prediction for the entire set of views,
while being invariant to the order of the views in input. For
this exercise we will implement a simple multi-view model
– see Figure F.10.8.

1. Using any image classification dataset, you can
simulate a multi-view model by applying a fixed
number of data transformations to the input (gray
Ignoring the batch
block in Figure F.10.8).
dimension,
shape
x ∼ (h, w, c) (height, width, channels), you obtain a

for each input

image of

262


Chapter 10: Transformer models

263

Figure F.10.8: Multi-view model to be implemented in this
chapter. The image is augmented through a set of random data
augmentation strategies to obtain a set of views of the input
( gray ). Each view is processed by the same convolutional

backbone to obtain a fixed-sized dimensional embedding ( red ).
The set of embeddings are processed by a transformer block before
the final classification ( green ). Illustration by John Tenniel.

multi-view input of shape x ′ ∼ (v, h, w, c), where v is
the number of views. A single label y is associated
to this tensor – the label of the original image. The
number of views can also be different
from
mini-batch to mini-batch, as no part of the model is
constrained to a pre-specified number of view.

2. The multi-view model

is composed of

three
components. Denote by g(x) a model that processes
a single view to a fixed-dimensional embedding – for
example, this can be any convolutional backbone
you trained for the previous exercises. The first part
of the full model (red part in Figure F.10.8) applies g
) ∼ (e), where e is a
in parallel to all views, hi

= g(x i

263

DataaugmentationsConvolutionalbackboneConvolutionalbackboneConvolutionalbackboneTransformer blockPooling (e.g., average)Classificationhead"Lizard"ViewsMulti-view model
264

Building the transformer model

hyper-parameter (the output size of the backbone).

3. After concatenating the embeddings of the views we
obtain a matrix H ∼ (v, e).
In order for the full
model to be permutation invariant, any component
applied on H must be permutation equivariant.6 For
the purposes of this exercise, implement and apply a
single transformer block as per Section 10.3.1. You
can implement MHA using basic PyTorch, or you can
try a more advanced implementation using
einops.7
You can also compare with the
pre-implemented version in torch.nn.

4. The transformer block does not modify the input
shape. To complete the model, perform an average
over the views (which represent the tokens in this
scenario), and apply a final classification head. You
can also experiment with adding a class token
(Section 10.3.2).
It is easy to show that a model
built in this way is permutation invariant with
respect to the views.

6An average operation over the views is the simplest example of
permutation invariant layer. Hence, removing the MHA block from
Figure F.10.8 is also a valid baseline. Alternatively, deep sets [ZKR+17]
characterize the full spectrum of linear, permutation invariant layers.

7See https://einops.rocks/pytorch-examples.html.

264
