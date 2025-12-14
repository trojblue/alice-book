13 | Recurrent models

About this chapter

Transformer models are very effective at processing
sequences, but they are hindered by their quadratic
complexity in the sequence length. One possibility
is to replace them with recurrent layers, having only
constant-time for processing each element of a sequence,
irrespective of its length.
In this final chapter we
provide an overview of several recurrent models and
their characteristics. The field has been moving very
rapidly in the last two years, and we provide a wide
overview at the expense of precision – see [TCB+24] for
a recent survey.

13.1 Linearized attention models

13.1.1 Replacing the dot product

To provide some intuition on why recurrent neural
networks (RNNs) can be useful, we begin with a
generalization of the attention layer (called the linearized
attention layer [KVPF20]) that can be written in a
recurrent form. We start by rewriting the SA layer in an

315


316

Linearized attention models

abstract form with a generic scalar-valued attention
function α(•, •) instead of the dot product:

=

hi

(cid:80)n

j=1
(cid:80)n

α (cid:0)qi, k j
α (cid:0)qi, k j

(cid:1) v j
(cid:1)

j=1

(E.13.1)

where for the standard SA, α(x, y) = exp(x⊤y).
If the
elements of the sequence must be processed in order (as
in autoregressive generation), (E.13.1) is inconvenient
because its cost grows quadratically in the sequence
length. Even if a KV cache is used, memory still grows
linearly. By comparison, a convolutional layer has fixed
time and memory cost for each element to be processed,
but information is lost if a token is outside the receptive
field. What we would like, then, is a mechanism to
compress all the information of the sequence into a
fixed-size input (which we will call a memory or state
tensor), so that the cost of running the model on our
current input token plus the memory is constant. We call
models of this form recurrent.

In machine learning,

To begin, note that any non-negative α is a valid similarity
this requirement is
function.
equivalent to α being what is called a kernel function
[HSS08]. Many such kernel functions can be written as a
generalized dot product:

α(x, y) = φ(x)⊤φ(y)

(E.13.2)

for some function φ : (cid:82)c → (cid:82)e performing a feature
expansion (this is the cornerstone of methods such as
support vector machines, but discussing it at length would
go beyond the scope of the book - also, it will not be
required beyond this section).

316


Chapter 13: Recurrent models

317

Kernel functions

As an example of kernel function, the polynomial kernel
function α(x, y) = (1 + x⊤y)d can be rewritten as
(E.13.2) if φ(•) explicitly computes all polynomials of
its input up to order d [HSS08]. Some kernel functions
correspond to infinite-dimensional expansions (e.g., the
Gaussian kernel), in which case (E.13.2) can still be
recovered in terms of an approximated kernel expansion,
such as working with random Fourier features [SW17].

Based on (E.13.2) we can rewrite (E.13.1) as:

=

hi

(cid:80)n

j=1
(cid:80)n

φ(qi
φ(qi

)⊤φ(k j
)⊤φ(k j

)v⊤
j
)

j=1

where we have added a transpose operation on v j to be
) does not
consistent with the dimensions. Because φ(qi
depend on j we can bring it outside the sum to obtain:

=

hi

φ(qi
φ(qi

)⊤ (cid:80)n

j=1
)⊤ (cid:80)n

φ(k j
φ(k j

)v⊤
j
)

j=1

(E.13.3)

This is called a linearized attention model [KVPF20].
Computing (E.13.3) for all
tokens has complexity
(cid:79) (n(e2 + ev)), which is linear in the sequence length and
advantageous whenever n < e2. φ can be chosen freely,
in [KVPF20] they consider a quadratic feature
e.g.,
expansion or even a simpler φ(x) = ELU(x) + 1 for short
sequences.

317


318

Linearized attention models

13.1.2 A recurrent formulation

We now rewrite the linearized attention model
in a
recurrent form, by considering what happens for a causal
variant of the layer.
First, we modify (E.13.3) by
constraining the sum only on past input elements to make
it causal:

Attention memory Si

=

hi

φ(qi

)⊤ (cid:80)i

j=1

φ(k j

)v⊤
j

φ(qi

)⊤ (cid:80)i

j=1

φ(k j

)

(E.13.4)

Normalizer memory zi

This is our first example of a recurrent layer.
To
understand the name, we note that the attention and
normalizer memories can be written recursively as:

= Si−1
Si
= zi−1
zi

+ φ(ki
+ φ(ki

)v⊤
i
)

(E.13.5)

(E.13.6)

where the base case of the recurrence is given by their
initialization:

S0
z0

= 0
= 0

The output is then given by:

=

hi

φ(qi
φ(qi

)⊤Si
)⊤zi

(E.13.7)

(E.13.8)

(E.13.9)

Equations (E.13.5)-(E.13.9) are particularly interesting
for an autoregressive scenario: for any new token to be
generated, we update the two memory states (equations

318


Chapter 13: Recurrent models

319

Figure F.13.1:
Overview of a recurrent
layer: past tokens are
shown in gray, current
input token in blue, the
memory state in yellow.

(E.13.5) and (E.13.6)), and we use these updated states
to compute the output for the i-th element. Importantly,
the total computation for generating a new token is
constant, and the cost in memory is also fixed since the
previous memories Si−1 and zi−1 can be discarded. We can
alternate between the two formulations of the layer: we
can use a vectorized variant for training (for efficient
implementation on GPUs) and the recurrent formulation
for inference.

13.2 Classical recurrent layers

13.2.1 General formulation

Let us now abstract away the key components of a recurrent
layer, using the previous section as reference. First, we need
a state of fixed size, which is used to compress all useful
information up to the i-th element of the sequence. We
denote it generically as si, and without lack of generality
we assume it is a single vector from now on. Second, we
need a transition function (recurrence) that updates the
state vector based on the previous value and the value of
the current token, which we denote as f (si−1, xi
). Third,
we need what we call a readout function that provides an
output for the i-th element of the sequence. We denote it
as g(si, xi

). See also Figure F.13.1 for a visualization.

319

RecurrenceRecurrenceReadoutPrevious tokenCurrent tokenMemory (state)
320

Classical recurrent layers

Definition D.13.1 (Recurrent layer)

Given a sequence of tokens x1, x2, . . ., a generic recurrent
layer can be written as:

= f (si−1, xi
si
)
= g(si, xi
hi

)

(E.13.10)

(E.13.11)

∼ (e) is initialized as zero by
where the state vector si
= 0. The size of the state vector, e, and the
convention, s0
∼ (o) are hyper-parameters.
size of the output vector hi
We call f the state transition function and g the readout
function.

In this format, a recurrent layer represents a discrete-time,
input-driven dynamical system, and it is a causal layer by
definition. In control engineering, this is also known as a
state-space model.
For tasks in which causality is
unnecessary, bidirectional layers [SP97] can also be
defined. In a bidirectional layer we initialize two recurrent
layers (with separate parameters), one of which processes
the sequence left-to-right, and the second one right-to-left.
Their output states are then concatenated to provide the
final output.

Recurrent neural networks (RNNs) can be built by
layers on the updated
stacking multiple recurrent
sequence h1, h2, . . . , hn
Interestingly, a
recurrent layer has no requirement on the length of the
sequence, which can (in principle) be unbounded. For this
reason, RNNs with unbounded precision or growing
architectures can be shown to be Turing-complete [CS21].

[PGCB14].

320


Chapter 13: Recurrent models

321

Implicit layers

What happens if we apply a recurrent layers to a single
token x?

si

= f (si−1, x)

(E.13.12)

If we run the state transition several time starting from
a known initialization s0, this is similar to a model
with several layers (one per transition) sharing the
same parameters. Suppose we run (E.13.12) an infinite
number of times. If the dynamic system has a stable
attractor, the output will be defined by the fixed-point
equation:

s = f (s, x)

(E.13.13)

If we take (E.13.13) as the definition of a layer, we
obtain what is called an implicit layer [BKK19]. The
implementation of implicit layers can be made feasible
by using fast solvers for the fixed-point equation and
computing the backward pass with the use of the implicit
function theorem [BKK19]. Implicit graph layers can
also be defined by running each diffusion operation to a
stable state [GMS05, SGT+08].

13.2.2 “Vanilla” recurrent layers

recurrent

layers were instantiated by
Historically,
considering two fully-connected layers as transition and
readout functions:

f (si−1, xi

) = φ(Asi−1
) = Csi
g(si, xi

+ Bxi

)
+ Dxi

(E.13.14)

(E.13.15)

where as always we ignore biases for simplicity, and we
have four trainable matrices A ∼ (e, e), B ∼ (e, c), C ∼
(o, e), and D ∼ (o, c), where c is the input dimensionality

321


322

Classical recurrent layers

(the size of each token). A layer in this form is sometimes
referred to generically as a “recurrent layer”, a “vanilla
recurrent layer”, or an Elman recurrent layer. When the
two matrices A and B are left untrained and we only have a
single layer, these models are called echo state networks
(ESNs) or reservoir computers [LJ09]. ESNs can be a
powerful baseline for time series forecasting, especially
when the untrained matrices (the reservoir) are initialized
in a proper way [GBGB21].

Despite their historical significance, layers of this form are
extremely inefficient (and hard) to train. To see this, note
that by its design the computation across elements of the
sequence cannot be parallelized efficiently, as shown in
Box C.13.1. Hence, we need to resort to iterative
(for-loops) implementations, and even highly customized
CUDA implementations1 are slower than most alternative
sequence layers.

Another issue stems from the gradients involved in the
layer’s computations. Consider a simplified case having
only the transition function. We can unroll the full
computation as:

s1
s2

)

)

= f (s0, x1
= f (s1, x2
...
= f (sn−1, xn

sn

)

1https://docs.nvidia.com/deeplearning/performance/dl-

performance-recurrent/index.html

322


Chapter 13: Recurrent models

323

# Input tensor
x = torch.randn(batch_size,

sequence_length,
features)

# State tensor
s = torch.zeros(batch_size,
state_size)

# State update
state_update = nn.RNNCell(features,

state_size)

for i in range(x.shape[1]):

s = state_update(x[:, i, :], s)

Box C.13.1: Vanilla recurrence in PyTorch. It is impossible to
parallelize the for-loop because of the dependencies in the recurrence.
In PyTorch, the state update is called a recurrent cell, while the
recurrent layers, such as torch.nn.RNN, wrap a cell and perform
the complete for-loop.

323


324

Classical recurrent layers

This is similar to a model with n layers, except that the
parameters are shared (the same) across the layers. Below
we focus on the quantity ∂
Asn (the weight Jacobian with
respect to A), but similar considerations apply to all
gradients. Let us define the following cumulative product:

=

(cid:101)si

n
(cid:89)

j=i+1

∂

s j−1

f (s j−1, x j

)

(E.13.16)

This represents the gradient of the transition function from
the end of the sequence backwards to element i, as shown
in Figure F.13.2. Because of weight sharing, the gradient
we are looking for has a separate term for each element in
the sequence which involves these cumulative products:

∂
Asn

= ∂

A f (sn−1, xn

) +

(cid:2)∂

(cid:101)si

A f (si−1, xi

)(cid:3)

(E.13.17)

Gradient from element n
n−1
(cid:88)

i=1

Gradient from element i

The first term corresponds to a “standard” weight Jacobian,
describing the influence of A on the last element of the
sequence. The terms in the summation are the additional
contributions, one for each element of the sequence, which
are weighted by the chained input Jacobian computed over
the sequence itself.

form,

reverse mode

automatic
Written in this
differentiation is also called backpropagation through
time (BPTT), and it can be a strong source of instability or
gradient problems during gradient descent. To see this,
note that each input Jacobian in the inner product in
(E.13.17) involves a multiplication by the derivative of the

324


Chapter 13: Recurrent models

325

Figure F.13.2: Backward pass for a recurrent layer: the adjoint
values have to be propagated through all the transition steps. Each
state then contributes a single term to the full gradient of the
parameters.

activation function φ. Some of the earliest analyses of
vanishing and exploding gradients were done in this
context [Hoc98]. For long sequences, stability of the layer
is guaranteed only when the eigenvalues of the transition
matrix are properly constrained [GM17].
Layer
normalization was also originally developed to stabilize
training in RNNs, by computing statistics over the states’
sequence [BKH16].

Several techniques have been developed to partially solve
these instabilities in the context of recurrent layers. For
example, the sum in (E.13.17) can be truncated to a given
interval (truncated BPTT), or the gradients can be
thresholded if they exceed a pre-defined upper bound
(clipped gradients).

13.2.3 Gated recurrent networks

Over the years, several variants of the vanilla layer were
proposed to improve its performance. In this section we
focus on a popular class of such models, called gated RNNs.

325

RecurrenceRecurrenceRecurrenceReadout
326

Classical recurrent layers

One issue of RNNs is that the entire state gets overwritten
at each transition, which is reflected in the partial products
in (E.13.17). However, we can assume that, for many
sequences, only a few elements of these transitions are
important: as an example, in an audio signal, empty regions
or regions with no information are typical. In these cases,
we may be interested in sparsifying the transition (similarly
to how most attention weights tend to be close to zero) and,
consequently, setting most elements in (cid:101)si to 1. This can be
achieved with the addition of specialized gating layers.

We consider the simplest form of gated RNN, called light
gated recurrent unit (Li-GRU, [RBOB18]), having a single
gate. For our purposes, a gating function is simply a layer
that outputs values in the range [0, 1] that can be used
to “mask” the input. As an example, a gate over the state
can be obtained by a fully-connected layer with a sigmoid
activation function:

γ(si−1, xi

) = σ (Vsi−1

+ Uxi

)

where V and U have similar shapes to A and B. We can
≈ 0, the i-th feature of the
interpret this as follows: if γ
state should be kept untouched, while if γ
≈ 1, we should
1
propagate its updated value as output. Hence, we can
rewrite the transition function by properly masking the
new and old values as:

i

f (si−1, xi

) =

New values
(cid:125)(cid:124)
) ⊙ φ (Asi−1

(cid:122)
γ(si−1, xi
+ (1 − γ(si−1, xi
(cid:123)(cid:122)
Old values

(cid:124)

)) ⊙ si−1
(cid:125)

(cid:123)
)
+ Bxi

326


Chapter 13: Recurrent models

327

This can be seen as a soft (differentiable) approximation
to a “real” gate having only binary values, or as a convex
combination of the original layer and a skip connection.
We can theoretically control
this
approximation by adding an additional regularizer to the
loss that constrains the outputs of the gate to lie as close
as possible to 0 or 1.

the goodness of

Other gated recurrent layers can be obtained by adding
the original gated
additional gates to this design:
recurrent unit (GRU) adds a so-called “reset gate” to the
layer [CVMG+14], while long-short term memory units
(LSTMs) have a third “forget gate” [HS97]. LSTMs were
the first gated variant to be introduced in the literature,
and for a long time they have been the most successful
deep architecture for processing sequences [Sch15].
Because of this, research on LSTM models is still very
active [BPS+24].

13.3 Structured state space models

13.3.1 Linear recurrent layers

We now consider a simplified class of recurrent layers, in
which we remove the intermediate nonlinearity in the
transition function:

f (si−1, xi

) = Asi−1
) = Csi

+ Bxi
+ Dxi

(E.13.18)

(E.13.19)

g(si, xi

Written in this form, (E.13.18)-(E.13.19) are called state
space models (SSM).2 Intuitively, an SSM layer is “less

2Confusingly, any recurrent layer in the form (E.13.10)-(E.13.11) is
an SSM, but in the neural network’s literature the term SSM has come

327


328

Structured state space models

expressive” than a standard recurrent layer (because of
the lack of nonlinearities). However, this can be recovered
by adding activation functions after the output, or by
interleaving these layers with token-wise MLPs [ODG+23].

Interest in this class of models (re)-started in 2020, when
[GDE+20] analyzed a theoretical construction for the
matrix A in (E.13.18) that could efficiently compress
one-dimensional input sequences. The result was called
the HiPPO (High-Order
Projection
Operator) matrix. A family of neural networks built by a
stack of SSM layers based on the HiPPO theory soon
leading to the Structured State Space for
followed,
Sequence Modeling (S4) layer in 2021 [GGR22] and the
simplified S4 model (S5) in 2022 [SWL23].

Polynomial

Because of their roots in HiPPO theory, the proposed SSM
layers up to S4 considered a stack of 1D models, one for
each channel of the input, with transition matrices
initialized as HiPPO matrices. By contrast, S5 introduced a
standard multi-input, multi-output model of the form in
(E.13.18)-(E.13.19), which is the one we describe here. In
particular, we focus our analysis on a simplified variant
known as the linear recurrent unit (LRU) [OSG+23].

This formulation has a number of interesting properties,
mostly stemming from the associativity of the linear
transition function. To see this, we start by noting that the
recurrence has a closed form solution:

=

si

i
(cid:88)

j=1

Ai− jBx j

(E.13.20)

to be associated only with the linear variant. Sometimes we refer to
them as structured SSMs because, as we will see, we need to properly
constrain the transition matrix to make them effective.

328


Chapter 13: Recurrent models

329

We can view this summation from two different points of
view. First, we can aggregate all coefficients with respect
to the input sequence into a rank-3 tensor:

K = stack (cid:0)An−1B, An−2B, . . . , AB, B(cid:1)

We can compute all outputs via a single 1D convolution
of filter size equal to the length of the sequence (a long
convolution) between the input sequence stacked into a
single matrix X ∼ (n, c) and the pre-computed kernel K:

S = Conv1D(X, K)

Hence, the SSM layer can be interpreted as a convolution
[GJG+21]. If the transition matrix is applied on a single
channel, this can be exploited to speed-up computations by
operating in the frequency domain, e.g., in the FlashConv
implementation.3 However, a more efficient solution can
be found by exploiting a family of algorithms known as
associative (parallel) scans (or all-prefix-sums).

13.3.2 An interlude: associative scans

We introduce parallel scans in their general formulation
before seeing their application to linear SSMs. Consider
a sequence of elements (x1, x2, . . . , x n
), and an operation
⋆ which is assumed binary (it acts on any two elements
of the sequence) and associative. We want to compute all
partial applications of this operator to the sequence (using
separate colors for readability):

3https://www.together.ai/blog/h3

329


330

Structured state space models

Figure F.13.3: Parallel scan
on a sequence of six elements:
circles of the same color can be
computed in parallel; dashed
circles are the outputs of the
parallel scan.

x1, x1

⋆ x2, x1

⋆ x2

⋆ x3,

. . . , x1

⋆ x2

⋆ · · · ⋆ x n

This can be done trivially by an iterative algorithm which
computes the elements one-by-one, adding one element at
every iteration (this corresponds to how a standard
recurrent layer would be computed). However, we can
devise an efficient parallel algorithm by exploiting the
associativity of the operator ⋆ [Ble90]. The key intuition is
that multiple pairs of elements can be computed in
parallel and then aggregated recursively.

As a simple example, consider a sequence of 6 elements
x1, x2, x3, x4, x5, x6 (an in-depth example applied to SSMs
can be found in [SWL23]). We will denote by ˆx i the i-th
prefix we want to compute. The overall procedure is shown
schematically in Figure F.13.3. We first aggregate pairs of
adjacent values as:

→ ˆx2

s1
s2
s3

= x1
= x3
= x5

⋆ x2
⋆ x4
⋆ x6

330


Chapter 13: Recurrent models

331

where we use arrows to denote outputs of the algorithm.
We now perform a second level of aggregations:

And finally:

s1
= s1

⋆ x3
⋆ s2

→ ˆx3
→ ˆx4

o1

o1
o1

⋆ x5
⋆ s3

→ ˆx5
→ ˆx6

While this looks strange (we made 7 steps instead of 5), the
three blocks of computations can be trivially parallelized
if we have access to 3 separate threads. In general, by
organizing the set of computations in a balanced fashion,
we are able to compute the parallel scan in (cid:79) (T log n),
where T is the cost of the binary operator ⋆. An example
of implementation is the associative scan function in JAX.4

It is easy to show that the transition function in a linear SSM
is an example of an all-prefix-sums problem. We define the
), and the
elements of our sequence as pairs x i
binary operator as:

= (A, Bxi

(Z, z) ⋆ (V, v) = (VZ, Vz + v)

The prefixes of ⋆ are then given by [SWL23]:

x1

⋆ x2

⋆ . . . ⋆ x i

= (Ai, si

)

Hence, running a parallel scan gives us the powers of A as
the first elements of the output, and all the states of the
layer as the second element of the output. The complexity

4https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.

associative_scan.html

331


332

Structured state space models

of this operation is upper bounded by the complexity of
Ai−1A, which scales as (cid:79) (n3).
To make the entire
procedure viable, we can constrain A so that its powers
can be computed more efficiently. This is the topic of the
next section.

13.3.3 Diagonal SSMs

A common strategy to make the previous ideas feasible
is to work with diagonal transition matrices (or diagonal
matrices plus a low-rank term [GGR22]).
In this case,
powers of A can be computed easily by taking powers of
the diagonal entries in linear time. In addition, as we will
see, working with diagonal matrices allows us to control
the dynamics of the transition function to avoid numerical
instabilities.

In particular, a square matrix A is said to be diagonalizable
if we can find another square (invertible) matrix P and a
diagonal matrix Λ such that:

A = PΛP−1

(E.13.21)

Diagonalizable matrices are (in a sense) “simpler” that
generic matrices, For example, if such a decomposition
exists, it is easy to show that powers can also be computed
efficiently as:

Ai = PΛiP−1

Suppose that the transition matrix is diagonalizable. Then,
we can re-write the SSM in an equivalent form having
a diagonal transition matrix. We begin by substituting

332


Chapter 13: Recurrent models

333

(E.13.21) into the definition of the SSM and multiplying
on both sides by P−1:

P−1si

=

i
(cid:88)

j=1

Λi− j PB x j

New state vector ¯si

New input-state matrix ¯B

We now rewrite the readout function in terms of the new
variable ¯s:

New readout matrix ¯C

yi

= CP ¯si

+ Dxi

Putting everything together:

= Λ¯si−1
¯si
= ¯C¯si
yi

+ ¯Bxi
+ Dxi

(E.13.22)

(E.13.23)

Hence, whenever a diagonalization of A exists, we can
always rewrite the SSM into an equivalent form having a
diagonal transition matrix. In this case, we can directly
train the four matrices Λ = diag(λ), λ ∼ (e), ¯B ∼ (e, c),
¯C ∼ (o, e) and D ∼ (o, c), with the diagonal matrix being
parameterized by a single vector of dimension e.

Not all matrices can be diagonalized. However, an
approximate diagonalization can always be found if one
allows for matrices P and Λ to have complex-valued
entries [OSG+23]. Care must be taken to parameterize the
values over the diagonal so that the eigenvalues of the
transition matrix stay < 1 in absolute value, to avoid
diverging dynamics. We refer to [OSG+23] for a

333


334

Additional variants

description of both points and for a complete analysis of
the resulting LRU layer.

13.4 Additional variants

strengths of

Balancing the different
convolutions,
recurrence, and attention is an active research topic. To
close the book, we list some recurrent layers (or layers
that can be interpreted as recurrent) that have been
introduced very recently in the literature.

13.4.1 Attention-free transformers

One issue of the linearized transformer model (Section
13.1.1) is the quadratic complexity in the feature dimension
e. The attention-free transformer (ATF) was introduced as
a variant of the basic attention layer that is instead linear
in both sequence length and in the number of features
[ZTS+21].

The core idea is to replace the dot product interactions
between keys, query, and values with a simpler
multiplicative interaction (element-wise):

hi

= σ(qi

) ⊙

(cid:80)

j exp (cid:0)k j
(cid:80)

j exp (cid:0)k j

(cid:1) ⊙ v j
(cid:1)

(E.13.24)

This is similar to the self-attention layer, except that we
replace all dot products with element-wise (Hadamard)
It is also inspired by the linearized
multiplications.
attention layer in that the query is only used as a global
modulation factor, in this case after normalizing it with a
In fact, we can recover a standard
sigmoid operation.
attention formulation by rewriting (E.13.24) for a single
dimension z (exploiting the fact that we only perform

334


Chapter 13: Recurrent models

335

element-wise operations):

=

hiz

σ(qiz
(cid:80)

) (cid:80)

j exp(k jz
)

)

vjz

j exp(k jz

the ATF layer can be re-interpreted as a
Hence,
channel-wise variant of attention, in the sense that for
every channel we can rewrite it as an attention operation
over the elements of the sequence. To increase flexibility,
[ZTS+21] also considered adding relative embeddings
W ∼ (m, m) (where m is the maximum allowed length of
the sequences):

hi

= σ(qi

) ⊙

(cid:80)

+ Wi j

j exp (cid:0)k j
(cid:80)

j exp (cid:0)k j

+ Wi j

(cid:1) ⊙ v j
(cid:1)

(E.13.25)

The relative embeddings can also be trained via a low-
rank factorization to reduce the number of parameters.
See [ZTS+21] for this and for additional variants of the
basic ATF layer (e.g., hybridizing it with convolutional
operations). We can also convert (E.13.24) to a causal
(recurrent) variant by properly restricting the summation.

13.4.2 The Receptance Weighted Key Value

(RWKV) model

The RWKV model [PAA+23] extends the ATF layer by
incorporating a few additional architectural modifications.
At the time of writing, this is one of the only pre-trained
RNNs matching transformers at the largest scale, so we
describe it in more detail. First, the relative embeddings
are simplified by considering a single vector w ∼ (e) which

335


336

Additional variants

is scaled for each offset:

wi j

= −(i − j)w

In addition, experiments showed that having a separate
offset u (in place of w) for the current element is beneficial.
Written in causal form, this gives:

hi

= Wo

(cid:18)
σ(qi

)⊙

(cid:80)i−1

j=1 exp (cid:0)k j
(cid:80)i−1

+ wi j
j=1 exp (cid:0)k j

(cid:1) ⊙ v j
+ wi j

+ exp (ki
(cid:1) + exp (ki

(cid:19)

+ u) ⊙ vi
+ u)

where we highlight the differences from the basic ATF layer
in red. The query is called the receptance in [PAA+23],
and an additional output projection Wo is added at the end.
Second, the RWKV model modifies the standard MLP in
the transformer block with a differently gated token-wise
block. For a given input token x this can be written as:

y = σ(W1x) ⊙ W2 max(0, W3x)2

(E.13.26)

where W1, W2, and W3 are trainable parameters. This
is a standard MLP except for the left-most gate and the
use of the squared ReLU. As a final modification, all three
projections in the first block (and also the two appearances
of x in (13.4.2)) are replaced with convex combinations of
xi and xi−1 to improve performance, which is called token
shift.

13.4.3 Selective state space models

We have seen three classes of recurrent models: standard
recurrent layers (and their gated versions), linearized
attention layers, and structured state space models.

336


Chapter 13: Recurrent models

337

Figure F.13.4: Mamba block
(residual connections around
the block and normalization
are not shown). σ is the
sigmoid function. Adapted
from [GD23].

Although they look different, it is relatively easy to move
from one class of models to the other. To see this, let us
consider a linearized attention layer where we ignore the
denominator:

Si

= Si−1
hi

+ φ(ki
= φ(qi

)v⊤
i
)⊤Si

(E.13.27)

(E.13.28)

Apart from the matrix-valued state, we see this has the
form of a SSM layer, except that some matrices (e.g., C =
)⊤) are not fixed but they depend on the specific input
φ(qi
token. From the point of view of dynamic systems, we say
that standard SSMs describe time-invariant systems, while
(E.13.27)-(E.13.28) describe a time-varying system. This
has inspired another class of SSM layers whose matrices
are not constrained to be time-invariant, which have been
called selective SSMs. Most of these models leverage the
idea of attention layers of projecting the input multiple
times before the layer’s computations.

As an example, we focus here on the so-called Mamba layer
[GD23] which, at the time of writing, is one of the few
SSM layers that was scaled to match the performance of

337

LinearConvolutionMamba SSMLinearLinear
338

Additional variants

transformer models at very large contexts and parameters’
counts. First, in order to make the SSM layer time-varying,
a subset of its matrices are made input-dependent:5

si

= A(xi
hi

)si−1
= C(xi

+ B(xi
)si

)xi
+ Dxi

(E.13.29)

(E.13.30)

where A(•), B(•), and C(•) are linear projections of their
input tokens. To make this feasible, the layer is applied to
each channel of the input independently, and the transition
matrix is selected as diagonal, so that all matrices of the
SSM can be represented with a single vector of values. This
layer looses a simple parallel scan implementation and
requires a customized hardware-aware implementation
[GD23]. It can be shown that the Mamba SSM variant and
several other SSM layers are degenerate case of a gated
recurrent layer [GJG+21, GD23].

To make the overall architecture simpler, Mamba avoids
in favour of a gated
alternating MLPs and SSMs,
architecture (similar to the gated attention unit from
Section 11.3) where an MLP is used to weight the outputs
from the SSM. An additional depthwise convolution is
added for improved flexibility - see Figure F.13.4.

5The matrix D can be seen as a simple residual connection and
it is left untouched. The original layer has a slightly different
parameterization where A = exp(∆¯A), for some trainable ¯A and input-
dependent scalar value ∆. This does not change our discussion.

338


Goodbye (for now)

And so, Alice’s first trip in this differentiable wonderland
has come (for now) to an end. We only made a very broad
tour, with a focus on the many ways layers can be
designed and composed to create modern differentiable
models (a.k.a., neural networks).

are

many
There
discussed
topics we
including
only briefly,
how we can use these
models in practice: from
fine-tuning to generative
continual
modeling,
learning, multimodality,
explainability, and more.

We also avoided pesky
engineering
aspects:
training and serving large models is a huge engineering
feat which requires, among other things, distributed
training
and DevOps
techniques.6 And the emergence of LLMs has opened up
new avenues for their use where knowledge of their inner
from prompt
workings is not even a prerequisite,

compilers,

strategies,

fast

6As an example, see https://jax-ml.github.io/scaling-book/.

339


340

Additional variants

engineering to model chaining and agentic behaviours.

This book has a companion website,7 where I hope to
publish additional chapters that touch upon some of these
topics.
If time allows, some of them may be joined
together in a new volume.

I hope you appreciated the journey!
For comments,
suggestions, and feedback on the book do not hesitate to
contact me.

7https://sscardapane.it/alice-book

340
