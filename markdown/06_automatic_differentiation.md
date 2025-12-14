6 | Automatic

differentiation

About this chapter

The previous chapter highlighted the need for an
efficient, automatic procedure to compute gradients of
any possible sequence of operations. In this chapter we
describe such a method, called back-propagation in the
neural network’s literature or reverse-mode automatic
differentiation in the computer science one. Its analysis
has several insights, ranging from the model’s choice to
the memory requirements for optimizing it.

6.1 Problem setup

We consider
the problem of efficiently computing
gradients of generic computational graphs, such as those
induced by optimizing a scalar loss function on a
called automatic
fully-connected model,
a
differentiation (AD) [BPRS18].
You can think of a
computational graph as the set of atomic operations
(which we call primitives) obtained by running the

task

123


124

Problem setup

program itself. We will consider sequential graphs for
brevity, but everything can be easily extended to acyclic
and
graphs
[GW08, BR24] .

computational

even more

generic

The problem may seem trivial, since the chain rule of
Jacobians (Section 2.2, (E.2.23)) tells us that the gradient
of function composition is simply the matrix product of
the corresponding Jacobian matrices. However, efficiently
implementing this process is the key challenge of this
chapter, and the resulting algorithm (reverse-mode AD or
backpropagation) is a cornerstone of neural networks
and differentiable programming in general [GW08, BR24].
Understanding it is also key to understanding the design
(and the differences)
for
implementing and training such programs (such as
TensorFlow or PyTorch or JAX). A brief history of the
algorithm can be found in [Gri12].

frameworks

of most

To setup the problem, we assume we have at our disposal
a set of primitives:

y = fi

(x, wi

)

), parameterized by the vector wi

Each primitive represents an operation on an input vector
) (e.g., the
x ∼ (ci
weights of a linear projection), and giving as output another
vector y ∼ (c′
i

∼ (pi

).

There is a lot of flexibility in our definition of primitives,
which can represent basic linear algebra operations (e.g.,
matrix multiplication), layers in the sense of Chapter 5 (e.g.,
a fully-connected layer with an activation function), or even
larger blocks or models. This recursive composability is a
key property of programming and extends to our case.

124


Chapter 6: Automatic differentiation

125

We only assume that for each primitive we know how to
compute the partial derivatives with respect to the two
input arguments, which we call the input Jacobian and
the weight Jacobian of the operation:

Input Jacobian:

Weight Jacobian:

∂
x

∂

w

[ f (x, w)] ∼ (c′, c)
[ f (x, w)] ∼ (c′, p)

These are reasonable assumptions since we restrict our
analysis to differentiable models. Continuous primitives
with one or more points of non-differentiability, such as
the ReLU, can be made to fit into this framework with the
use of subgradients (Section 6.4.4). Non differentiable
operations such as sampling or thresholding can also be
included by finding a relaxation of their gradient or an
equivalent estimator [NCN+23]. We cover the latter case
in the next volume.

On our notation and higher-order Jacobians

We only consider vector-valued quantities for readability,
as all resulting gradients are matrices. In practice, existing
primitives may have inputs, weights, or outputs of higher
rank. For example, consider a basic fully-connected layer
on a mini-batched input:

f (X, W) = XW + b

In this case, the input X has shape (n, c), the weights have
shape (c, c′) and (c′) (with c′ a hyper-parameter), and the
output has shape (n, c′). Hence, the input Jacobian has
shape (n, c′, n, c), and the weight Jacobian has shape
(n, c′, c, c′), both having rank 4.

In our notation, we can consider the equivalent flattened

125


126

Problem setup

vectors x = vect(X) and w = [vect(W); b], and our
resulting “flattened” Jacobians have shape (nc′, nc) and
(nc′, cc′) respectively. This is crucial in the following, since
every time we refer to “the input size c” we are referring to
“the product of all
input shapes”,
including eventual
mini-batching dimensions. This also shows that, while we
may know how to compute the Jacobians, we may not
wish to fully materialize them in memory due to their
large dimensionality.

As a final note, our notation aligns with the way these
primitives are implemented in a functional library, such as
JAX. In an object-oriented framework (e.g., TensorFlow,
PyTorch), we saw that layers are implemented as objects
(see Box C.5.1 in the previous chapter), with the
parameters being a property of the object, and the
function call being replaced by an object’s method. This
style simplifies certain practices,
such as deferred
initialization of all parameters until the input shapes are
known (lazy initialization), but it adds a small layer of
abstraction to consider to translate our notation into
workable code. As we will see, these differences are
reflected in turn in the way AD is implemented in the two
frameworks.

6.1.1 Problem statement

With all these details out of the way, we are ready to state
the AD task. Consider a sequence of l primitive calls,
followed by a final summation:

126


Chapter 6: Automatic differentiation

127

h1
h2

= f1
= f2
...
= fl
hl
y = (cid:88)

(x, w1
)
(h1, w2

)

)

(hl−1, wl
hl

This is called an evaluation trace of the program.
Roughly, the first l − 1 operations can represent several
layers of a differentiable model, operation l can be a
per-input loss (e.g., cross-entropy), and the final operation
sums the losses of the mini-batch. Hence, the output of
our program is always a scalar, since we require it for
numerical optimization. We abbreviate the previous
program as F (x).

Definition D.6.1 (Automatic differentiation)

Given a program F (x) composed of a sequence of
differentiable primitives, automatic differentiation
(AD) refers to the task of simultaneously and efficiently
computing all weight Jacobians of the program given
knowledge of
computational graph and all
individuals input and weight Jacobians:

the

AD(F (x)) = (cid:8)∂
wi

y(cid:9)l

i=1

As we will see,
there are two major classes of AD
algorithms, called forward-mode and backward-mode,
corresponding to a different ordering in the composition
of the individual operations. We will also see that the
backward-mode (called back-propagation in the neural

127


128

Problem setup

networks’ literature) is significantly more efficient in our
context. While we focus on a simplified scenario, it is
relatively easy to extend our derivation to acyclic graphs
of primitives (as already mentioned), and also to
situations where parameters are shared across layers
(weight sharing). We will see an example of weight
sharing in Chapter 13.

6.1.2 Numerical and symbolic

differentiation

Before moving on to forward-mode AD, we comment on
the difference between AD and other classes of algorithms
for differentiating functions. First, we could directly apply
the definition of gradients (Section 2.2) to obtain a suitable
numerical approximation of the gradient. This process is
called numerical differentiation. However, each scalar
value to be differentiated requires 2 function calls in a naive
implementation, making this approach unfeasible except
for numerical checks over the implementation.

Second, consider this simple function:

f (x) = a sin(x) + b x sin(x)

We can ask a symbolic engine to pre-compute the full,
symbolic equation of the derivative.
This is called
symbolic differentiation and shown in Python in Box
C.6.1.

In a realistic implementation, the intermediate value
h = sin(x) would be computed only once and stored in an
intermediate variable, which can also be reused for the
corresponding computation in the gradient trace (and a
similar reasoning goes for the cos(x) term in the

128


Chapter 6: Automatic differentiation

129

import sympy as sp
x, a, b = sp.symbols('x a b')
y = a*sp.sin(x) + b*x*sp.sin(x)
sp.diff(y, x)
# [Out]: acos(x)+bxcos(x)+bsin(x)

Box C.6.1: Symbolic differentiation in Python using SymPy.

derivative). This is less trivial than it appears: finding an
optimal implementation for the Jacobian which avoids any
unnecessary computation is an NP-complete task (optimal
Jacobian accumulation). However, we will see that we
can exploit the structure of our program to devise a
suitably efficient implementation of AD that is significantly
better than a symbolic approach like the above (and it is,
in fact, equivalent to a symbolic approach allowing for the
presence of subsequences [Lau19]).

6.2 Forward-mode differentiation

We begin by recalling the chain rule of Jacobians. Consider
a combination of two primitive functions:

h = f1

(x) , y = f2

(h)

In terms of their gradients, we have:

x y = ∂
∂

h y · ∂

x h

If x, h, and y have dimensions a, b, and c respectively,
the previous Jacobian requires the multiplication of a c × b
matrix (in green) with a b×a one (in red). We can interpret
the rule as follows: if we have already computed f1 and its

129


130

Forward-mode differentiation

Jacobian (red term), once we apply f2 we can “update” the
gradient by multiplying with the corresponding Jacobian
(green term).

called

forward-mode

We can immediately apply this insight to obtain a working
automatic
algorithm
differentiation (F-AD). The idea is that every time we
apply a primitive function, we initialize its corresponding
weight Jacobian (called tangent in this context), while
simultaneously updating all previous tangent matrices. Let
us see a simple worked-out example to illustrate the main
algorithm.

Consider the first instruction, h1
in our
program. Because nothing has been stored up to now, we
initialize the tangent matrix for w1 as its weight Jacobian:

= f1

(x, w1

),

(cid:210)W1

= ∂

w1

h1

We now proceed to the second instruction, h2
We update
simultaneously initializing the second one:

previous

).
(h1, w2
tangent matrix while

= f2

the

Input Jacobian of f2

(cid:210)W1

← (cid:2)∂

h1

(cid:3)

h2

(cid:210)W1

(cid:210)W2

= ∂

w2

h2

Updated tangent matrix for w1

The update requires the input Jacobian of the primitive,
while the second term requires the weight Jacobian of
the primitive. Abstracting away, consider the generic i-
). We initialize the
= fi
th primitive given by hi
tangent matrix for wi while simultaneously updating all

(hi−1, wi

130


Chapter 6: Automatic differentiation

131

previous matrices:

Input Jacobian of fi

(cid:104)

←

(cid:210)W j

∂

hi−1

hi

(cid:105)

(cid:210)W j

∀ j < i

(cid:210)Wi

= ∂

wi

hi

Weight Jacobian of fi

There are i−1 updates in the first row (one for each tangent
matrix we have already stored in memory), with the red
term – the input Jacobian of the i-th operation – being
shared for all previous tangents. The last operation in the
program is a sum, and the corresponding gradient gives us
the output of the algorithm:1

∇

wi

y = 1⊤

(cid:210)Wi

∀i

(E.6.1)

Done! Let us analyze the algorithm in more detail. First,
all the operations we listed can be easily interleaved with
the original program, meaning that the space complexity
will be roughly proportional to the space complexity of the
program we are differentiating.

On the negative side, the core operation of the algorithm
(the update of (cid:210)Wi) requires a multiplication of two
matrices, generically shaped (c′
), where
ci, c′
i are input/output shapes, and p j is the shape of w j.
This is an extremely expensive operation: for example,
assume that inputs and outputs are both shaped (n, d),

) and (ci, p j

i , ci

1To be fully consistent with notation, the output of (E.6.1) is a row
vector, while we defined the gradient as a column vector. We will ignore
this subtle point for simplicity until it is time to define vector-Jacobian
products later on the in the chapter.

131


132

Reverse-mode differentiation

where n is the mini-batch dimension and d represents the
input/output features. Then, the matrix multiplication
will have complexity (cid:79) (n2d 2p j
), which is quadratic in
both mini-batch size and feature dimensionality. This can
easily become unfeasible, especially for high-dimensional
inputs such as images.

We can obtain a better trade-off by noting that the last
operation of the algorithm is a simpler matrix-vector
product, which is a consequence of having a scalar output.
This is explored in more detail in the next section.

6.3 Reverse-mode differentiation

To proceed, we unroll the computation of a single gradient
term corresponding to the i-th weight matrix:

∇

wi

y = 1⊤(cid:2)∂

hl−1

(cid:3) · · · (cid:2)∂
hi

hl

hi+1

(cid:3)(cid:2)∂

wi

(cid:3)

hi

(E.6.2)

Remember that, notation apart, (E.6.2) is just a potentially
long series of matrix multiplications, involving a constant
term (a vector 1 of ones), a series of input Jacobians (the
red term) and a weight Jacobian of the corresponding
weight matrix (the green term). Let us define a shorthand
for the red term:

= 1⊤

(cid:101)hi

l
(cid:89)

j=i+1

∂

h j−1

h j

(E.6.3)

Because matrix multiplication is associative, we can
perform the computations in (E.6.2) in any order. In F-AD,
we proceeded from the right
since it
corresponds to the ordering in which the primitive
functions were executed. However, we can do better by
noting two interesting aspects:

to the left,

132


Chapter 6: Automatic differentiation

133

1. The leftmost term in (E.6.2) is a product between a
vector and a matrix (which is a consequence of
having a scalar
term in output), which is
computationally better than a product between two
matrices. Its output is also another vector.

2. The term in (E.6.3) (the product of all

input
Jacobians from layer i to layer l) can be computed
recursively starting from the last term and iteratively
multiplying by the input Jacobians in the reverse
order.

We can put together these observations to develop a second
approach to automatic differentiation, that we call reverse-
mode automatic differentiation (R-AD), which is outlined
next.

1. Differently from F-AD, we start by executing the entire
program to be differentiated, storing all intermediate
outputs.

2. We inizialize a vector (cid:101)h = 1⊤, which corresponds to

the leftmost term in (E.6.2).

3. Moving in reverse order, i.e., for an index i ranging
in l, l − 1, l − 2, . . . , 1, we first compute the gradient
with respect to the i-th weight matrix as:

∂

wi

y = (cid:101)h (cid:2)∂

wi

(cid:3)

hi

which is the i-th gradient we need. Next, we update
our “back-propagated” input Jacobian as:

(cid:101)h ← (cid:101)h (cid:2)∂

hi−1

(cid:3)

hi

Steps (1)-(3) describe a program which is roughly

133


134

Reverse-mode differentiation

symmetrical to the original program, that we call the dual
or reverse program. The terms (cid:101)h are called the adjoints
and they store (sequentially) all the gradients of the
output with respect to the variables h1, h2, . . . , hl in our
program.2

In the terminology of neural networks, we sometimes say
that the original (primal) program is a forward pass (not
to be confused with forward-mode), while the reverse
program is a backward pass. Differently from F-AD, in
R-AD the full primal program must be executed before the
reverse program can be run, and we need specialized
mechanisms to store all intermediate outputs to “unroll”
the computational graph.
frameworks
Different
implement this differently, as outlined next.

Computationally, R-AD is significantly more efficient than
F-AD. In particular, both operations in step (3) of R-AD are
vector-matrix products scaling only linearly in all shape
quantities. The tradeoff is that executing R-AD requires a
large amount of memory, since all intermediate values of
the primal program must be stored on disk with a suitable
such as gradient
strategy.
checkpointing, can be used to improve on this tradeoff by
increasing computations and partially reducing the
memory requirements. This is done by only storing a few
(called checkpoints) while
intermediate
recomputing the remaining values during the backward
pass. See Figure F.6.1 for a visualization.

Specific techniques,

outputs

2Compare this with F-AD, where the tangents represented instead

the gradients of the hi variables with respect to the weights.

134


Chapter 6: Automatic differentiation

135

(a)

(b)

(c)

(d)

Figure F.6.1: An example of gradient checkpointing. (a) We
execute a forward pass, but we only store the outputs of the first,
second, and fourth blocks (checkpoints). (b) The backward pass
(red arrows) stops at the third block, whose activations are not
available. (c) We run a second forward pass starting from the
closest checkpoint to materialize again the activations. (d) We
complete the forward pass. Compared to a standard backward
pass, this requires 1.25x more computations. In general, the less
checkpoints are stored, the higher the computational cost of the
backward pass.

6.4 Practical considerations

6.4.1 Vector-Jacobian products

Looking at step (3) in the R-AD algorithm, we can make
an interesting observation: the only operation we need
is a product between a row vector v and a Jacobian of f
(either the input or the weight Jacobian). We call these two
operations the vector-Jacobian products (VJPs) of f .3 In
the next definition we restore dimensional consistency by
adding a transpose to the vector.

3By contrast, F-AD can be formulated entirely in terms of the
transpose of the VJP, called a Jacobian-vector product (JVP). For a
one-dimensional output, the JVP is the directional derivative (E.2.20)
from Section 2.2. Always by analogy, the VJP represents the application
of a linear map connected to infinitesimal variations of the output of
the function, see [BR24].

135


136

Practical considerations

Definition D.6.2 (Vector-Jacobian product (VJP))

Given a function y = f (x), with x ∼ (c) and y ∼ (c′), its
VJP is another function defined as:

vjp f

(v) = v⊤∂ f (x)

(E.6.4)

where v ∼ (c′).
f (x1, . . . , xn
vjp f ,x1

(v), ..., vjp f ,xn

(v).

f has multiple parameters
), we can define n individual VJPs denoted as

If

In particular, in our case we can define two types of VJPs,
corresponding to the partial derivative of the primitive with
respect to the input and the weight arguments:

vjp f ,x
vjp f ,w

(v) = v⊤∂
(v) = v⊤∂

x f (x, w)
w f (x, w)

(E.6.5)

(E.6.6)

We can now rewrite the two operations in step (3) of the R-
AD algorithm as two VJP calls of the primitive function with
the adjoint values (ignoring the i indices for readability),
corresponding to the adjoint times the weight VJP, and the
adjoint times the input VJP:

w y = vjp f ,w
∂
(cid:0)
(cid:101)h ← vjp f ,h

(cid:0)
(cid:101)h(cid:1)
(cid:101)h(cid:1)

(E.6.7)

(E.6.8)

Hence, we
can implement an entire automatic
differentiation system by first choosing a set of primitives
and then augmenting them with the
operations,
corresponding VJPs, without having to materialize the
This is shown
Jacobians in memory at any point.

136


Chapter 6: Automatic differentiation

137

Figure F.6.2: For performing R-AD, primitives must be augmented
with two VJP operations to be able to perform a backward pass,
corresponding to the input VJP (E.6.5) and the weight VJP (E.6.6).
One call for each is sufficient to perform the backward pass through
the primitive, corresponding to (E.6.7)-(E.6.8).

schematically in Figure F.6.2.

In fact, we can recover the Jacobians’ computation by
repeatedly calling the VJPs with the basis vectors
e1, . . . , en, to generate them one row at a time, e.g., for the
input Jacobian we have:

x f (x, w) =
∂







vjp f ,x
vjp f ,x
...
vjp f ,x

(e1
(e2

(en



)
)




)

To understand why this reformulation can be convenient,
let us look at the VJPs of a fully-connected layer, which
is composed of linear projections and (elementwise) non-
linearities. First, consider a simple linear projection with
no bias:

f (x, W) = Wx

The input Jacobian here is simply W, but the weight
Jacobian is a rank-3 tensor (Section 2.2). By comparison,

137

Forward passBackward pass
138

Practical considerations

the input VJP has no special structure:

vjp f ,x

(v) = v⊤W⊤ = [Wv]⊤

(E.6.9)

The weight VJP, instead, turns out to be a simple outer
product, which avoids rank-3 tensors completely:

vjp f ,w

(v) = vx⊤

(E.6.10)

Working out the VJP

(cid:80)

To compute (E.6.10), we can write y = v⊤Wx =
(cid:80)
=
j Wi j vi x j, from which we immediately get
vi x j, which is the elementwise definition of the outer
product.

∂ y
∂ Wi j

i

Hence, every time we apply a linear projection in the
forward pass, we modify the back-propagated gradients by
the transpose of its weights, and we perform an outer
product to compute the gradient of W.

Consider now an element-wise activation function with no
trainable parameters, e.g., the ReLU:

f (x, {}) = φ(x)

Because we have no trainable parameters, we need only
consider the input VJP. The gradient is a diagonal matrix
having as elements the derivatives of φ:

[∂
x

φ(x)]

ii

= φ′(x i

)

The input VJP is a multiplication of a diagonal matrix by a
vector, which is equivalent to an Hadamard product (i.e., a

138


Chapter 6: Automatic differentiation

139

# Original function (sum-of-squares)
def f(x: Float[Array, "c"]):

return (x**2).sum()

grad_f = func.grad(f)
print(grad_f(torch.randn(10)).shape)
# [Out]: torch.Size([10])

Box C.6.2: Gradient computation as a higher-order function.
The torch.func interface replicates the JAX API. In practice, the
function can be traced (e.g., with torch.compile) to generate
an optimized computational graph.

scaling operation):

vjpx

( f , v) = v ⊙ φ′(x)

(E.6.11)

Interestingly, also in this case we can compute the VJP
without having to materialize the full diagonal matrix.

6.4.2 Implementing a R-AD system

There are many ways to implement the R-AD system,
ranging form Wengert lists (as done in TensorFlow) to
source-to-source code transformations [GW08]. Here, we
discuss briefly some common implementations in existing
frameworks.

First, describing primitives as functions with two arguments
f (x, w) aligns with functional frameworks such as JAX,
where everything is a function. Consider a function f (x)
with a c-dimensional input and a c′-dimensional output.
From this point of view, a VJP can be implemented as a
higher-order function with signature:

((cid:82)c → (cid:82)c′) → (cid:82)c → ((cid:82)c′ → (cid:82)c)

(E.6.12)

139


140

Practical considerations

i.e., given a function f and an input x′, a VJP returns
another function that can be applied to a c′-dimensional
vector v to return v⊤∂ f (x′). Similarly, the gradient for a
one-dimensional function can be implemented as another
higher-order function with signature:

((cid:82)c → (cid:82)) → ((cid:82)c → (cid:82)c)

(E.6.13)

taking as input the function f (x) and returning another
function that computes ∇ f (x).
In JAX, these ideas are
implemented in the functions jax.grad and jax.jvp
respectively, which is also replicated in PyTorch in the
torch.func module - see Box C.6.2 for an example.4

As we mentioned, in practice our models are implemented
as compositions of objects whose parameters are
encapsulated as properties (Box C.5.1). One possibility is
to “purify” the object to turn it into a pure function, e.g.:5

# Extract the parameters
params = dict(model.named_parameters())
# Functional call over the
# model's forward function
y = torch.func.functional_call(

model, params, x
)

More in general, frameworks like PyTorch are augmented
with techniques to handle this scenario directly, without
introducing intermediate operations.
In PyTorch, for
example, tensors’ objects are augmented with information
about the operation that generated them (Figure F.6.3,

4Many operations, such as computing an Hessian, can be achieved
by smartly composing JVPs and VJPs based on their signatures: https:
//jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html.

5https://sjmielke.com/jax-purify.htm

140


Chapter 6: Automatic differentiation

141

Figure F.6.3: Left:
in PyTorch, a tensor is augmented with
information about its gradient (empty at initialization), and about
the operation that created it. Right: during a backward pass, the
grad_fn property is used to traverse the computational graph in
reverse, and gradients are stored inside the tensor’s grad property
whenever requires_grad is explicitly set to True (to avoid
consumming unnecessary memory).

left). Whenever a backward() call is requested on a
scalar value, these properties are used to traverse the
computational graph in reverse, storing the corresponding
gradients inside the tensors that requires them (Figure
F.6.3, right).

This is just a high-level overview of how these systems are
implemented in practice, and we are leaving behind many
details, for which we refer to the official documentations.6

6.4.3 Choosing an activation function

Coincidentally, we can now motivate why ReLU is a good
choice as activation function. A close look at (E.6.11) tells
us that every time we add an activation function in our
model, the adjoints in the backward pass are scaled by a
factor of φ′(x). For models with many layers, this can give
rise to two pathological behaviors:

6I definitely suggest trying to implement an R-AD system from
scratch: many didactical implementations can be found online, such
as https://github.com/karpathy/micrograd.

141

datagradgrad_fndatagradgrad_fnrequires_grad=Truerequires_grad=Falsebackward()jvp(...)datagradgrad_fnrequires_grad=TrueTensor dataGradient dataPointer forcomputational graph
142

Practical considerations

1. If φ′(·) < 1 everywhere, there is the risk of the
gradient being shrank to 0 exponentially fast in the
number of layers. This is called the vanishing
gradient problem.

2. Conversely, if φ′(·) > 1 everywhere, the opposite
problem appears, with the gradients exponentially
converging to infinity in the number of layers. This
is called the exploding gradient problem.

These are serious problems in practice, because libraries
represent floating point numbers with limited precision
(typically 32 bits or lower), meaning that underflows or
overflows can manifest quickly when increasing the number
of layers.

Linear non-linear models

Surprisingly, a stack of linear layers implemented in
floating point precision is not fully linear because of small
discontinuities at machine precision! This is generally
not an issue, but it can be exploited to train fully-linear
deep neural networks.a

ahttps://openai.com/research/nonlinear-computation-in-

deep-linear-networks

As an example of how vanishing gradients can appear,
consider the sigmoid function σ(s). We already mentioned
that this was a common AF in the past, due to it being a
soft approximation to the step function. We also know
that σ′(s) = σ(s)(1 − σ(s)). Combined with the fact that
σ(s) ∈ [0, 1], we obtain that:

σ′(s) ∈ [0, 0.25]

142


Chapter 6: Automatic differentiation

143

(a) Sigmoid

(b) ReLU

Figure F.6.4: (a) Plot of the sigmoid function (red) and its
derivative (green). (b) Plot of ReLU (red) and its derivative (green).

Hence, the sigmoid is a prime candidate for vanishing
gradient issues: see Figure F.6.4a.

Designing an AF that never exhibits vanishing or
exploding gradients is non trivial, since the only function
having φ′(s) = 1 everywhere is the identity function. We
then need a function which is “linear enough” to avoid
gradient issues, but “non-linear” enough to separate the
linear layers. The ReLU ends up being a good candidate
since:

sReLU(s) =
∂

(cid:168)

0 s < 0
1 s > 0

The gradient is either zeroed-out, inducing sparsity in the
computation, or multiplied by 1, avoiding scaling issues -
this is shown in Figure F.6.4b.

As a side note, the ReLU’s gradient is identical irrespective
of whether we replace the input to the ReLU layer with its
output (since we are only masking the negative values while
keeping the positive values untouched). Hence, another
benefit of using ReLU as activation function is that we can
save a small bit of memory when performing R-AD, by
overwriting the layer’s input in the forward pass without

143

−10−50510s0.00.20.40.60.81.0Sigmoidanditsderivativeσ(s)σ0(s)−3−2−10123s0.00.51.01.52.02.53.0ReLUanditsderivativeReLU(s)DerivativeofReLU(s)
144

Practical considerations

impacting the correctness of the AD procedure: this is
done in PyTorch, for example, by setting the in_place
parameter.7

6.4.4 Subdifferentiability and AD

There is a small detail we avoided discussing until now:
the ReLU is non-differentiable in 0, making the overall
network non-smooth. What happens in this case? The
“pragmatic” answer is that, by minimizing with stochastic
gradient descent from a random (non-zero) initialization,
the probability of ending up exactly in s = 0 is practically
null, while the gradient is defined in ReLU(ϵ) for any |ϵ| >
0.

For a more technical answer, we can introduce the concept
of subgradient of a function.

Definition D.6.3 (Subgradient)

Given a convex function f (x), a subgradient in x is a
point z such that, for all y:

f ( y) ≥ f (x) + z( y − x)

Note the similarity with the definition of convexity: a
subgradient is the slope of a line “tangent” to f (x), such
that the entire f
is
differentiable in x, then only one such line exists, which is
the derivative of f in x. In a non-smooth point, multiple
subgradients exists, and they form a set called the

is lower bounded by it.

If

f

7https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

144


Chapter 6: Automatic differentiation

145

subdifferential of f in x:

∂

x f (x) = {z | z is a subgradient of f (x)}

With this definition in hand, we can complete our analysis
of the gradient of ReLU by replacing the gradient with its
subdifferential in 0:

sReLU(s) =
∂






{0}
{1}
[0, 1]

s < 0
s > 0
s = 0

Hence, any value in [0, 1] is a valid subgradient in 0, with
most implementations in practice favoring ReLU′(0) = 0.
Selecting subgradients at every step of an iterative descent
procedure is called subgradient descent.

In fact, the situation is even more tricky, because the
subgradient need not be defined for non-convex functions.
In that case, one can resort to generalizations that relax
the previous definition to a local neighborhood of x, such
as the Clarke subdifferential.8 Subdifferentiability can also
create problems in AD, where different implementations of
the same functions can provide different (possibly invalid)
subgradients, and more refined concepts of chain rules
must be considered for a formal proof [KL18, BP20].9

8https://en.wikipedia.org/wiki/Clarke_generalized_derivative
9Consider this example reproduced from [BP20]: define two
(s) = 0.5(ReLU(s) +
functions, ReLU2
(s)). They are both equivalent to ReLU, but in PyTorch a
ReLU2
backward pass in 0 returns 0.0 for ReLU, 1.0 for ReLU2, and 0.5 for
ReLU3.

(s) = ReLU(−s) + s and ReLU3

145


146

Practical considerations

From theory to practice

If you followed the exercises in Chapter
5, you already saw an application
of R-AD in both PyTorch and JAX,
and this chapter (especially Section
6.4.2)
should have clarified their
implementation.

It is a good idea to try and re-implement a simple R-AD
system, similar to the one of PyTorch. For example,
the micrograd
focusing on scalar-valued quantities,
repository10 is a very good didactical implementation. The
only detail we do not cover is that, once you move to a
general acyclic graph, an ordering of the variables in the
computational graph before the backward pass is essential
to avoid creating wrong backpropagation paths.
In
micrograd, this is achieved via a non-expensive topological
sorting of the variables.

It is also interesting to try and implement a new primitive
(in the sense used in this chapter) in PyTorch, which
requires specifying its forward pass along with its VJPs.11
One example can be one of the trainable activation
functions from Section 5.4. This is a didactical exercise, in
the sense that this can be implemented equivalently by
subclassing nn.Module and letting PyTorch’s AD engine
work out the backward pass.

All these steps can also be replicated in JAX:

• Implement a didactic version of

JAX with

10https://github.com/karpathy/micrograd
11https://pytorch.org/docs/master/notes/extending.html

146


Chapter 6: Automatic differentiation

147

autodidax.12

• Write out a new primitive by implementing the

corresponding VJP.13

• Read the JAX Autodiff Cookbook14 to discover
advanced use-cases for the automatic differentiation
engine, such as higher-order derivatives, Hessians,
and more.

12https://jax.readthedocs.io/en/latest/autodidax.html
13https://jax.readthedocs.io/en/latest/notebooks/Custom_

derivative_rules_for_Python_code.html

14https://jax.readthedocs.io/en/latest/notebooks/autodiff_

cookbook.html

147


148

Practical considerations

148


Part II

A strange land

“Curiouser and curiouser!” cried Alice

(she was so much surprised, that for

the moment she quite forgot

how to speak good English).

— Chapter 2, The Pool of Tears

149
