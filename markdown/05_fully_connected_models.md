5 | Fully-connected

models

About this chapter

In this chapter we show how differentiable models can
be built by composing a sequence of so-called fully-
connected layers. For historical reasons, these models
are also known as multilayer perceptrons (MLPs). MLPs
interleave linear blocks (similar to Chapter 4) with non-
linear functions, sometimes called activation functions.

5.1 The limitations of linear models

Linear models are fundamentally limited, in the sense that
by definition they cannot model non-linear relationships
across features. As an example, consider two input vectors
x and x′, which are identical except for a single feature
indexed by j:

=

x ′
i

(cid:168)

xi
2x i

if i ̸= j
otherwise

101


102

The limitations of linear models

For example, this can represent two clients of a bank, which
are identical in all aspects except for their income, with x′
having double the income of x. If f is a linear model (with
no bias) we have:

Original output

f (x′) = f (x) + w j x j

Change induced by x ′
j

= 2x j

Hence, the only consequence of the change in input is a
small linear change of output dictated by w j. Assuming we
are scoring the users, we may wish to model relationships
such as “an income of 1500 is low, except if the age < 30”.1
Clearly, this cannot be done with a linear model due to the
analysis above.

The prototypical example of this is the XOR dataset, a
two-valued dataset where each feature can only take
values in {0, 1}. Hence, the entire dataset is given by only
4 possibilities:

f ([0, 0]) = 0 , f ([0, 1]) = 1 , f ([1, 0]) = 1 , f ([1, 1]) = 0

where the output is positive whenever only one of the two
inputs is positive. Despite its simplicity, this is also
non-linearly separable, and cannot be solved with 100%
accuracy by a linear model - see Figure F.5.1 for a
visualization.

1You probably shouldn’t do credit scoring with machine learning

anyways.

102


Chapter 5: Fully-connected models

103

Figure F.5.1: Illustration of the XOR dataset: green squares are
values of one class, red circles are values of another class. No linear
model can separate them perfectly (putting all squares on one side
and all circles on the other side of the decision boundary). We say
that the dataset is not linearly separable.

5.2 Composition and hidden layers

A powerful idea in programming is decomposition, i.e.,
breaking down a problem into its constituent parts
recursively, until each part can be expressed in simple,
manageable operations.
Something similar can be
achieved in our case by imagining that our model f is, in
fact, the composition of two trainable operations:

f (x) = ( f2

◦ f1

)(x)

)(x) = f2

◦ f1 is the composition of the two functions:
where f2
(x)), and we assume that each function
◦ f1
( f2
instantiates its own set of trainable parameters. We can
keep subdividing the computations:

( f1

103

010011
104

Composition and hidden layers

f (x) = ( fl

◦ fl−1

◦ · · · ◦ f2

◦ f1

)(x)

where we now have a total of l functions that are being
composed. Note that as long as each fi does not change
the “type” of its input data, we can chain together as many
of these transformations as we want, and each one will add
its own set of trainable parameters.

For example, in our case the input x is a vector, hence any
vector-to-vector operation (e.g., a matrix multiplication
(x) = Wx) can be combined together an endless number
fi
of times. However, some care must be taken. Suppose we
chain together two different linear projections:

h = f1
y = f2

(x) = W1x + b1
(h) = w⊤
2 h + b2

(E.5.1)

(E.5.2)

It is easy to show that the two projections “collapse” into a
single one:

y = (w⊤

)
2 W1
(cid:124) (cid:123)(cid:122) (cid:125)
≜ A

x + (w⊤
(cid:124)

2 b1
(cid:123)(cid:122)
≜ c

+ b2
)
(cid:125)

= Ax + c

The idea of fully-connected (FC) models, also known as
multi-layer perceptrons (MLPs) for historical reasons, is
to insert a simple elementwise non-linearity φ : (cid:82) → (cid:82)
in-between projections to avoid the collapse:

Element-wise non-linearity

h = f1

(x) = φ (W1x + b1

)

(E.5.3)

y = f2

(h) = w⊤

2 h + b2

(E.5.4)

104


Chapter 5: Fully-connected models

105

The second block can be linear, as in (E.5.4), or it can be
wrapped into another non-linearity depending on the task
(e.g., a softmax function for classification). The function
φ can be any non-linearity, e.g., a polynomial, a square
root, or the sigmoid function σ. As we will see in the next
chapter, choosing it has a strong effect on the gradients
of the model and, consequently, on optimization, and the
challenge is to select a φ which is “non-linear enough” to
prevent the collapse while staying as close as possible to
the identity in its derivative. A good default choice is the
so-called rectified linear unit (ReLU).

Definition D.5.1 (Rectified linear unit)

The rectified linear unit (ReLU) is defined elementwise
as:

ReLU(s) = max(0, s)

(E.5.5)

We will have a lot more to say on the ReLU in the next
chapter. With the addition of φ, we can now chain as many
transformations as we want:

y = w⊤
l

φ (Wl−1

(φ (Wl−2

φ (· · · ) + bl−2

)) + bl−1

) (E.5.6)

In the rest of the chapter we focus on analyzing training
and approximation properties of this class of models. First,
however, a brief digression on naming conventions.

On neural network terminology

As we already mentioned, neural networks have a long
history and a long baggage of terminology, which we briefly
summarize here. Each fi is called a layer of the model, with
fl being the output layer, fi, i = 1, . . . , l − 1 the hidden
layers and, with a bit of notational overloading, x being

105


106

Composition and hidden layers

the input layer. With this terminology, we can restate the
definition of the fully-connected layer in batched form
below.

Definition D.5.2 (Fully-connected layer)

For a batch of n vectors, each of size c, represented as a
matrix X ∼ (n, c), a fully-connected (FC) layer is defined
as:

FC(X) = φ (XW + b)
(E.5.7)
The parameters of the layer are the matrix W ∼ (c, c′)
and the bias vector b ∼ (c′), for a total of (c + 1)c′
parameters (assuming φ does not have parameters). Its
hyper-parameters are the width c′ and the non-linearity
φ.

(x) are called the activations of the layer,
The outputs fi
where we can sometimes distinguish between the
pre-activation and the post-activation (before and after
the non-linearity). The non-linearity φ itself can be called
the activation function. Each output of fi is called a
neuron. Although much of this terminology is outdated, it
is still pervasive and we will use it when needed.

The size of the each layer (the shape of the output) is
an hyperparameter that can be selected by the user, as it
only influences the input shape of the next layer, which
is known as the width of the layer. For a large number
of layers, the number of hyperparameters grows linearly
and their selection becomes a combinatorial task. We will
return on this point in Chapter 9, when we discuss the
design of models with dozens (or hundreds) of layers.

The layer concept
is also widespread in common
frameworks. A layer such as (E.5.7) can be defined as an

106


Chapter 5: Fully-connected models

107

class FullyConnectedLayer(nn.Module):

def __init__(self, c: int, cprime: int):

super().__init__()
# Initialize the parameters
self.W = nn.Parameter(

torch.randn(c, cprime))

self.b = nn.Parameter(

torch.randn(1, cprime))

def forward(self, x):

return relu(x @ self.W + self.b)

Box C.5.1: The FC layer in (E.5.7) implemented as an object in
PyTorch. We require a special syntax to differentiate trainable
parameters, such as W, from other non-trainable tensors:
in
PyTorch, this is obtained by wrapping the tensors in a Parameter
object. PyTorch also has its collection of layers in torch.nn,
including the FC layer (implemented as torch.nn.Linear).

object having two functions: an initialization function that
randomly initializes all parameters of the model based on
the selected hyper-parameters, and a call function that
provides the output of the layer itself. See Box C.5.1 for an
example. Then, a model can be defined by chaining
together instances of such layers. For example, in PyTorch
this can be achieved by the Sequential object:

model = nn.Sequential(

FullyConnectedLayer(3, 5),
FullyConnectedLayer(5, 4)

)

Note that from the point of view of their input-output
signature, there is no great difference between a layer as
defined in Box C.5.1 and a model as defined above, and
we could equivalently use model as a layer of a larger one.
This compositionality is a defining characteristic of

107


108

Composition and hidden layers

differentiable models.

5.2.1 Approximation properties of MLPs

Training MLPs proceeds similarly to what we discussed for
linear models. For example, for a regression task, we can
minimize the mean-squared error:

min
}l
{Wk,bk
k=1

1
n

(cid:88)

i

( yi

− f (xi

))2

where the minimization is now done on all parameters of
the model simultaneously. We will see in the next chapter
a general procedure to compute gradients in this case.

For now, we note that the main difference with respect to
having a linear model is that adding an hidden layer
makes the overall optimization problem non-convex, with
multiple local optima depending on the initialization of
the model. This is an important aspect historically, as
alternative approaches to supervised learning (e.g.,
support vector machines [HSS08]) provide non-linear
models while remaining convex. However, the results of
the last decade show that highly non-convex models can
achieve significantly good performance in many tasks.2

From a theoretical perspective, we can ask what is the
significance of having added hidden layers, i.e., if linear
models can only solve tasks which are linearly separable,
functions that can be
what
approximated by adding hidden layers? As it turns out,

is instead the class of

2The reason differentiable models generalize so well

is an
interesting, open research question, to which we return in Chapter
9. Existing explanations range from an implicit bias of (stochastic)
gradient descent [PPVF21] to intrinsic properties of the architectures
themselves [AJB+17, TNHA24].

108


Chapter 5: Fully-connected models

109

having a single hidden layer is enough to have universal
approximation capabilities. A seminal result in this sense
was proved by G. Cybenko in 1989 [Cyb89].

Theorem 5.1 (Universal approximation of MLPs)
Given a continuous function g : (cid:82)d → (cid:82), we can always
find a model f (x) of the form (E.5.3)-(E.5.4) (an MLP
with a single hidden layer) and sigmoid activation
functions, such that for any ϵ > 0:

| f (x) − g(x)| ≤ ϵ , ∀x

where the result holds over a compact domain. Stated
differently, one-hidden-layer MLPs are “dense” in the space
of continuous functions.

The beauty of this theorem should not distract from the fact
that this is purely a theoretical construct, that makes use
of the fact that the width of the hidden layer of the model
can grow without bounds. Hence, for any x for which the
previous inequality does not hold, we can always add a new
unit to reduce the approximation error (see Appendix B). In
fact, it is possible to devise classes of functions on which the
required number of hidden neurons grows exponentially
in the number of input features [Ben09].3

Many other authors, such as [Hor91], have progressively
refined this result to include models with fundamentally
In
any possible activation function, including ReLUs.
addition, universal approximation can also be proved for
models having finite width but possibly infinite depth

3One of these problems, the parity problem, is closely connected
to the XOR task: https://blog.wtf.sg/posts/2023-02-03-the-new-xor-
problem/.

109


110

Stochastic optimization

[LPW+17]. A separate line of research has investigated the
approximation capabilities of overparameterized models, in
which the number of parameters exceeds the training data.
In this case, training to a global optimum can be proved in
many interesting scenarios [DZPS19, AZLL19] (informally,
for sufficiently many parameters, the model can achieve
the minimum of the loss on each training sample and,
hence, the global minimum of the optimization problem).
See Appendix B for a one-dimensional visualization of
Cybenko’s theorem.

Approximation and learning capabilities of differentiable
models are immense fields of study, with countless books
devoted to them, and we have only mentioned some
significant results here. In the rest of the book, we will be
mostly concerned with the effective design of the models
themselves, whose behavior can be more complex and
difficult to control (and design) than these theorems
suggest.

5.3 Stochastic optimization

To optimize the models we can perform gradient descent
on the corresponding empirical risk minimization problem.
However, this can be hard to achieve when n (the size of
the dataset) grows very large. We will see in the next
chapter that computing the gradient of the loss requires a
time linear in the number of examples, which becomes
unfeasible or slow for n in the order of 104 or more,
especially for large models (memory issues aside).

Fortunately, the form of the problem lends itself to a nice
approximation, where we use subsets of the data to
compute a descent direction. To this end, suppose that for

110


Chapter 5: Fully-connected models

111

t

⊂ (cid:83)

iteration t of gradient descent we sample a subset
n of r points (with r ≪ n) from the dataset, which
(cid:66)
we call a mini-batch. We can compute an approximated
loss by only considering the mini-batch as:

(cid:101)Lt

= 1
r

(cid:88)

(xi , yi

)∈(cid:66)

t

l( yi, f (x i

)) ≈ 1
n

(cid:88)

(xi , yi

)∈(cid:83)

n

l( yi, f (x i

))

Mini-batch

(E.5.8)

Full dataset

If we assume the elements in the mini-batch are sampled
i.i.d. from the dataset, (cid:101)Lt is a Monte Carlo approximation of
the full loss, and the same holds for its gradient. However,
its computational complexity grows only with r, which
can be controlled by the user. Roughly speaking, lower
dimensions r of the mini-batch result in faster iterations
with higher gradient variance, while higher r results in
slower, more precise iterations. For large models, memory
is in general the biggest bottleneck, and the mini-batch size
r can be selected to fill up the available hardware for each
iteration.

Gradient descent applied on mini-batches of data is an
example of stochastic gradient descent (SGD). Due to
the properties discussed above, SGD can be proven to
converge to a minimum in expectation, and it is the
preferred
training
differentiable models.

strategy when

optimization

The last remaining issue is how to select the mini-batches.
For large datasets, sampling elements at random can be
expensive, especially if we need to move them back and
forth from the GPU memory. An intermediate solution that
lends itself to easier optimization is the following:

111


112

Stochastic optimization

Figure F.5.2: Building the mini-batch sequence: after shuffling,
stochastic optimization starts at mini-batch 1, which is composed
of the first r elements of the dataset. It proceeds in this way to
mini-batch b (where b = n
r , assuming the dataset size is perfectly
divisible by r). After one such epoch, training proceed with mini-
batch b +1, which is composed of the first r elements of the shuffled
dataset. The second epoch ends at mini-batch 2b, and so on.

1. Begin by shuffling the dataset.

2. Then, subdivide the original dataset into mini-batches
of r consecutive elements and process each of them
sequentially. Assuming a dataset of size n = r b, this
results in b mini-batches and hence b steps of SGD. If
we are executing the code on a GPU, this step includes
sending the mini-batch to the GPU memory.

3. After completing all mini-batches constructed in this

way, return to point 1 and iterate.

One complete loop of this process is called an epoch of
training, and it is a very common hyper-parameter to
for a dataset of 1000 elements and
specify (e.g.,
mini-batches of 20 elements, “training for 5 epochs” means
training for 250 iterations). The expensive shuffling

112

12bEpochShuffleDatasetSGDStep 1SGDStep 2SGDStep bRepeatShuffled datasetBuild mini-batches
Chapter 5: Fully-connected models

113

# A dataset composed by two tensors
dataset = torch.utils.data.TensorDataset(

torch.randn(1000, 3),
torch.randn(1000, 1))

# The data loader provides
# shuffling and mini-batching
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset,

shuffle=True,
batch_size=32)

for xb, yb in dataloader:

# Iterating over mini-batches (one epoch)
# xb has shape (32, 3)
# yb has shape (32, 1)

Box C.5.2: Building the mini-batch sequence with PyTorch’s data
loader: all frameworks provide similar tools.

operation is only done once per epoch, while in-between
an epoch mini-batches can be quickly pre-fetched and
optimized by the framework. This is shown schematically
in Figure F.5.2. Most frameworks provide a way to
organize the dataset into elements that can be individually
indexed, and a separate interface to build the mini-batch
sequence.
In PyTorch, for example, this is done by the
Dataset and DataLoader interfaces, respectively - see
Box C.5.2.

This setup also leads itself to a simple form of parallelism
If we assume each
across GPUs or across machines.
machine is large enough to hold an entire copy of the
model’s parameters, we can process different mini-batches
in parallel over the machines and then sum their local
contributions for
the final update, which is then
broadcasted back to each machine. This is called a data

113


114

Activation functions

Figure F.5.3: A simple form of distributed stochastic optimization:
we process one mini-batch per available machine or GPU (by
replicating the weights on each of them) and sum or average the
corresponding gradients before broadcasting back the result (which
is valid due to the linearity of the gradient operation). This requires
a synchronization mechanism across the machines or the GPUs.

parallel setup in PyTorch,4 and it is shown visually in
Figure F.5.3. More complex forms of parallelism, such as
tensor parallelism, are also possible, but we do not cover
them in this book.

5.4 Activation functions

We close the chapter by providing a brief overview on the
selection of activation functions. As we stated in the
previous section, almost any element-wise non-linearity is
theoretically valid. However, not all choices have good
As an example, consider a simple
performance.
for some user-defined positive
polynomial
integer p:

function,

φ(s) = sp

4https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

114

Dataset (main memory)GPU 2GPU 1ComputegradientComputegradientAggregate gradientsand broadcast back theparameters
Chapter 5: Fully-connected models

115

For large p,
this will grow rapidly on both sides,
compounding across layers and resulting in models which
are hard to train and with numerical instabilities.

Historically, neural networks were introduced as
approximate models of biological neurons (hence, the
name artificial NNs). In this sense, the weights w⊤ in the
dot product w⊤x were simple models of synapses, the bias
b was a threshold, and the neuron was “activated” when
the cumulative sum of the inputs surpassed the threshold:

s = w⊤x − b , φ(s) = (cid:73)

s≥0

where (cid:73)
b is an indicator function which is 1 when b is true,
Because this activation function is
0 otherwise.
non-differentiable, the sigmoid σ(s) can be used as a
soft-approximation. In fact, we can define a generalized
(s) = σ(as),
sigmoid function with a tunable slope a as σ
and we have:

a

lim
a→∞

σ

a

(s) = (cid:73)

s≥0

Another common variant was the hyperbolic tangent, which
is a scaled version of the sigmoid in [−1, +1]:

tanh(s) = 2σ(s) − 1

Modern neural networks, popularized by AlexNet in 2012
[KSH12], have instead used the ReLU function in (E.5.5).
The relative benefits of ReLU with respect to sigmoid-like
functions will be discussed in the next chapter. We note
here that ReLUs have several counter-intuitive properties.

115


116

Activation functions

For example, they have a point of non-differentiability in
0, and they have a large output sparsity since all negative
inputs are set to 0. This second property can result in what
is known as “dead neurons”, wherein certain units have a
constant 0 output for all inputs. This can be solved by a
simple variant of ReLU, known as Leaky ReLU:

LeakyReLU(s) =

(cid:168)

s
αs

if s ≥ 0
otherwise

(E.5.9)

for a very small α, e.g., α = 0.01. We can also train a
different α for each unit (as the function is differentiable
with respect to α). In this case, we call the AF a parametric
ReLU (PReLU) [HZRS15]. Trainable activation functions
are, in general, an easy way to add a small amount of
flexibility with a minor amount of parameters – in the case
of PReLU, one per neuron.

Fully-differentiable variants of ReLU are also available, such
as the softplus:

softplus(s) = log(1 + exp(s))

(E.5.10)

The softplus does not pass through the origin and it is
always greater than 0. Another variant, the exponential
linear unit (ELU), preserves the passage at the origin while
switching the lower bound to −1:

ELU(s) =

(cid:168)

s
exp(s) − 1

if s ≥ 0
otherwise

(E.5.11)

Yet another class of variants can be defined by noting the

116


Chapter 5: Fully-connected models

117

ReLU

LeakyReLU

Softplus

ELU

GELU

Figure F.5.4: Visual comparison of ReLU and four variants:
LeakyReLU (E.5.9), Softplus (E.5.10), ELU (E.5.11), and GELU.
LeakyReLU is shown with α = 0.1 for better visualization, but in
practice α can be closer to 0 (e.g., 0.01)..

similarity of ReLU with the indicator function. We can
rewrite the ReLU as:

ReLU(s) = s · (cid:73)

s≥0

Hence, ReLU is identical to the indicator function on the
negative quadrant, while replacing 1 with s on the positive
quadrant. We can generalize this by replacing the indicator
function with a weighting factor β(s):

GeneralizedReLU(s) = s · β(s)

Choosing β(s) as the cumulative Gaussian distribution
function, we obtain the Gaussian ELU (GELU) [HG16],
while for β(s) = σ(s) we obtain the sigmoid linear unit
(SiLU) [HG16], also known as the Swish [RZL17]. We
plot some of these AFs in Figure F.5.4. Apart from some
minor details
(e.g., monotonicity in the negative
quadrant), they are all relatively similar, and it is in
general very difficult to obtain a significant boost in
performance by simply replacing the activation function.

Multiple trainable variants of each function can be

117

−3−2−1012−10123−3−2−1012−10123−3−2−1012−10123−3−2−1012−10123−3−2−1012−10123
118

Activation functions

obtained by adding trainable parameters to the functions.
For example, a common trainable variant of the Swish
with four parameters {a, b, c, d} is obtained as:

Trainable-Swish(s) = σ(as + b)(cs + d)

(E.5.12)

We can also design non-parametric activation functions,
in the sense of activation functions that do not have a fixed
number of trainable parameters. For example, consider a
generic set of (non-trainable) scalar functions φ
i indexed
by an integer i. We can build a fully flexible activation
function as a linear combination of n such bases:

φ(s) =

n
(cid:88)

i=1

φ

α

i

i

(s)

(E.5.13)

where n is an hyper-parameter, while the coefficients α
i are
trained by gradient descent. They can be the same for all
functions, or different for each layer and/or neuron. Based
on the choice of φ
i we obtain different classes of functions:
if each φ
i is a ReLU we obtain the adaptive piecewise
linear (APL) function [AHSB14], while for more general
kernels we obtain the kernel activation function (KAF)
[MZBG18, SVVTU19]. Even more general models can be
obtained by considering functions with multiple inputs and
multiple outputs [LCX+23]. See [ADIP21] for a survey.

In general, there is no answer to the question of “what
is the best AF”, as it depends on the task, dataset, and
architecture. ReLU is a common choice because it performs
well, is highly optimized in code, and it has a minor cost
overhead.
It is important to consider the fundamental
computational trade-off that, for a given budget, more
complex AFs can result in having smaller width or smaller
depth, potentially hindering the performance of the entire

118


Chapter 5: Fully-connected models

119

architecture. For this reason, AFs with a lot of trainable
parameters are less common.

Design variants

Not every layer fits into the framework of linear
projections and element-wise non-linearities, and we
describe here three common variants. First, the gated
linear unit (GLU) [DFAG17] combines the structure of
(E.5.12) with multiplicative (Hadamard) interactions:

f (x) = σ (W1x) ⊙ (W2x)

(E.5.14)

where W1 and W2 are trained. Another possibility, the
SwiGLU, replaces the sigmoid in (E.5.14) with a Swish
function [Sha20]. Gated MLPs are in fact a popular
choice in modern LLMs, see Chapter 11.

Second, in a maxout network [GWFM+13] each unit
produces the maximum of k (hyper-parameter) different
projections. Finally, replacing the linear projection W
)
(x j
with a matrix of trainable non-linearities Wi j
of the form (E.5.13) has also been proposed recently
under the name of Kolmogorov-Arnold networks
(KANs, [LWV+24]):

→ φ

i j

hi

= (cid:88)
j

φ

(x j

)

i j

119


120

Activation functions

From theory to practice

This chapter has introduced two key
requirements for any general-purpose
framework for training differentiable
models:

1. A way

to

handle

large
datasets that need to be shuffled,
separated into mini-batches, and moved back and
forth from the GPU. In PyTorch, most of this is
implemented via the Dataset and DataLoader
interfaces, as in Box C.5.2.

2. A mechanism to build models from the combination
of basic blocks, known as layers. In PyTorch, layers
are implemented in the torch.nn module, and they
can be composed via the Sequential interface or
by subclassing the Module class, as in Box C.5.1.

I suggest you now try to replicate one of the many quick
guides available on the documentation of PyTorch.5
Everything should be reasonably clear, apart from the
gradient computation mechanism, introduced in the next
chapter. This is also a good time to investigate Hugging
Face Datasets, which combines a vast repository of
datasets with a framework-agnostic interface to process
and cache them backed by Apache Arrow.6

JAX does not provide high-level utilities. For data loading
you can use any existing tool, including PyTorch’s data
loaders and Hugging Face Datasets. For building models,

5https://pytorch.org/tutorials/beginner/basics/quickstart_

tutorial.html

6https://huggingface.co/docs/datasets/en/quickstart

120


Chapter 5: Fully-connected models

121

the easiest way is to rely on an external library. Because
JAX is fully functional, object-oriented abstractions like
Box C.5.1 are not possible. My personal suggestion is
Equinox [KG21], which provides a class-like experience by
combining the basic data structure of JAX (the pytree)
with callable nodes.

121


122

Activation functions

122
