2 | Mathematical
preliminaries

About this chapter

We compress here the mathematical concepts required to
follow the book. We assume prior knowledge on all these
topics, focusing more on describing specific notation and
giving a cohesive overview. When possible, we stress
the relation between some of this material (e.g., tensors)
and their implementation in practice.

The chapter is composed of three parts that follow
sequentially from each other, starting from linear algebra,
moving to the definition of gradients for n-dimensional
objects, and finally how we can optimize functions by
exploiting such gradients. A self-contained overview of
probability theory is given in Appendix A, with a focus on
the maximum likelihood principle.

This chapter is full of content and definitions: bear with
me for a while!

17


18

Linear algebra

2.1 Linear algebra

We recall here some basic concepts from linear algebra that
will be useful in the following (and to agree on a shared
notation). Most of the book revolves around the idea of a
tensor.

Definition D.2.1 (Tensors)

A tensor X is an n-dimensional array of elements of the
same type. In the book we use:

X ∼ (s1, s2, . . . , sn

)

to quickly denote the shape of the tensor.

For n = 0 we obtain scalars (single values), while we
have vectors for n = 1, matrices for n = 2, and higher-
dimensional arrays otherwise. Recall that we use lowercase
x for scalars, lowercase bold x for vectors, uppercase bold
X for matrices. Tensors in the sense described here are
fundamental in deep learning because they are well suited
to a massively-parallel implementation, such as using GPUs
or more specialized hardware (e.g., TPUs, IPUs).

A tensor is described by the type of its elements and its
shape. Most of our discussion will be centered around
tensors of floating-point values (the specific format of which
we will consider later on), but they can also be defined
for integers (e.g., in classification) or for strings (e.g., for
text). Tensors can be indexed to get slices (subsets) of
their values, and most conventions from NumPy indexing1

1If you want a refresher: https://numpy.org/doc/stable/user/basics.
indexing.html. For readability in the book we index from 1, not from
0. See also the exercises at the end of the chapter.

18


Chapter 2: Mathematical preliminaries

19

apply. For simple equations we use pedices: for example,
for a 3-dimensional tensor X ∼ (a, b, c) we can write X i to
denote a slice of size (b, c) or X i jk for a single scalar. We
use commas for more complex expressions, such as X i,:, j:k
to denote a slice of size (b, k − j). When necessary to avoid
clutter, we use a light-gray notation:

[X ]

i jk

to visually split the indexing part from the rest, where the
argument of [ • ] can also be an expression.

2.1.1 Common vector operations

We are mostly concerned with models that can be written
as composition of differentiable operations. In fact, the
majority of our models will consist of basic compositions of
sums, multiplications, and some additional non-linearities
such as the exponential exp(x), sines and cosines, and
square roots.

Vectors x ∼ (d) are examples of 1-dimensional tensors.
Linear algebra books are concerned with distinguishing
between column vectors x and row vectors x⊤, and we will
try to adhere to this convention as much as possible. In
code this is trickier, because row and column vectors
correspond to 2-dimensional tensors of shape (1, d) or
(d, 1), which are different from 1-dimensional tensors of
shape (d). This is important to keep in mind because most
frameworks implement broadcasting rules2 inspired by
NumPy, giving rise to non-intuitive behaviors. See Box
C.2.1 for an example of a very common error arising in

2In a nutshell, broadcasting aligns the tensors’ shape from the right,
and repeats a tensor whenever possible to match the two shapes:
https://numpy.org/doc/stable/user/basics.broadcasting.html.

19


20

Linear algebra

import torch
x = torch.randn((4, 1)) # "Column"
y = torch.randn((4,))
print((x + y).shape)
# [Out]: (4,4) (because of broadcasting!)

# 1D tensor

Box C.2.1: An example of (probably incorrect) broadcasting,
resulting in a matrix output from an elementwise operation on two
vectors due to their shapes. The same result can be obtained in
practically any framework (NumPy, TensorFlow, JAX, ...).

implicit broadcasting of tensors’ shapes.

Vectors possess their own algebra (which we call a vector
space), in the sense that any two vectors x and y of the
same shape can be linearly combined z = ax+ by to provide
a third vector:

zi

= ax i

+ b yi

If we understand a vector as a point in d-dimensional
Euclidean space, the sum is interpreted by forming a
parallelogram, while the distance of a vector from the
origin is given by the Euclidean (ℓ

2) norm:

∥x∥ =

(cid:118)
(cid:116)(cid:88)

x 2
i

i

The squared norm ∥x∥2 is of particular interest, as it
corresponds to the sum of the elements squared. The
fundamental vector operation we are interested in is the
inner product (or dot product), which is given by
multiplying the two vectors element-wise, and summing
the resulting values.

20


Chapter 2: Mathematical preliminaries

21

Definition D.2.2 (Inner product)

The inner product between two vectors x, y ∼ (d) is given
by the expression:

〈x, y〉 = x⊤y = (cid:88)

x i yi

i

(E.2.1)

The notation 〈•, •〉 is common in physics, and we use it
sometimes for clarity. Importantly, the dot product between
two vectors is a scalar. For example, if x = [0.1, 0, −0.3]
and y = [−4.0, 0.05, 0.1]:

〈x, y〉 = −0.4 + 0 − 0.03 = −0.43

A simple geometric interpretation of the dot product is
given by its relation with the angle α between the two
vectors:

x⊤y = ∥x∥∥y∥ cos(α)

(E.2.2)

Hence, for two normalized vectors such that ∥•∥ = 1, the
dot product is equivalent to the cosine of their angle, in
which case we call the dot product the cosine similarity.
The cosine similarity cos(α) oscillates between 1 (two
vectors pointing in the same direction) and −1 (two
vectors pointing in opposite directions), with the special
case of 〈x, y〉 = 0 giving rise to orthogonal vectors
pointing in perpendicular directions. Looking at this from
another direction, for two normalized vectors (having
unitary norm), once we fix the first argument x we get:

y∗ = arg max 〈x, y〉 = x

(E.2.3)

21


22

Linear algebra

where arg max denotes the operation of finding the value
of y corresponding to the highest possible output value.
From (E.2.3) we see that, to maximize the dot product, the
second vector must equal the first one. This is important,
because in the following chapters x will represent an input,
while w will represent (adaptable) parameters, so that the
dot product is maximized whenever x ‘resonates’ with w
(template matching).

We close with two additional observations that will be
useful. First, we can write the sum of the elements of a
vector as its dot product with a vector 1 composed entirely
of ones, 1 = [1, 1, . . . , 1]⊤

:

〈x, 1〉 =

d
(cid:88)

i=1

x i

Second, the distance between two vectors can also be
written in terms of their dot products:

∥x − y∥2 = 〈x, x〉 + 〈y, y〉 − 2〈x, y〉

The case y = 0 gives us ∥x∥2 = 〈x, x〉. Both equations can
be useful when writing equations or in the code.

2.1.2 Common matrix operations

In the 2-dimensional case we have matrices:

X =





X 11
...
X n1

· · · X 1d
...
...
· · · X nd


 ∼ (n, d)

22


Chapter 2: Mathematical preliminaries

23

In this case we can talk about a matrix with n rows and d
columns. Of particular importance for the following, a
matrix can be understood as a stack of n vectors
(x1, x2, . . . , xn
), where the stack is organized in a row-wise
fashion:

X =









x⊤
1
...
x⊤
n

We say that X represents a batch of data vectors. As we will
see, it is customary to define models (both mathematically
and in code) to work on batched data of this kind. A
fundamental operation for matrices is multiplication:

Definition D.2.3 (Matrix multiplication)

For any two matrices X ∼(a,b) and Y ∼ (b, c) of
compatible shape, matrix multiplication Z = XY, with
Z ∼ (a, c) is defined element-wise as:

Zi j

= 〈Xi, Y⊤

j

〉

(E.2.4)

i.e., the element (i, j) of the product is the dot product
between the i-th row of X and the j-th column of Y.

As a special case, if the second term is a vector we have a
matrix-vector product:

z = Wx

(E.2.5)

If we interpret X as a batch of vectors, matrix
multiplication XW⊤ is a simple vectorized way of

23


24

Linear algebra

computing n dot products as in (E.2.5), one for each row
of X, with a single linear algebra operation. As another
example, matrix multiplication of a matrix by its
transpose, XX⊤ ∼ (n, n), is a vectorized way to compute all
possible dot products of pairs of rows of X simultaneously.

We close by mentioning a few additional operations on
matrices that will be important.

Definition D.2.4 (Hadamard multiplication)

For two matrices of the same shape, the Hadamard
multiplication is defined element-wise as:

[X ⊙ Y]

= X i j Yi j

i j

While Hadamard multiplication does not have all the
standard matrix
interesting algebraic properties of
multiplication,
it is commonly used in differentiable
models for performing masking operations (e.g., setting
operations.
some
Multiplicative interactions have also become popular in
some recent families of models, as we will see next.

elements

scaling

zero)

or

to

Sometimes we write expressions such as exp(X), which
are to be interpreted as element-wise applications of the
operation:

[ exp(X)]

= exp(X i j

)

i j

(E.2.6)

By comparison, “true” matrix exponentiation is defined for
a squared matrix as:

24


Chapter 2: Mathematical preliminaries

25

X = torch.randn((5, 5))
# Element-wise exponential
X = torch.exp(X)
# Matrix exponential
X = torch.linalg.matrix_exp(X)

Box C.2.2: Difference between the element-wise exponential of a
matrix and the matrix exponential as defined in linear algebra
textbooks. Specialized linear algebra operations are generally
encapsulated in their own sub-package.

mat-exp(X) =

∞
(cid:88)

k=0

1
k!

Xk

(E.2.7)

Importantly, (E.2.6) can be defined for tensors of any shape,
while (E.2.7) is only valid for (squared) matrices. This is
why all frameworks, like PyTorch, have specialized modules
that collect all matrix-specific operations, such as inverses
and determinants. See Box C.2.2 for an example.

Finally, we can write reduction operations (sum, mean, ...)
across axes without specifying lower and upper indices, in
which case we assume that the summation runs along the
full axis:

(cid:88)

=

Xi

i

n
(cid:88)

i=1

Xi

In PyTorch and other frameworks, reduction operations
correspond to methods having an axis argument:

r = X.sum(axis=1)

25


26

Linear algebra

On the definition of matrix multiplication

+ βx2

) = α f (x1

Why is matrix multiplication defined as (E.2.4) and not
as Hadamard multiplication? Consider a vector x and
some generic function f defined on it. The function is
said to be linear if f (αx1
).
) + β f (x2
Any such function can be represented as a matrix A
(this can be seen by extending the two vectors in a
basis representation). Then, the matrix-vector product
f (x) = Ax,
Ax corresponds to function application,
and matrix multiplication AB corresponds to function
composition f ◦ g, where ( f ◦ g)(x) = f (g(x)) and
g(x) = Bx.

Computational complexity

I will use matrix multiplication to introduce the topic of
complexity of an operation. Looking at (E.2.4), we see that
computing the matrix Z ∼ (a, c) from the input arguments
X ∼ (a, b) and Y ∼ (b, c) requires ac inner products of
dimension b if we directly apply the definition (what we
call the time complexity), while the memory requirement
for a sequential implementation is simply the size of the
output matrix (what we call instead the space complexity).

To abstract away from the specific hardware details,
computer science focuses on the so-called big-(cid:79) notation,
from the German ordnung (which stands for order of
approximation). A function f (x) is said to be (cid:79) (g(x)),
where we assume both inputs and outputs are
non-negative, if we can find a constant c and a value x0
such that:

f (x) ≤ c g(x) for any x ≥ x0

(E.2.8)

26


Chapter 2: Mathematical preliminaries

27

meaning that as soon as x grows sufficiently large, we can
ignore all factors in our analysis outside of g(x). This is
called an asymptotic analysis. Hence, we can say that a
naive implementation of matrix multiplication is (cid:79) (a bc),
growing linearly with respect to all three input parameters.
For two square matrices of size (n, n) we say matrix
multiplication is cubic in the input dimension.

Reasoning in terms of asymptotic complexity is important
(and elegant), but choosing an algorithm only in terms of
big-(cid:79) complexity does not necessarily translate to
practical performance gains, which depends on many
details such as what hardware is used, what parallelism is
supported, and so on.3 As an example, it is known that the
best asymptotic algorithm for multiplying two square
matrices of size (n, n) scales as (cid:79) (nc) for a constant
c < 2.4 [CW82], which is much better than the cubic
(cid:79) (n3) requirement of a naive implementation. However,
these algorithms are much harder to parallelize efficiently
on highly-parallel hardware such as GPUs, making them
uncommon in practice.

Note that from the point of view of asymptotic complexity,
having access to a parallel environment with k processors
has no impact, since it can only provide (at best) a constant
1
k speedup over a non-parallel implementation. In addition,
asymptotic complexity does not take into consideration the
time it takes to move data from one location to the other,

3When you call a specific primitive in a linear algebra framework,
such as matrix multiplication A @ B in PyTorch, the specific low-
level implementation that is executed (the kernel) depends on the
run-time hardware, through a process known as dispatching. Hence,
the same code can run via a GPU kernel, a CPU kernel, a TPU kernel,
etc. This is made even more complex by compilers such as XLA (https:
//openxla.org/xla), which can optimize code by fusing and optimizing
operations with a specific target hardware in mind.

27


28

Linear algebra

which can become the major bottleneck in many situations.4
In these cases, we say the implementation is memory-bound
as opposed to compute-bound. Practically, this can only be
checked by running a profiler over the code. We will see
that analyzing the complexity of an algorithm is far from
trivial due to the interplay of asymptotic complexity and
observed complexity.

2.1.3 Higher-order tensor operations

Vectors and matrices are interesting because they allow us
to define a large number of operations which are undefined
or complex in higher dimensions (e.g., matrix exponentials,
matrix multiplication, determinants, ...). When moving to
higher dimensions, most of the operations we are interested
into are either batched variants of matrix operations, or
specific combinations of matrix operations and reduction
operations.

As an example of the former, consider two tensors X ∼
(n, a, b) and Y ∼ (n, b, c). Batched matrix multiplication
(BMM) is defined as:

[BMM(X , Y )]

= XiYi

i

∼ (n, a, c)

(E.2.9)

Operations in most frameworks operate transparently on
batched versions of their arguments, which are assumed
like in this case to be leading dimensions (the first
dimensions). For example, batched matrix multiplication
in PyTorch is the same as standard matrix multiplication,
see Box C.2.3.

As an example of a reduction operation, consider two

4https://docs.nvidia.com/deeplearning/performance/dl-

performance-gpu-background/index.html

28


Chapter 2: Mathematical preliminaries

29

X = torch.randn((4, 5, 2))
Y = torch.randn((4, 2, 3))
(torch.matmul(X, Y)).shape # Or X @ Y
# [Out]: (4, 5, 3)

Box C.2.3: BMM in PyTorch is equivalent to standard matrix
multiplication. Most operations can work on (possibly) batched
inputs.

tensors X , Y ∼ (a, b, c). A generalized version of the dot
product (GDT) can be written as:

GDT(X , Y ) = (cid:88)

[X ⊙ Y ]

i jk

i, j,k

(E.2.10)

which is simply a dot product over the ‘flattened’ versions
of its inputs. This brief overview covers most of the tensor
operations we will use in the rest of the book, with
additional material introduced when necessary.

2.1.4 Einstein’s notation

This is an optional section that covers einsum,5 a set of
conventions that allows the user to specify practically
every tensor operation (including reductions,
sums,
multiplications) with a simple syntax based on text strings.

To introduce the notation, let us consider again the two
examples shown before in (E.2.9) and (E.2.10), writing
down explicitly all the axes:

5https://numpy.org/doc/stable/reference/generated/numpy.einsum.

html

29


30

Linear algebra

# Batched matrix multiply
M = torch.einsum('ijz,izk->ijk', A, B)
# Generalized dot product
M = torch.einsum('ijk,ijk->', A, B)

Box C.2.4: Examples of using einsum in PyTorch.

Mi jk
M = (cid:88)

= (cid:88)

Ai jz Bizk

(E.2.11)

(cid:88)

z
(cid:88)

X i jkYi jk

(E.2.12)

i

j

k

In line with Einstein’s notation,6 we can simplify the two
equations by removing the sums, under the convention
that any index appearing on the right but not on the left is
summed over:

Mi jk

= Ai jz Bizk

≜ (cid:88)

Ai jz Bizk

(E.2.13)

M = X i jkYi jk

z
(cid:88)

(cid:88)

≜ (cid:88)

i

j

k

X i jkYi jk

(E.2.14)

Then, we can condense the two definitions by isolating the
indices in a unique string (where the operands are now on
the left):

• ‘ijz,izk→ijk’ (batched matrix multiply);

• ‘ijk,ijk→’ (generalized dot product).

6The notation we use is a simplified version which ignores the
distinction between upper and lower indices: https://en.wikipedia.
org/wiki/Einstein_notation.

30


Chapter 2: Mathematical preliminaries

31

M = jax.numpy.einsum('ijz,izk->ijk', A, B)

Box C.2.5: Example of using einsum in JAX - compare with Box
C.2.4.

There is a direct one-to-one correspondence between the
definitions in (E.2.13)-(E.2.14) and their simplified string
definition. This is implemented in most frameworks in the
einsum operation, see Box C.2.4.

The advantage of this notation is that we do not need to
remember the API of a framework to implement a given
operation; and translating from one framework to the
other is transparent because the einsum syntax is
equivalent. For example, PyTorch has several matrix
multiplication methods, including matmul and bmm, with
different broadcasting rules and shape constraints, and
In
einsum provides a uniform syntax for all of them.
addition, the einsum definition of our batched matrix
multiplication is identical to, e.g., the definition in JAX,
see Box C.2.5.

Working with transposed axes is also simple. For example,
for A ∼ (n, a, b) and B ∼ (n, c, b), a batched multiplication
of [A]
is obtained by switching the
i
corresponding axes in the einsum definition:

times [B⊤]

i

M = torch.einsum('ijz,ikz->ijk', A, B)

Because of these reasons, einsum and its generalizations
(like the popular einops7 package) have gained a wide
popularity recently.

7http://einops.rocks

31


32

Gradients and Jacobians

2.2 Gradients and Jacobians

As the name differentiable implies, gradients play a pivotal
role in the book, providing a way to optimize our models
through semi-automatic mechanisms deriving from
gradient descent. We recall here some basic definitions
and concepts concerning differentiation of multi-valued
functions. We focus on properties that will be essential for
later, partially at the expense of mathematical precision.

2.2.1 Derivatives of scalar functions

Starting from a simple function y = f (x) with a scalar input
and a scalar output, its derivative is defined as follows.

Definition D.2.5 (Derivative) The derivative of f (x)
is defined as:

f ′(x) = lim
h→0

f (x + h) − f (x)
h

(E.2.15)

We use a variety of notation to denote derivatives: ∂ will
denote generically derivatives and gradients of any
∂
dimension (vectors, matrices); ∂
∂ x to highlight the
input argument we are differentiating with respect to
(when needed); while f ′(x) is specific to scalar functions
and it is sometimes called Lagrange’s notation.

x or

We are not concerned here about the existence of the
derivative of the function (which is not guaranteed
everywhere even for a continuous function), which we
assume as given. We will only touch upon this point when
discussing derivatives of non-smooth functions, such as
f (x) = |x| in 0 later on in Chapter 6.

32


Chapter 2: Mathematical preliminaries

33

Figure F.2.1: Plot
of the function
f (x) = x 2 − 1.5x,
shown along with the
derivatives on two
separate points.

Derivatives of simple functions can be obtained by direct
application of the definition, e.g., the derivatives of a
polynomial, logarithm, or sine should be familiar:

∂ x p = px p−1
∂ log(x) = 1
x
∂ sin(x) = cos(x)

Geometrically, the derivative can be understood as the slope
of the tangent passing through a point, or equivalently as
the best first-order approximation of the function itself in
that point, as shown in Figure F.2.1. This is a fundamental
point of view, because the slope of the line tells us how the
function is evolving in a close neighborhood: for a positive
slope, the function is increasing on the right and decreasing
on the left (again, for a sufficiently small interval), while
for a negative slope the opposite is true. As we will see,
this insight extends to vector-valued functions.

We recall some important properties of derivatives that
extend to the multi-dimensional case:

• Linearity: the derivative is linear, so the derivative

33

−4−20246810x020406080f(x)∂f(x)<0∂f(x)>0
34

Gradients and Jacobians

of a sum is the sum of derivatives:

∂ (cid:148)

f (x) + g(x)(cid:151) = f ′(x) + g ′(x) .

• Product rule:
∂ (cid:148)

f (x)g(x)(cid:151) = f ′(x)g(x) + f (x)g ′(x) ,

• Chain rule: the derivative of function composition is
given by multiplying the corresponding derivatives:

∂ (cid:148)

f (g(x))(cid:151) = f ′(g(x))g ′(x)

(E.2.16)

2.2.2 Gradients and directional derivatives

Consider now a function y = f (x) taking a vector x ∼ (d)
as input. Talking about infinitesimal perturbations here
does not make sense unless we specify the direction of this
perturbation (while in the scalar case we only had “left” and
“right”, in this case we have infinite possible directions in
the Euclidean space). In the simplest case, we can consider
moving along the i-th axis, keeping all other values fixed:

∂

xi

f (x) =

∂ y
∂ xi

= lim
h→0

f (x + hei
h

) − f (x)

,

(E.2.17)

∼ (d) is the i-th basis vector (the i-th row of the

where ei
identity matrix):

[ei

]

j

=

(cid:168)

1

if i = j
0 otherwise

(E.2.18)

(E.2.17) is called a partial derivative. Stacking all partial
derivatives together gives us a d-dimensional vector called

34


Chapter 2: Mathematical preliminaries

35

the gradient of the function.

Definition D.2.6 (Gradient)

The gradient of a function y = f (x) is given by:

∇ f (x) = ∂ f (x) =



∂

x1



∂

xd





f (x)
...
f (x)

(E.2.19)

Because gradients are fundamental, we use the special
notation ∇ f (x) to distinguish them. What about
displacements in a general direction v? In this case we
obtain the directional derivative:

Dv f (x) = lim
h→0

f (x + hv) − f (x)
h

,

(E.2.20)

Movement in space can be decomposed by considering
individual displacements along each axis, hence it is easy
to prove that the directional derivative is given by the dot
product of the gradient with the displacement vector v:

Dv f (x) = 〈∇ f (x), v〉 = (cid:88)

∂

xi

f (x)vi

(E.2.21)

Displacement on the i-th axis

i

Hence, knowing how to compute the gradient of a function
is enough to compute all possible directional derivatives.

2.2.3 Jacobians

Let us now consider the generic case of a function y = f (x)
with a vector input x ∼ (d) as before, and this time a vector
output y ∼ (o). As we will see, this is the most general
case we need to consider. Because we have more than one

35


36

Gradients and Jacobians

output, we compute a gradient for each output, and their
stack provides an (o, d) matrix called the Jacobian of f .

Definition D.2.7 (Jacobian) The Jacobian matrix of
a function y = f (x), x ∼ (d), y ∼ (o) is given by:

∂ f (x) =






∂ y1
∂ x1
...
∂ yo
∂ x1

. . .
...
. . .






∂ y1
∂ xd
...
∂ yo
∂ xd

∼ (o, d)

(E.2.22)

We recover the gradient for o = 1, and the standard
derivative for d = o = 1.
Jacobians inherit all the
properties of derivatives: importantly, the Jacobian of a
composition of functions is now a matrix multiplication of
the corresponding individual Jacobians:

∂ [ f (g(x))] = [∂ f (•)] ∂ g(x)

(E.2.23)

where the first derivative is evaluated in g(x) ∼ (h). See
[PP08, Chapter 2] for numerical examples of worked out
gradients and Jacobians. Like in the scalar case, gradients
and Jacobians can be understood as linear functions
tangent to a specific point. In particular, the gradient is
the best “first-order approximation” in the following sense.
For a point x0, the best linear approximation in an
infinitesimal neighborhood of f (x0

) is given by:

Slope of the line

(cid:101)f (x) = f (x0

) + 〈 ∂ f (x0

) , x − x0

〉

Displacement from x0

This is called Taylor’s theorem. See Box C.2.6 and Figure
F.2.2 for a visualization in the scalar case f (x) = x 2 − 1.5x.

36


Chapter 2: Mathematical preliminaries

37

# Generic function
f = lambda x: x**2-1.5*x

# Derivative (computed manually for now)
df = lambda x: 2*x-1.5

# Linearization at 0.5
x = 0.5
f_lin = lambda h: f(x) + df(x)*(h-x)

# Numerical check
print(f(x + 0.01))
# -0.5049
print(f_lin(x + 0.01)) # -0.5050

Box C.2.6: Example of computing a first-order approximation
(scalar case). The result is plotted in Figure F.2.2.

On the dimensionality of the Jacobians

We close with a pedantic note on dimensionality that will
be useful in the following. Consider the following function:

y = Wx

When viewed as a function of x, the derivative is, as before,
an (o, d) matrix, and it can be shown that:

∂
x

[Wx] = W

When viewed as a function of W, instead, the input is itself
an (o, d) matrix, and the “Jacobian” in this case has shape
(o, o, d) (see box in the following page). However, we can
always imagine an identical (isomorphic) function taking
as input the vectorized version of W, vect(W) ∼ (od), in
which case the Jacobian will be a matrix of shape (o, od).

37


38

Gradients and Jacobians

Figure F.2.2: The
function f (x) =
x 2 −1.5x and its first-
order approximation
shown in 0.5.

Working out the Jacobian

To compute the Jacobian ∂
expression element-wise as:

WWx, we can rewrite the

yi

= (cid:88)
j

Wi j x j

from which we immediately find that:

∂ yi
∂ Wi j

= x j

(E.2.24)

Note that to materialize the Jacobian explicitly (store it
in memory), we would need a lot of repeated values. As
we will see in Chapter 6, this can be avoided because,
in practice, we only care about the application of the
Jacobian on another tensor.

This quick example clarifies what we mean by our statement
that working with vector inputs and outputs is enough from
a notational point of view. However, it will be important
to keep this point in mind in Chapter 6, when we will use
matrix Jacobians for simplicity of notation (in particular,
to avoid the proliferation of indices), but the sizes of these
Jacobians may “hide” inside the actual shapes of the inputs

38

−1.0−0.50.00.51.0x−0.50.00.51.01.52.02.5f(x)f(x)Linearizedat0.5
Chapter 2: Mathematical preliminaries

39

and the outputs, most importantly the batch sizes. We will
see in Chapter 6 that explicit computation of Jacobians can
be avoided in practice by considering the so-called vector-
Jacobian products. This can also be formalized by viewing
Jacobians as abstract linear maps - see [BR24] for a formal
overview of this topic.

2.3 Gradient descent

To understand the usefulness of having access to gradients,
consider the problem of minimizing a generic function f (x),
with x ∼ (d):

x∗ = arg min

f (x)

(E.2.25)

x

where, similarly to arg max, arg min f (x) denotes the
operation of finding the value of x corresponding to the
lowest possible value of f (x). We assume the function has
a single output (single-objective optimization), and that
the domain over which we are optimizing x is
unconstrained.

In the rest of the book x will encode the parameters of our
model, and f will describe the performance of the model
itself on our data, a setup called supervised learning that
we introduce in the next chapter. We can consider
minimizing instead of maximizing with no loss of
generality,
to
maximizing − f (x) and vice versa (to visualize this, think
of a function in 1D and rotate it across the x-axis,
picturing what happens to its low points).

since minimizing f (x)

is equivalent

In very rare cases, we may be able to express the solution
in closed-form (we will see one example in the context of
least-squares optimization in Section 4.1.2). In general,

39


40

Gradient descent

however, we are forced to resort to iterative procedures.
Suppose we start from a random guess x0 and that, for every
iteration, we take a step, that we decompose in terms of its
magnitude η
t (the length of the step) and the direction pt:

Guess at iteration t

xt

= xt−1

+ η

tpt

(E.2.26)

Displacement at iteration t

t

the step size (or,

We call η
in machine learning
terminology, the learning rate, for reasons that will
become clear in the next chapter). A direction pt for
which there exists an η
) is called
t such that f (xt
a descent direction. If we can select a descent direction
for every iteration, and if we are careful in the choice of
step size, the iterative algorithm in (E.2.26) will converge
to a minimum in a sense to be described shortly.

) ≤ f (xt−1

For differentiable functions, we can precisely quantify all
descent directions by using the directional derivative from
(E.2.20), as they can be defined as the directions inducing
a negative change with respect to our previous guess xt−1:

pt is a descent direction ⇒ Dpt

f (xt−1

) ≤ 0

Using what we learned in Section 2.2 and the definition of
the dot product in terms of cosine similarity from (E.2.2)
we get:

Dpt

f (xt−1

) = 〈∇ f (xt−1

), pt

〉 = ∥∇ f (xt−1

)∥∥pt

∥ cos(α)

where α is the angle between pt and ∇ f (xt−1
). Considering
the expression on the right, the first term is a constant with
respect to pt. Because we have assumed pt only encodes
the direction of movement, we can also safely restrict it

40


Chapter 2: Mathematical preliminaries

41

∥ = 1, rendering the second term another constant.
to ∥pt
Hence, by the properties of the cosine we deduce that any
pt whose angle is between π/2 and 3π/2 with ∇ f (xt−1
)
=
is a descent direction. Among these, the direction pt
) (with an angle of π) has the lowest possible
−∇ f (xt−1
directional derivative, and we refer to it as the steepest
descent direction.

Putting together this insight with the iterative procedure
in (E.2.26) gives us an algorithm to minimize any
differentiable function, that we call (steepest) gradient
descent.

Definition D.2.8 (Steepest gradient descent)

Given a differentiable function f (x), a starting point x0,
and a step size sequence η
t, gradient descent proceeds
as:

xt

= xt−1

− η

∇ f (xt−1

)

t

(E.2.27)

We will not be concerned with the problem of finding an
appropriate step size, which we will just assume “small
enough” so that the gradient descent iteration provides a
reduction in f . In the next section we focus on what points
are obtained by running gradient descent from a generic
initialization. Note that gradient descent is as efficient
as the procedure we use to compute the gradient: we
introduce a general algorithm to this end in Chapter 6.

2.3.1 Convergence of gradient descent

When discussing the convergence of gradient descent, we
need to clarify what we mean by “a minimizer” of a function.
If you do not care about convergence and you trust gradient
descent, proceed with no hesitation to the next section.

41


42

Gradient descent

Definition D.2.9 (Minimum)

A local minimum of f (x) is a point x+ such that the
following is true for some ϵ > 0:

f (x+) ≤ f (x) ∀x : ∥x − x+∥ < ϵ

Ball of size ϵ centered in x+

In words, the value of f (x+) is a minimum if we consider
a sufficiently small neighborhood of x+. Intuitively, in such
a point the slope of the tangent will be 0, and the gradient
everywhere else in the neighborhood of x+ will point
upwards. We can formalize the first idea by the concept of
stationary points.

Definition D.2.10 (Stationary points)

A point x+ is called a stationary point of
∇ f (x+) = 0.

f (x) if

Stationary points are not limited to minima: they can be
maxima (the minima of − f (x)) or saddle points, which
are inflexion points where the curvature of the function
is changing (see Figure F.2.3 for an example). In general,
without any constraint on f , gradient descent can only be
proven to converge to a generic stationary point depending
on its initialization.

Can we do better? Picture a parabola: in this case, the
function does not have any saddle points, and it only has a
single minimum. This minimum is also special, in the sense
that the function in that point attains its lowest possible
value across the entire domain: we say this is a global
minimum.

42


Chapter 2: Mathematical preliminaries

43

Figure F.2.3:
Simple example
of a saddle point
(try visualizing the
tangent line in that
point to see it is
indeed stationary).

Definition D.2.11 (Global minimum)

A global minimum of f (x) is a point x∗ such that
f (x∗) ≤ f (x) for any possible input x.

Intuitively, gradient descent will converge to this global
minimum if run on a parabola (from any possible
initialization) because all gradients will point towards it.
We can generalize this idea with the concept of convexity
of a function. There are many possible definitions of
convexity, we choose the one below for simplicity of
exposition.

Definition D.2.12 (Convex function)

A function f (x) is convex if for any two points x1 and x2
and α ∈ [0, 1] we have:

Line segment from f (x1

) to f (x2

)

f ( αx1

+ (1 − α)x2

) ≤ α f (x1

) + (1 − α) f (x2

)

(E.2.28)

Interval from x1 to x2

43

xf(x)Saddlepoint(neitherminimumnormaximum)
44

Gradient descent

The left-hand side in (E.2.28) is the value of f on any
point inside the interval ranging from x1 to x2, while the
right-hand side is the corresponding value on a line
connecting f (x1
).
If the function is always
below the line joining any two points, it is convex (as an
example, a parabola pointing upwards is convex).

) and f (x2

Convexity qualifies the simplicity of optimizing the function,
in the following sense [JK+17]:

1. For a generic non-convex function, gradient descent
converges to a stationary point. Nothing more can
be said unless we look at higher-order derivatives
(derivatives of the derivatives).

2. For a convex function, gradient descent will converge
to a global minimum, irrespective of initialization.

3. If the inequality in (E.2.28) is satisfied in a strict way
(strict convexity), the global minimizer will also be
unique.

This is a hard property: to find a global minimum in a non-
convex problem with gradient descent, the only solution
is to run the optimizer infinite times from any possible
initialization, turning it into an NP-hard task [JK+17].

This discussion has a strong historical significance. As we
will see in Chapter 5, any non-trivial model is non-convex,
meaning that its optimization problem may have several
This is in contrast to alternative
stationary points.
algorithms for supervised learning, such as support vector
machines, which maintain non-linearity while allowing for
convex optimization. Interestingly, complex differentiable
models seem to work well even in the face of such
restriction, in the sense that their optimization, when

44


Chapter 2: Mathematical preliminaries

45

started from a reasonable initialization, converge to points
with good empirical performance.

2.3.2 Accelerating gradient descent

The negative gradient describes the direction of steepest
descent, but only in an infinitesimally small neighborhood
of the point. As we will see in Chapter 5 (where we
introduce stochastic optimization), these directions can be
extremely noisy, especially when dealing with large
models. A variety of techniques have been developed to
accelerate convergence of the optimization algorithm by
selecting better descent directions. For computational
reasons, we are especially interested in methods that do
not require higher-order derivatives (e.g., the Hessian), or
multiple calls to the function.

We describe here one such technique, momentum, and we
refer to [ZLLS23, Chapter 12], for a broader introduction.8
If you picture gradient descent as a ball “rolling down
a hill”, the movement is relatively erratic, because each
gradient can point in a completely different direction (in
fact, for a perfect choice of step size and a convex loss
function, any two gradients in subsequent iterations will be
orthogonal). We can smooth this behavior by introducing
a “momentum” term that conserves some direction from
the previous gradient iteration:

Steepest descent

Momentum term

gt

xt

= − η

= xt−1

t

∇ f (xt−1
+ gt

) + λgt−1

8See also this 2016 blog post by S. Ruder: https://www.ruder.io/

optimizing-gradient-descent/.

45


46

Gradient descent

Figure F.2.4: GD and GD with momentum when minimizing the
function x sin(2x) starting from x = 1 + ϵ, with λ = 0.3.

= 0. See Figure F.2.4 for an example.
where we initialize g0
The coefficient λ determines how much the previous term
is dampened. In fact, unrolling two terms:

gt

= −η

= −η

∇ f (xt−1
∇ f (xt−1

t

t

) + λ(−η

∇ f (xt−2

) + λgt−2

)

t

) − λη

∇ f (xt−2

) + λ2gt−2

t

Generalizing, the iteration at time t − n gets dampened by
a factor λn−1. Momentum can be shown to accelerate
training by smoothing the optimization path [SMDH13].
Another common technique is adapting the step size for
each parameter based on the gradients’ magnitude
[ZLLS23]. A common optimization algorithm combining
several of these ideas is Adam [KB15]. One advantage of
Adam is that it is found to be relatively robust to the
choice of its hyper-parameters,9 with the default choice

9A hyper-parameter is a parameter which is selected by the user, as

opposed to being learnt by gradient descent.

46

0.51.01.52.02.53.0x−2.5−2.0−1.5−1.0−0.50.00.51.0f(x)StandardGDMomentumGD
Chapter 2: Mathematical preliminaries

47

in most frameworks being a good starting point in the
majority of cases. Designing novel optimizers that can
‘unseat’ Adam (or its variations, such as AdamW [LH19])
from its place as de-facto default optimizer in deep
learning remains an open research problem, e.g., see
[BN24]
recent work on designing customized
optimizers for neural networks from first principles.

for

One disadvantage of using accelerated optimization
algorithms can be increased storage requirements:
for
example, momentum requires us to store the previous
gradient iteration in memory, doubling the space needed
by the optimization algorithm (although in most cases, the
memory required to compute the gradient is the most
influential factor in terms of memory, as we will see in
Section 6.3).

47


48

Gradient descent

From theory to practice

About the exercises

This book does not have classical end-of-chapter
exercises, which are covered in many existing textbooks.
Instead, I propose a self-learning path to help you explore
two frameworks (JAX and PyTorch) as you progress in
the book. Solutions to the exercises will be published
on the book’s website.a These sections are full of URLs
linking to online material – they might be expired or
moved by the time you search for them.

ahttps://www.sscardapane.it/alice-book

Starting from the basics

The starting block for any designer
of differentiable models is a careful
study of NumPy. NumPy implements
a generic set of functions to manipulate
multidimensional arrays (what we call
tensors in the book), as long as functions
to index and transform their content. You can read more on
the library’s quick start.10 You should feel comfortable in
handling arrays in NumPy, most notably for their indexing:
the rougier/numpy-10011 repository provides a nice,
slow-paced series of exercises to test your knowledge.

10https://numpy.org/doc/stable/user/quickstart.html
11https://github.com/rougier/numpy-100

48


Chapter 2: Mathematical preliminaries

49

Moving to a realistic framework

Despite its influence, NumPy is limited in his support for
parallel hardware such as GPUs (unless additional libraries
are used), and for his lack of automatic differentiation
(introduced in Chapter 6). JAX replicates the NumPy’s
interface while adding extended hardware support, the
automatic computation of gradients, and additional
transformations such as the vectorized map (jax.vmap).
Frameworks such as PyTorch also implement a NumPy-like
interface at their core, but they make minor adjustments
in nomenclature and functionality and they add high-level
utilities for building differentiable models. Take your time
to skim the documentation of jax.numpy.array and
torch.tensor to understand how much they have in
common with NumPy. For now, you can ignore high-level
modules such as torch.nn. We will have more to say
about how these frameworks are designed in Chapter 6,
after we introduce their gradient computation mechanism.

Implementing a gradient descent algorithm

To become proficient with all three frameworks (NumPy,
JAX, PyTorch), I suggest to replicate the exercise below
thrice – each variant should only take a few minutes if you
know the syntax. Consider a 2D function f (x), x ∼ (2),
where we take the domain to be [0, 10]:12

f (x) = sin(x1

) cos(x2

) + sin(0.5x1

) cos(0.5x2

)

Before proceeding in the book, repeat this for each
framework:

12I asked ChatGPT to generate a nice function with several minima
and maxima. Nothing else in the book is LLM-generated, which I feel
is becoming an important disclaimer to make.

49


50

Gradient descent

1. Implement the function in a vectorized way, i.e.,
given a matrix X ∼ (n, 2) of n inputs, it should return
a vector f (X) ∼ (n) where [ f (X)]

).

= f (Xi

i

2. Implement another function to compute its gradient
(hard-coded – we have not touched automatic
differentiation yet).

3. Write a basic gradient descent procedure and
visualize the paths taken by the optimization process
from multiple starting points.

4. Try adding a momentum term and visualizing the
norm of the gradients, which should converge to
zero as the algorithm moves towards a stationary
point.

If you are using JAX or PyTorch to solve the exercise, point
(3) is a good place to experiment with vmap for vectorizing
a function.

50
