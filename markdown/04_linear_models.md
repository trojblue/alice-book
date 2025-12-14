4 | Linear models

About this chapter

Programming is done by choosing the appropriate
sequence of primitive operations to solve a task. By
analogy, building a model is done by choosing the
correct sequence of differentiable blocks. In this chapter
we introduce the simplest block, linear models, which
assume that inputs act additively on the output via a
weighted average. In a sense, all differentiable models
are smart variations and compositions of linear blocks.

4.1 Least-squares regression

Summarizing the previous chapter, a supervised learning
problem can be defined by choosing the input type x, the
output type y, the model f , and the loss function l. In this
chapter we consider the simplest possible choices for all of
them, namely:

• The input is a vector x ∼ (c), corresponding to a set
of features (e.g., c personal features of a client of a
bank). We use the scalar c (short for channels) to

71


72

Least-squares regression

Table T.4.1: Basic shapes to remember for this chapter. For
uniformity, we will use the same letters as much as possible
throughout the book.

size of the dataset
features

n
c
m classes

denote the number of features to be consistent with
the following chapters.

• The output is a single real value y ∈ (cid:82).

In the
unconstrained case, we say this is a regression task.
If y can only take one out of m possible values, i.e.,
y ∈ {1, . . . , m}, we say this is a classification task.
In the special case of m = 2, we say this is a binary
classification task.

• We take f to be a linear model, providing us with
simple closed form solutions in some cases, most
notably least-squares regression (Section 4.1.2).

The basic shapes to remember are summarized in Table
T.4.1. We begin by discussing the choice of loss in the
regression case. We start from the regression case since, as
we show later, classification can be solved by small
modifications to the regression case.

4.1.1 The squared loss and variants

Finding a loss for regression is relatively simple, since the
prediction error e = ( ˆy − y) between the predicted output
of the model ˆy = f (x) and the true desired output y
is a well-defined target, being a continuous function of
the model’s output that decreases monotonically. Since in

72


Chapter 4: Linear models

73

general we do not care about the sign of the prediction
error, a common choice is the squared loss:

l( ˆy, y) = ( ˆy − y)2

(E.4.1)

Here and in the following we use the symbol ˆy to denote the
prediction of a model. As we will see, working with (E.4.1)
grants several benefits to our solution. Among others, the
gradient of the squared loss is a linear function of the
model’s output, allowing us to solve it in closed form for
the optimal solution.

Recalling the maximum likelihood principle (Section
3.2.2), the squared loss can be obtained by assuming that
the outputs of the model follow a Gaussian distribution
centered in f (x) and with a constant variance σ2:

p( y | f (x)) = (cid:78) ( y | f (x), σ2)

In this case the log-likelihood (for a single point) can be
written as:1

log(p( y | f (x), σ2)) =

− log(σ) − 1
2

log(2π) − 1
2σ2

( y − f (x))2

(E.4.2)

Minimizing (E.4.2) for f , we see that the first two terms on
the right-hand side are constant, and the third reverts to the
squared loss. Minimizing for σ2 can be done independently
from the optimization of f , with a simple closed form
solution (see below, equation (E.4.9)).

1Recalling that log(ab) = log(a) + log(b) and log(a b) = b log(a).

73


74

Least-squares regression

Figure F.4.1: Visualization of the squared loss, the absolute loss,
and the Huber loss with respect to the prediction error e = ( ˆy − y).

Coming up with variants to the squared loss is also easy.
For example, one drawback of the squared loss is that
higher errors will be penalized with a strength that grows
quadratically in the error, which may provide undue
influence to outliers, i.e., points that are badly mislabeled.
Other choices that diminish the influence of outliers can be
the absolute value loss l( ˆy, y) = | ˆy − y| or the Huber loss
(a combination of the squared loss and the absolute loss):

L( y, ˆy) =

(cid:168) 1
2

( y − ˆy)2
(cid:0)| y − ˆy| − 1

if | y − ˆy| ≤ 1
otherwise

(cid:1)

(E.4.3)

2

which is quadratic in the promixity of 0 error, and linear
otherwise (with the − 1
2 term added to ensure continuity).
See Figure F.4.1 for a visualization of these losses with
respect to the prediction error.

The absolute loss seems an invalid choice in our context,
since it has a point of non-differentiability in 0 due to
the absolute value. We will see later that functions with

74

−2−1012e01234L(e)SquaredlossAbsolutelossHuberloss(δ=1.5)
Chapter 4: Linear models

75

one (or a small number) of points of this form are not
truly problematic. Mathematically, they can be handled
by the notion of subgradient (a slight generalization of
the derivative). Practically, you can imagine that if we
start from a random initialization, gradient descent will
never reach these points with perfect precision, and the
derivatives of |ϵ| for any ϵ > 0 is always defined.

4.1.2 The least-squares model

With a loss function in hand, we consider the following
model (a linear model) to complete the specification of our
first supervised learning problem.

Definition D.4.1 (Linear models)

A linear model on an input x is defined as:

f (x) = w⊤x + b

where w ∼ (c) and b ∈ (cid:82) (the bias) are trainable
parameters.

The intuition is that the model assigns a fixed weight wi to
each input feature x i, and provides a prediction by linearly
summing all the effects for a given input x, reverting to a
to b whenever x = 0.
default prediction equal
Geometrically, the model defines a line for d = 1, a plane
for d = 2, and a generic hyperplane for d > 1. From a
notational perspective, we can sometimes avoid writing a
bias term by assuming a constant term of 1 as the last
feature of x:

(cid:152)(cid:139)

(cid:129)(cid:149)x
1

(cid:152)

= w⊤ (cid:149)x
1

f

= w⊤

1:cx + wc+1

75


76

Least-squares regression

Combining the linear model, the squared loss, and an
empirical risk minimization problem we obtain the
least-squares optimization problem.

Definition D.4.2 (Least-squares)

The least-squares optimization problem is given by:

w∗, b∗ = arg min

w,b

1
n

n
(cid:88)

i=1

(cid:0) yi

− w⊤xi

− b(cid:1)2

(E.4.4)

Before proceeding to the analysis of this problem, we
rewrite the least-squares in a vectorized form that only
involves matrix operations (matrix products and norms).
This is useful because, as already stated, modern code for
training
around
n-dimensional arrays, with optimized hardware to
perform matrix operations on them. To this end, we first
stack all the inputs and outputs of our training set into an
input matrix:

differentiable models

built

is


 ∼ (n, c)

X =





x⊤
1
...
x⊤
n

and a similar output vector y = [ y1, . . . , yn
. We can
write a batched model output (the model output for a mini-
batch of values) as:

]⊤

f (X) = Xw + 1b

(E.4.5)

Same bias b for all n predictions

76


Chapter 4: Linear models

77

def linear_model(w: Float[Tensor, "c"],

b: Float,
X: Float[Tensor, "n c"])
-> Float[Tensor, "n"]:

return X @ w + b

Box C.4.1: Computing a batched linear model as in (E.4.5). For
clarity, we are showing the array dimensions as type hints using
jaxtyping (https://docs.kidger.site/jaxtyping/).

Equations like (E.4.5) can be replicated almost line-by-line
in code - see Box C.4.1 for an example in PyTorch.

Of only marginal interest for now but of more importance
for later, we note that the row ordering of the input matrix
and of the output vector are fundamentally arbitrary, in
the sense that permuting their rows will only result in a
corresponding permutation of the rows of f (X). This is
a simple example of a phenomenon called permutation
equivariance that will play a much more important role
later on.

The least-squares optimization problem written in a
vectorized form becomes:

LS(w, b) = 1
n

∥y − Xw − 1b∥2

(E.4.6)

where we recall that the norm of a vector is defined as
∥e∥2 = (cid:80)

i e2
i .

4.1.3 Solving the least-squares problem

To solve the least-squares problem through gradient
descent, we need the equation for its gradient. Although

77


78

Least-squares regression

from torch import linalg
def ls_solve(X: Float[Tensor, "n c"],

y: Float[Tensor, "n"],
numerically_stable = True) \
-> Float[Tensor, "c"]:

# Explicit solution
if not numerically_stable:

return linalg.inv(X.T @ X) @ X.T @ y

else:

return linalg.solve(X.T @ X, X.T @ y)

Box C.4.2: Solving the least-squares problem with the closed form
solution. The numerically stable variant calls a solver specialized
for systems of linear equations.

we will soon develop a general algorithmic framework to
compute these gradients automatically (Chapter 6), it is
instructive to look at the gradient itself in this simple
scenario. Ignoring the bias (for the reasons stated above,
we can incorporate it in the weight vector), and other
constant terms we have:

∇LS(w) = X⊤ (Xw − y)

The LS problem is convex in the weights of the model, as
can be understood informally by noting that the equations
describe a paraboloid in the space of the weights (a
quadratic function). The global minima are then described
by the equations:

X⊤ (Xw − y) = 0 ⇒ X⊤Xw = X⊤y

These are called the normal equations. Importantly, the
normal equations describe a linear system of equations in

78


Chapter 4: Linear models

79

w,2 meaning that under the appropriate conditions
(corresponding to the invertibility of the matrix X⊤X) we
can solve for the optimal solution as:

w∗ = (cid:0)X⊤X(cid:1)−1

X⊤y

(E.4.7)

Tidbits of information

X⊤ is

The matrix X† = (cid:0)X⊤X(cid:1)−1
called the
pseudoinverse (or Moore-Penrose inverse) of
the non-square matrix X, since X†X = I. Performing the
inversion in (E.4.7) is not always possible: for example,
if one feature is a scalar multiple of the other, the matrix
X does not have full rank (this is called collinearity).
Finally, note that the predictions of the least-squares
model can be written as ˆy = My, with M = XX†. Hence,
least-squares can also be interpreted as performing
a weighted average of the training labels, where the
weights are given by a projection on the column space
induced by X. This is called the dual formulation of
least-squares. Dual formulations provide an intrinsic
level of debugging of the model, as they allow to check
which inputs were the most relevant for a prediction by
checking the corresponding dual weights [ICS22].

This is the only case in which we will be able to express
the optimal solution in a closed form way, and it is
instructive to compare this solution with the gradient
descent one. To this end, we show in Box C.4.2 an
example of solving the least-squares in closed form using
(E.4.7), and in Box C.4.3 the equivalent gradient descent
formulation. A prototypical evolution of the loss in the

2That is, we can write them as Aw = b, with A = X⊤X and b = X⊤y.

79


80

Least-squares regression

def ls_gd(X: Float[Tensor, "n c"],
y: Float[Tensor, "n 1"],
lr=1e-3) \
-> Float[Tensor, "c"]:

# Initializing the parameters
w = torch.randn((X.shape[1], 1))

# Fixed number of iterations
for i in range(15000):

# Note the sign (why?)
w = w + lr * X.T @ (y - X @ w)

return w

Box C.4.3: Same task as Box C.4.2, solved with a naive
implementation of gradient descent with a fixed learning rate that
defaults to η = 0.001.

latter case is plotted in Figure F.4.2. Since we selected a
very small learning rate, each step in the gradient descent
procedure provides a stable decrease in the loss, until
convergence. Practically, convergence could be checked by
numerical means, e.g., by evaluating the difference in
some numerical
norm between two iterations
threshold ϵ > 0:

for

∥wt+1

− wt

∥2 < ϵ

(E.4.8)

As we will see, understanding when more complex models
have converged will be a more subtle task.

Considering again the Gaussian log-likelihood in (E.4.2),
we can also optimize the term with respect to σ2 once the
weights have been trained, obtaining:

80


Chapter 4: Linear models

81

Figure F.4.2: An example
of running code from Box
C.4.2, where the data is
composed of n = 10 points
drawn from a linear model
w⊤x+ϵ, with wi
∼ (cid:78) (0, 1)
and ϵ ∼ (cid:78) (0, 0.01).
Details apart, note the
very smooth descent: each
step provides a decrease in
loss.

σ2
∗

= 1
n

n
(cid:88)

( yi

i=1

− w⊤

∗ xi

)2 .

(E.4.9)

which has the intuitive meaning that the variance of the
model is constant (by definition) and given by the average
squared prediction error on our training data. More
sophisticated probabilistic models can be obtained by
assuming the variance itself is predicted by the model
(heteroscedastic models), see [Bis06].

4.1.4 Some computational considerations

Even if the inverse can be computed, the quality of the
solution will depend on the condition number of X⊤X, and
large numerical errors can occur for poorly conditioned
matrices.3 In addition, the computational cost of solving
(E.4.7) may be prohibitive. The matrix inversion will scale,
roughly, as (cid:79) (c3). As for the matrix multiplications, the

3The condition number of a matrix A is defined as κ(A) = ∥A∥∥A−1∥
for some choice of matrix norm ∥•∥. Large conditions number can
make the inversion difficult, especially if the floating-point precision is
not high.

81

02004006008001000Iteration01234567Loss
82

Least-squares regression

algorithm requires a multiplication of a c × n matrix with
another n × c one, and a multiplication between a c × c
matrix and a c × n one. Both these operations will scale as
(cid:79) (c2n).

In general, we will always prefer algorithms that scale
linearly both in the feature dimension c and in the batch
size n, since super-linear algorithms will become quickly
impractical (e.g., a batch of 32 RGB images of size
1024 × 1024 has c ≈ 1e7). We can avoid a quadratic
complexity in the equation of the gradient by computing
the multiplications in the correct order, i.e., computing the
matrix-vector product Xw first. Hence, pure gradient
descent is linear in both c and n, but only if proper care is
taken in the implementation: generalizing this idea is the
fundamental insight for the development of reverse-mode
automatic differentiation, a.k.a.
back-propagation
(Section 6.3).

4.1.5 Regularizing the least-squares solution

Looking again at the potential instability of the inversion
operation, suppose we have a dataset for which the matrix
is almost singular, but we still wish to proceed with the
closed form solution. In that case, it is possible to slightly
modify the problem to achieve a solution which is “as close
as possible” to the original one, while being feasible to
compute. For example, a known trick is to add a small
multiple, λ > 0, of the identity matrix to the matrix being
inverted:

w∗ = (cid:0)X⊤X+λI(cid:1)−1

X⊤y

This pushes the matrix to be “more diagonal” and
improves its condition number.
Backtracking to the
original problem, we note this is the closed form solution

82


Chapter 4: Linear models

83

of a modified optimization problem:

LS-Reg(w) = 2
n

∥y − Xw∥2 +

λ

2

∥w∥2

This problem is called regularized least-squares (or ridge
regression), and the red part in the loss is an instance of
ℓ
2-regularization (or, more generally, regularization). Note
that regularization does not depend on the dataset, as it
simply encodes a preference for a certain type of solution
(in this case, low-norm weights), where the strength of
the preference itself is defined by the hyper-parameter λ.
From a Bayesian perspective (Section 3.3), the regularized
least-squares corresponds to a MAP solution when defining
a Gaussian prior over the weights centered in zero with
constant variance.

4.2 Linear models for classification

∈ {1, . . . , m},
We now move to classification, in which yi
where m defines the number of classes. As we will see
later, this is a widely influential problem, encompassing a
range of tasks in both computer vision (e.g.,
image
classification) and natural
language processing (e.g.,
next-token prediction). We can tackle this problem by
slight variations with respect to the regression case.

While we can solve the task by regressing directly on the
integer value yi, it is instructive to consider why this might
not be a good idea. First, it is difficult for a model to
directly predict an integer value, since this requires some
thresholding that would render its gradient zero almost
Instead, we could regress on a real value
everywhere.
∈ [1, m] inside the interval from 1 to m (as we will show,
(cid:101)yi
bounding the output of the model inside an interval can be

83


84

Linear models for classification

done easily). During inference, given the output ˆyi
we map back to the original domain by rounding:

= f (xi

),

Predicted class = round( ˆyi

)

= 1.3 would be mapped to class 1, while
For example, ˆyi
= 3.7 would be mapped to class 4. Note that this is a
ˆyi
post-hoc processing of the values that is only feasible at
inference time. The reason this is not a good modeling
choice is that we are introducing a spurious ordering of
the classes which might be exploited by the model itself,
where class 2 is “closer” to class 3 than it is to class 4. We
can avoid this by moving to a classical one-hot encoded
version of y, which we denote by yoh ∼ Binary(m):

[yoh]

j

=

(cid:168)

1

0

if y = j
otherwise

for class 1, yoh = [0 1 0]⊤

For example, in the case of three classes, we would have
yoh = [1 0 0]⊤
for class 2,
and yoh = [0 0 1]⊤
for class 3 (this representation should
be familiar to readers with some background in machine
learning, as it is a standard representation for categorical
variables).

(cid:112)

1 and yoh

One-hot vectors are unordered, in the sense that given two
generic outputs yoh
2 , their Euclidean distance is
either 0 (same class) or
2 (different classes). While we
can perform a multi-valued regression directly on the one-
hot encoded outputs, with the mean-squared error known
as the Brier score in this case, we show below that a better
and more elegant solution exists, in the form of logistic
regression.

84


Chapter 4: Linear models

85

4.2.1 The softmax function

We cannot train a model to directly predict a one-hot
encoded vector (for the same reasons described above),
but we can achieve something similar by a slight
relaxation. To this end, we re-introduce the probability
simplex.

Definition D.4.3 (Probability simplex)

The probability simplex ∆
such that:

n is the set of vectors x ∼ ∆(n)

≥ 0,

x i

(cid:88)

i

= 1

xi

Geometrically, you can picture the set of one-hot vectors as
the vertices of an n-dimensional polytope, and the simplex
as its convex hull: values inside the simplex, such as
[0.2, 0.05, 0.75], do not precisely correspond to a vertex,
but they allow for gradient descent because we can
smoothly move inside the polytope. Given a value x ∈ ∆
n,
we can project to its closest vertex (the predicted class) as:

arg max
i

{xi

}

As the name implies, we can interpret values inside the
simplex as probability distributions, and projection on the
closest vertex as finding the mode (the most probable class)
in the distribution. In this interpretation, a one-hot encoded
vector is a “special case” where all the probability mass is
concentrated on a single class (which we know to be the
correct one).

In order to predict a value in this simplex, we need two

85


86

Linear models for classification

modifications to the linear model from (E.4.4): first, we
need to predict an entire vector simultaneously; and second,
we need to constrain the outputs to lie in the simplex. As
a first step, we modify the linear model to predict an m-
dimensional vector:

y = Wx + b

(E.4.10)

where W ∼ (m, c) can be interpreted as m linear
regression models running in parallel, and b ∼ (m). This
output is unconstrained and it is not guaranteed to be in
the simplex. The core idea of logistic regression is to
combine the linear model in (E.4.10) with a simple,
parameter-free transformation that projects inside the
simplex, called the softmax function.

Definition D.4.4 (Softmax function)

The softmax function is defined for a generic vector x ∼
(m) as:

[softmax(x)]

=

i

exp(x i
)
j exp(x j

(cid:80)

)

(E.4.11)

To understand what is happening, we decompose the terms
in (E.4.11) by introducing two intermediate terms. First,
the numerator of the softmax converts each number to a
positive value hi by exponentiation:

hi

= exp(xi

)

(E.4.12)

Second, we compute a normalization factor Z as the sum
of these new (non-negative) values:

86


Chapter 4: Linear models

87

Z = (cid:88)

h j

j

(E.4.13)

The output of the softmax is then given by dividing hi by
Z, thus ensuring that the new values sum to 1:

=

yi

hi
Z

(E.4.14)

Another perspective comes from considering a more general
version of the softmax, where we add an additional hyper-
parameter τ > 0 called the temperature:

softmax(x; τ) = softmax(x/τ)

The softmax keeps the relative ordering among the values of
xi for all values of τ, but their absolute distance is increased
or decreased based on the temperature. In particular, we
have the following two limiting cases:

lim
τ→∞

softmax(x; τ) = 1/c

softmax(x; τ) = arg max

x

i

lim
τ→0

(E.4.15)

(E.4.16)

For infinite temperature, relative distances will disappear
and the output reverts to a uniform distribution. At the
contrary, at 0 temperature the softmax reverts to the
(poorly differentiable) argmax operation. Hence, softmax
can be seen as a simple differentiable approximation to
the argmax, and a better name should be softargmax.
However, we will retain the most standard name here. See

87


88

Linear models for classification

(a) Inputs

(b) τ = 1

(c) τ = 10

(d) τ = 100

Figure F.4.3: Example of softmax applied to a three-dimensional
vector (a), with temperature set to 1 (b), 10 (c), and 100 (d).
As the temperature increases, the output converges to a uniform
distribution. Note that inputs can be both positive or negative, but
the outputs of the softmax are always constrained in [0, 1].

Figure F.4.3 for a visualization of a softmax applied on a
generic
different
temperature values.

three-dimensional

vector with

4.2.2 The logistic regression model

We can summarize our previous discussion by combining
the softmax in (E.4.11) with the linear model in (E.4.10)
to obtain a linear model for classification:

ˆy = softmax (Wx + b)

The pre-normalized outputs h = Wx + b are called the
logits of the model, a name that will be discussed in more
detail in the next section.

We now need a loss function. Considering the probabilistic
viewpoint from Section 3.2.2, because our outputs are
restricted to the probability simplex, we use them as the
parameters of a categorical distribution:

88

−0.50.00.51.01.52.02.5−3−2−1012−0.50.00.51.01.52.02.50.00.20.40.60.81.0−0.50.00.51.01.52.02.50.00.20.40.60.81.0−0.50.00.51.01.52.02.50.00.20.40.60.81.0
Chapter 4: Linear models

89

Exponent is always either 0 or 1

p( yoh | ˆy) = (cid:89)

y oh
i

ˆy
i

i
One-hot encoded class

Computing the maximum likelihood solution in this case
(try it) gets us the cross-entropy loss.

Definition D.4.5 (Cross-entropy loss)

The cross-entropy loss function between yoh and ˆy is
given by:

CE(yoh, ˆy) = −(cid:88)

y oh
i

log( ˆyi

)

i

(E.4.17)

The loss can also be derived as the KL divergence between
the two probability distributions. While unintuitive at first,
it has a very simple interpretation by noting that only one
value of yoh will be non-zero, corresponding to the true
(cid:9). We can then simplify the loss as:
class y = arg max

(cid:8) y oh

i

i

CE( y, ˆy) = − log( ˆy y

)

(E.4.18)

Probability assigned to the true class

From (E.4.18), we see that the effect of minimizing the CE
loss is to maximize the output probability corresponding
to the true class. This works since, due to the denominator
in the softmax, any increase in one output term will
automatically lead to a decrease of the other terms.
Putting everything together, we obtain the logistic
regression optimization problem:

89


90

More on classification

LR(W, b) = 1
n

n
(cid:88)

i=1

CE (cid:0)yoh

i

, softmax(Wxi

+ b)(cid:1) .

Differently from least-squares, we cannot compute a
closed form solution anymore, but we can still proceed
with gradient descent. We will show in the next section an
example of gradient in this case, and in Section 6.3 a
generic technique to compute gradients in cases such as
this one.

4.3 More on classification

4.3.1 Binary classification

Consider now the specific case of m = 2.
In this case
we have y ∈ {0, 1}, and the problem reduces to binary
classification, sometimes called concept learning (as we
need to learn whether a certain binary “concept” is present
or absent in the input). With a standard logistic regression,
this would be modelled by a function having two outputs.
However, because of the softmax denominator, the last
output of a logistic regression is always redundant, as it
can be inferred knowing that the outputs must sum to 1:

(x) =

fm

m−1
(cid:88)

i=1

(x)

fi

Based on this, we can slightly simplify the formulation by
considering a scalar model with a single output f (x) ∈
[0, 1], such that:

90


Chapter 4: Linear models

91

Figure F.4.4:
Plot of the sigmoid
function. Note that
σ(0) = 0.5.

Predicted class = round( f (x)) =

(cid:168)

0

1

if f (x) ≤ 0.5
otherwise

To achieve the desired normalization in [0, 1], the first
output of a two-valued softmax can be rewritten as exp(x1
)
) ,
1+exp(x1
and we can further simplify it by dividing both sides by
exp(x1

). The result is the sigmoid function.

Definition D.4.6 (Sigmoid function)

The sigmoid function σ(s) : (cid:82) → [0, 1] is given by:

σ(s) =

1
1 + exp(−s)

The sigmoid provides a generic transformation projecting
any real value to the [0, 1] interval (with the two extremes
being reached only asymptotically). Its graph is shown in
Figure F.4.4.

The binary logistic regression model is obtained by
combining a one-dimensional linear model with a sigmoid

91

−10−50510s0.00.20.40.60.81.0Sigmoidσ(s)
92

More on classification

rescaling of the output:

f (x) = σ (cid:0)w⊤x + b(cid:1)

The cross-entropy similarly simplifies to:

Loss for class 1

Loss for class 2

CE( ˆy, y) = − y log( ˆy) − (1 − y) log(1 − ˆy)

(E.4.19)

Hence, in the binary classification case we can solve the
problem with two equivalent approaches: (a) a two-valued
model with the standard softmax, or (b) a simplified one-
valued output with a sigmoid output transformation.

As an interesting side-note, consider the gradient of the
binary logistic regression model with respect to w (a similar
gradient can also be written for the standard multi-class
case):

∇CE( f (x), y) = ( f (x) − y)x

Note the similarity with the gradient of a standard linear
model for regression. This similarity can be further
understood by rewriting our model as:

Logits

Sigmoid inverse: σ−1( y)

w⊤x + b = log

(cid:129) y

(cid:139)

1 − y

(E.4.20)

This clarifies why we were referring to the model as a

92


Chapter 4: Linear models

93

“linear model” for classification: we can always rewrite it
in terms of a non-linear
as a purely linear model
transformation of the output (in this case, the inverse of
the sigmoid, also known as the log-odds).
In fact, the
logistic regression model is part of a broader family of
models extending this idea, called generalized linear
models. For the curious reader, the name logits can be
understood in this context in reference to the probit
function.4

4.3.2 The logsumexp trick

This is a more technical subsection that clarifies an
implementation aspect of what we described up to now.
Looking at frameworks like TensorFlow or PyTorch, we
can find multiple existing implementations of
the
cross-entropy loss, based on whether the output is
described as an integer or as a one-hot encoded vector.
This can be understood easily, as we have already seen
that we can formulate the cross-entropy loss in both cases.
However, we can also find variants that accept logits
instead of the softmax-normalized outputs, as shown in
Box C.4.4.

To understand why we would need this, consider the i-th
term of the cross-entropy in terms of the logits p:

(cid:130)

− log

(cid:140)

.

exp pi
j exp p j

(cid:80)

This term can give rise to several numerical issues, notably
due to the interplay between the (potentially unbounded)
logits and the exponentiation. To solve this, we first rewrite

4https://en.wikipedia.org/wiki/Probit

93


94

More on classification

from torch.nn import functional as F
# Binary cross-entropy
F.binary_cross_entropy
# Binary cross-entropy accepting logits
F.binary_cross_entropy_with_logits
# Cross-entropy, but from logits
F.cross_entropy
# Cross-entropy with log f(x) as inputs
F.nll_loss

Box C.4.4: Cross entropy losses in PyTorch. Some losses are
only defined starting from the logits of the model, instead of the
post-softmax output. These are the functional variants of the
losses - equivalent object-oriented variants are also present in most
frameworks.

it as:

(cid:130)

− log

(cid:140)

exp pi
j exp p j

(cid:80)

= −pi

+ log

(cid:130)

(cid:88)

exp p j

j
(cid:123)(cid:122)
≜ logsumexp(p)

(cid:124)

(cid:140)

(cid:125)

The first term does not suffer from instabilities, while the
second term (the logsumexp of the logits) is a function of
the entire logits’ vector, and it can be shown to be invariant
for a given scalar c ≥ 0 in the following sense:5

logsumexp(p) = logsumexp(p − c) + c

Note that ∇softmax(•) = logsumexp(•).
By taking
c = max(p) we can prevent numerical problems by
bounding the maximum logit value at 0. However, this is
only possible if we have access to the original logits, which
is why numerically stable variants of the cross-entropy
require them as inputs. This creates a little amount of

5https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

94


Chapter 4: Linear models

95

ambiguity, in that the softmax can now be included as
either part of the model, or as part of the loss function.

4.3.3 Calibration and classification

We close the chapter by briefly discussing the important
topic of calibration of the classifier. To understand it,
consider the following fact: although our model provides
an entire distribution over the possible classes, our training
criterion only targets the maximization of the true class.
Hence, the following sentence is justified:

The predicted class of f (x) is arg max

[ f (x)]

i

i.

Instead, this more general sentence might not be correct:

The probability of x being of class i is [ f (x)]

i.

When the confidence scores of the network match the
probability of a given prediction being correct, we say the
network’s outputs are calibrated.

Definition D.4.7 (Calibration)

A classification model f (x) giving in output the class
probabilities is said to be calibrated if the following holds
for any possible prediction:

[ f (x)]

i

= p( y = i | x)

Although the cross entropy should recover the conditional
probability distribution over an unrestricted class of
models and in the limit of infinite data [HTF09],
in
practice the mismatch between the two may be high
[BGHN24], especially for the more complex models we
will introduce later on.

95


96

More on classification

To understand the difference between accuracy and
calibration, consider these two scenarios. First, consider a
binary classification model that has perfect accuracy, but
always predicts the true class with 0.8 confidence. In this
case, the model is clearly underconfident in its predictions,
since by looking at the confidence we may assume that
20% of them would be incorrect. Second, consider a 4
class problem with perfectly balanced classes, with a
model that always predict [0.25, 025, 0.25, 0.25]. In this
case, the model is perfectly calibrated, but useless from
the point of view of accuracy.

Having access to a calibrated model is very important in
situations in which different predictions may have different
costs. This can be formalized by defining a so-called cost
matrix assigning a cost Ci j for any input of class i predicted
as class j. A standard example is a binary classification
problem having the matrix of costs shown in Table T.4.2.

Table T.4.2: Example of cost matrix for a classification problem
having asymmetric costs of misclassification.

True class 0 True class 1

Predicted class 0
Predicted class 1

0
1

10
0

We can interpret Table T.4.2 as follows: making a correct
prediction incurs no cost, while making a false negative
mistake (0 instead of 1) is 10 times more costly than making
a false positive mistake. As an example, an incorrect false
negative mistake in a medical diagnosis is much worse than
a false positive error, in which a further test may correct the
mistake. A calibrated model can help us in better estimating
the average risk of its deployment, and to fine-tune our
balance of false positive and false negative mistakes.

96


Chapter 4: Linear models

97

To see this, denote by C ∼ (m, m) the generic matrix of
costs for a multiclass problem (like the 2 × 2 matrix in
Table T.4.2). The rational choice is to select a class which
minimizes the expected cost based on the scores assigned
by our model:

arg min
i

m
(cid:88)

j=1

Ci j

[ f (x)]

j

= 1 whenever i ̸= j and 0 otherwise, this reduces
If Ci j
to selecting the argmax of f , but for a general matrix of
costs the choice of predicted class will be influenced by the
relative costs of making specific mistakes. This is a simple
example of decision theory [Bis06].

4.3.4 Estimating the calibration error

To estimate whether a model is calibrated we can bin its
predictions, and compare its calibration to the accuracy
in each bin. To this end, suppose we split the interval
[0, 1] into b equispaced bins, each of size 1/b. Take a
validation set of size n, and denote by (cid:66)
i the elements
whose confidence falls into bin i. For each bin, we can
further compute the average confidence pi of the model
(which will be, approximately, in the middle of the bin),
)
and the average accuracy ai. Plotting the set of pairs (ai, pi
on an histogram is called a reliability diagram, as shown
in Figure F.4.5. To have a single, scalar metric of calibration
we can use, for example, the expected calibration error
(ECE):

Calibration for bin i

ECE = (cid:88)

i

|(cid:66)

|

i

n

|ai

− pi

|

(E.4.21)

Fraction falling into bin i

97


98

More on classification

Figure F.4.5: An example of reliability plot with b = 10 bins.
The blue bars show the average accuracy of the model on that bin,
while the red bars show the miscalibration for the bin, which can
be either under-confident (below the diagonal) or over-confident
(above the diagonal). The weighted sum of the red blocks is the
ECE in (E.4.21).

Other metrics, such as the maximum over the bins, are
If the model is found to be uncalibrated,
also possible.
modifications need to be made.
Examples include
rescaling the predictions via temperature scaling
[GPSW17] or optimizing with a different loss function
such as the focal loss [MKS+20].

We close by mentioning an alternative to direct calibration
of the model, called conformal prediction, which has
become popular recently [AB21].
Suppose we fix a
threshold γ, and we take the set of classes predicted by the
model whose corresponding probability is higher than γ:

(cid:67) (x) = {i | [ f (x)]

> γ}

i

(E.4.22)

i.e., the answer of the model is now a set (cid:67) (x) of potential
classes. An example is shown in Figure F.4.6. The idea of
conformal prediction is to select the minimum γ such that

98

0.00.20.40.60.81.0Conﬁdence0.00.20.40.60.81.0AccuracyPerfectcalibrationMiscalibration
Chapter 4: Linear models

99

Figure F.4.6: Calibration by turning the model’s output into a
set: we return all classes whose predicted probability exceeds a
given threshold. By properly selecting the threshold we can bound
the probability of the true class being found in the output set.

the probability of finding the correct class y in the set is
higher than a user-defined error α:6

p( y ∈ (cid:67) (x)) ≥ 1 − α

(E.4.23)

Intuitively, there is an inversely proportional relation
between γ and α.
Conformal prediction provides
automatic algorithms to guarantee (E.4.23) at the cost of
not having a single class in output anymore.

6Note that it is always possible to satisfy this property by selecting

γ = 0, i.e., including all classes in the output set.

99

0246Class0.00.10.20.30.4ConﬁdenceThresholdSelectedclasses
100

More on classification

From theory to practice

From Chapter 2 you should have a good
grasp of NumPy, JAX, and PyTorch’s
torch.tensor.
This is all that is
needed for this chapter, and nothing
else is required. From the next chapter
we will progress to their higher-level
APIs.

I suggest a short exercise to let you train your first
differentiable model from scratch:

1. Load a toy dataset:

for example, one of those

contained in scikit-learn datasets module.7

2. Build a linear model (for regression or classification
depending on the dataset). Think about how to make
the code as modular as possible: as we will see, you
will need at least two functions, one for initializing
the parameters of the model and one for computing
the model’s predictions.

3. Train the model via gradient descent. For now you
can compute the gradients manually: try to imagine
how you can make also this part modular, i.e., how
do you change the gradient’s computation if you want
to dynamically add or remove the bias from a model?

4. Plot the loss function and the accuracy on an
independent test set. If you know some standard
machine learning, you can compare the results to
other supervised learning models, such as a decision
tree or a k-NN, always using scikit-learn.

7https://scikit-learn.org/stable/datasets/toy_dataset.html

100
