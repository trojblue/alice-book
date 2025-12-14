A | Probability theory

About this chapter

Machine learning deals with a wide array of uncertainties
(such as in the data collection phase), making the use of
probability fundamental. We review here - informally -
basic concepts associated with probability distributions
and probability densities that are helpful in the main text.
This appendix introduces many concepts, but many of
them should be familiar. For a more in-depth exposition
of probability in the context of machine learning and
neural networks, see [Bis06, BB23].

A.1 Basic laws of probability

Consider a simple lottery, where you can buy tickets with 3
possible outcomes: “no win”, “small win”, and “large win”.
For any 10 tickets, 1 of them will have a large win, 3 will
have a small win, and 6 will have no win. We can represent
this with a probability distribution describing the relative
frequency of the three events (we assume an unlimited

341


342

Basic laws of probability

supply of tickets):

p(w = ‘no win’) = 6/10
p(w = ‘small win’) = 3/10
p(w = ‘large win’) = 1/10

Equivalently, we can associate an integer value w = {1, 2, 3}
to the three events, and write p(w = 1) = 6/10, p(w =
2) = 3/10, and p(w = 3) = 1/10. We call w a random
variable. In the following we always write p(w) in place
of p(w = i) for readability when possible. The elements of
the probability distribution must be positive and they must
sum to one:

p(w) ≥ 0,

p(w) = 1

(cid:88)

The space of all such vectors is called the probability
simplex.

w

Remember that we use p ∼ ∆(n) to denote a vector
of size n belonging to the probability simplex.

Suppose we introduce a second random variable r, a binary
variable describing whether the ticket is real (1) or fake
(2). The fake tickets are more profitable but less probable
overall, as summarized in Table T.A.1.

We can use the numbers in the table to describe a joint
probability distribution, describing the probability of two
random variables taking a certain value jointly:

p(r = 2, w = 3) = 8/100

Alternatively, we can define a conditional probability
distribution, e.g., answering the question “what is the
probability of a certain event given that another event has

342


Appendix A: Probability theory

343

Table T.A.1: Relative frequency of winning at an hypothetical
lottery, in which tickets can be either real or fake, shown for a set
of 100 tickets.

r = 1 (real ticket)

r = 2 (fake ticket)

w = 1 (no win)
w = 2 (small win)
w = 3 (large win)

Sum

occurred?”:

58
27
2

87

2
3
8

13

p(r = 1 | w = 3) = p(r = 1, w = 3)

p(w = 3)

= 0.2

This is called the product rule of probability. As before, we
can make the notation less verbose by using the random
variable in-place of its value:

p(r, w) = p(r | w)p(w)

(E.A.1)

If p(r | w) = p(r) we have p(r, w) = p(r)p(w), and we
say that the two variables are independent. We can use
conditional probabilities to marginalize over one random
variable:

p(w) = (cid:88)

p(w, r) = (cid:88)

p(w | r)p(r)

(E.A.2)

r

r

This is called the sum rule of probability. The product and
sum rules are the basic axioms that define the algebra of
probabilities.
By combining them we obtain the
fundamental Bayes’s rule:

343


344

Real-valued distributions

p(r | w) = p(w | r)p(r)

p(w)

= p(w | r)p(r)

(cid:80)

r ′ p(w | r ′)p(r ′)

(E.A.3)

Bayes’s rule allows us to “reverse” conditional distributions,
e.g., computing the probability that a winning ticket is real
or fake, by knowing the relative proportions of winning
tickets in both categories (try it).

A.2 Real-valued distributions

In the real-valued case, defining p(x) is more tricky,
because x can take infinitely possible values, each of
which has probability 0 by definition. However, we can
work around this by defining a probability cumulative
density function (CDF):

P(x) =

(cid:90) x

0

p(t)d t

and defining the probability density function p(x) as its
derivative. We ignore most of the subtleties associated with
working with probability densities, which are best tackled
in the context of measure theory [BR07]. We only note
that the product and sum rules continue to be valid in this
case by suitably replacing sums with integrals:

p(x, y) = p(x | y)p( y)

p(x) =

(cid:90)

y

p(x | y)p( y)d y

(E.A.4)

(E.A.5)

Note that probability densities are not constrained to be
less than one.

344


Appendix A: Probability theory

345

A.3 Common distributions

The previous random variables are example of categorical
probability distributions, describing the situation in which
a variable can take one out of k possible values. We can
write this down compactly by defining as p ∼ ∆(k) the
vector of probabilities, and by x ∼ Binary(k) a one-hot
encoding of the observed class:

p(x) = Cat(x; p) = (cid:89)

p xi
i

i

We use a semicolon to differentiate the input of the
If k = 2, we can
distribution from its parameters.
equivalently rewrite the distribution with a single scalar
value p. The resulting distribution is called a Bernoulli
distribution:

p(x) = Bern(x; p) = p x (1 − p)(1−x)

In the continuous case, we will deal repeatedly with the
Gaussian distribution, denoted by (cid:78) (x; µ, σ2), describing
a bell-shaped probability centered in µ (the mean) and
with a spread of σ2 (the variance):

p(x) = (cid:78) (x; µ, σ2) =

(cid:112)

1
2πσ2

exp

(cid:129)
− 1
2

(cid:16) x − µ
σ

(cid:17)2(cid:139)

345


346

Moments and expected values

In the simplest case of mean zero and unitary variance,
µ = 0, σ2 = 1, this is also called the normal distribution.
For a vector x ∼ (k), a multivariate variant of the Gaussian
distribution is obtained by considering a mean vector µ ∼
(k) and a covariance matrix Σ ∼ (k, k):

p(x) = (cid:78) (x; µ, Σ) =

(2π)−k/2 det(Σ)−1/2 exp (cid:0)(x − µ)⊤Σ−1(x − µ)(cid:1)

Two interesting cases are Gaussian distributions with a
diagonal covariance matrix, and the even simpler isotropic
Gaussian having a diagonal covariance with all entries
identical:

Σ = σ2I

The first can be visualized as an axis-aligned ellipsoid, the
isotropic one as an axis-aligned sphere.

A.4 Moments and expected values

In many cases we need to summarize a probability
distribution with one or more values. Sometimes a finite
number of values are enough: for example, having access
to p for a categorical distribution or to µ and σ2 for a
Gaussian distribution completely describe the distribution
itself. These are called sufficient statistics.

More in general, for any given function f (x) we can define
its expected value as:

(cid:69)

p(x) [ f (x)] = (cid:88)

f (x)p(x)

(E.A.6)

x

346


Appendix A: Probability theory

347

In the real-valued case, we obtain the same definition by
replacing the sum with an integral. Of particular interest,
when f (x) = x p we have the moments (of order p) of the
distribution, with p = 1 called the mean of the distribution:

(cid:69)

p(x) [x] = (cid:88)

x p(x)

x

We may want to estimate some expected values despite not
having access to the underlying probability distribution. If
we have access to a way of sampling elements from p(x),
we can apply the so-called Monte Carlo estimator:

(cid:69)

p(x) [ f (x)] ≈ 1
n

(cid:88)

∼p(x)

xi

f (x i

)

(E.A.7)

where n controls the quality of the estimation and we use
∼ p(x) to denote the operation of sampling from the
xi
probability distribution p(x). For the first-order moment,
this reverts to the very familiar notation for computing the
mean of a quantity from several measurements:

(cid:69)

p(x) [x] = 1
n

(cid:88)

x i

∼p(x)

xi

A.5 Distance between distributions

At times we may also require some form of distance
between probability distributions, in order to evaluate how
close two distributions are. The Kullback-Leibler (KL)
divergence between p(x) and q(x) is a common choice:

347


348

Maximum likelihood estimation

KL(p ∥ q) =

(cid:90)

p(x) log

p(x)
q(x)

d x

The KL divergence is not a proper metric (it is asymmetric
and does not respect the triangle inequality). It is lower
bounded at 0, but it is not upper bounded. The divergence
can only be defined if for any x such that q(x) = 0, then
p(x) = 0 (i.e., the support of p is a subset of the support
of q). The minimum of 0 is achieved whenever the two
distributions are identical. The KL divergence can be
written as an expected value, hence it can be estimated via
Monte Carlo sampling as in (E.A.7).

A.6 Maximum likelihood estimation

Monte Carlo sampling shows that we can estimate
quantities of interest concerning a probability distribution
if we have access to samples from it. However, we may be
interested in estimating the probability distribution itself.
Suppose we have a guess about its functional form f (x; s),
where s are the sufficient statistics (e.g., mean and
variance of a Gaussian distribution), and a set of n
∼ p(x). We call these samples identical
samples x i
(because
same probability
distribution) and independently distributed, in short, i.i.d.
Because of
joint distribution
independence,
factorizes for any choice of s:

from the

come

their

they

p(x1, . . . , x n

) =

n
(cid:89)

i=1

f (x i; s)

Large products are inconvenient computationally, but we
can equivalently rewrite this as a sum through a logarithmic

348


Appendix A: Probability theory

349

transformation:

L(s) =

n
(cid:88)

i=1

log( f (xi; s))

Finding the parameters s that maximize the previous
quantity is called the maximum likelihood (ML)
approach. Because of its importance, we reframe it briefly
below.

Definition D.A.1 (Maximum likelihood)

Given a parametric family of probability distributions
}n
f (x; s), and a set of n values {xi
i=1 which are i.i.d.
samples from an unknown distribution p(x), the best
approximation to p(x) according to the maximum
likelihood (ML) principle is:

s∗ = arg max

s

n
(cid:88)

i=1

log( f (xi; s))

If f is differentiable, we can maximize the objective through
gradient descent. This is the core approach we follow
for training differentiable models. For now, we close the
appendix by describing simple examples of ML estimation
in the case of standard probability distributions. We do
not provide worked out calculations, for which we refer to
[Bis06, BB23].

Maximum likelihood for the Bernoulli distribution

Consider first the case of a Bernoulli distribution with
unknown parameter p. In this case, the ML estimator is:

349


350

Maximum likelihood estimation

p∗ =

(cid:80)

i x i
n

which is the ratio of positive samples over the entire dataset.

Maximum likelihood for the Gaussian distribution

For the Gaussian distribution, we can rewrite its log
likelihood as:

L(µ, σ2) = − n
2

log(2πσ2) − 1
2σ2

n
(cid:88)

(x i

i=1

− µ)2

Maximizing for µ and σ2 separately returns the known
rules for computing the empirical mean and variance of a
Gaussian distribution:

µ∗ = 1
n

(cid:88)

xi

i

σ2∗ = 1
n

(cid:88)

(x i

i

− µ∗)2

(E.A.8)

(E.A.9)

The two can be computed sequentially. Because we are
using an estimate for the mean inside the variance’s
formula, the resulting estimation is shown to be slightly
This can be corrected by modifying the
biased.
normalization term to 1
this is known as Bessel’s
correction.1 For large n, the difference between the two
variants is minimal.

n−1 ;

1https://en.wikipedia.org/wiki/Bessel%27s_correction

350
