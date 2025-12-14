3 | Datasets and losses

About this chapter

This chapter formalizes the supervised learning scenario.
We introduce the concepts of datasets, losses, empirical
risk minimization, and the basic assumptions made
in supervised learning. We close by providing a
probabilistic formulation of supervised learning built on
the notion of maximum likelihood. This short chapter
serves as the backbone for the rest of the book.

3.1 What is a dataset?

We consider a scenario in which manually coding a certain
function is unfeasible (e.g., recognizing objects from
real-world images), but gathering examples of the desired
behaviour is sufficiently easy. Examples of this abound,
ranging from speech recognition to robot navigation. We
formalise this idea with the following definition.

51


52

What is a dataset?

Definition D.3.1 (Dataset)

n

)}n

= {(x i, yi

A supervised dataset (cid:83)
n of size n is a set of n pairs
(cid:83)
) is an example of
i=1, where each (xi, yi
an input-output relationship we want to model. We
further assume that each example is an identically and
independently distributed (i.i.d.)
draw from some
unknown (and unknowable) probability distribution
p(x, y).

See Appendix A if upon reading the definition you want to
brush up on probability theory. The last assumption appears
technical, but it is there to ensure that the relationship we
are trying to model is meaningful. In particular, samples
being identically distributed means that we are trying
to approximate something which is sufficiently stable and
unchanging through time. As a representative example,
consider the task of gathering a dataset to recognise car
models from photos. This assumption will be satisfied if
we collect images over a short time span, but it will be
invalid if collecting images from the last few decades, since
car models will have changed over time. In the latter case,
training and deploying a model on this dataset will fail
as it will be unable to recognise new models or will have
sub-optimal performance when used.

samples being independently distributed
Similarly,
means that our dataset has no bias in its collection, and it
is sufficiently representative of the entire distribution.
Going back to the previous example, gathering images
close to a Tesla dealership will be invalid, since we will
collect an overabundance of images of a certain type while
loosing on images of other makers and models. Note that
the validity of these assumptions depends on the context:
a car dataset collected in Italy may be valid when

52


Chapter 3: Datasets and losses

53

deploying our model in Rome or Milan, while it may be
invalid when deploying our model in Tokyo or in Taiwan.
The i.i.d. assumption should always be checked carefully
to ensure we are applying our supervised learning tools to
a valid scenario. Interestingly, modern LLMs are trained
on such large distributions of data
even
understanding what tasks are truly in-distribution against
what is out-of-distribution (and how much the models are
able to generalize) becomes blurred [YCC+24].

that

More on the i.i.d. property

Importantly, ensuring the i.i.d. property is not a one-shot
process, and it must be checked constantly during the
lifetime of a model. In the case of car classification, if
unchecked, subtle changes in the distribution of cars
over time will degrade the performance of a machine
learning model, an example of domain shift. As another
example, a recommender system will change the way
users interact with a certain app, as they will start
reacting to suggestions of the recommender system itself.
This creates feedback loops [CMMB22] that require
constant re-evaluation of the performance of the system
and of the app.

3.1.1 Variants of supervised learning

There exists many variations on the standard supervised
learning scenario, although most successful applications
make use of supervised learning in some form or another.
For example, some datasets may not have available
targets yi, in which case we talk about unsupervised
learning. Typical applications of unsupervised learning are
clustering algorithms, in which we want to aggregate our

53


54

What is a dataset?

Figure F.3.1: Differentiable models process data by transforming
it sequentially via linear algebra operations. In many cases, after
we optimize these programs, the internal representations of the
input data of the model (what we call a pre-trained model)
have geometric properties:
for example, semantically similar
images are projected to points that are close in this “latent” space.
Transforming data from a non-metric space (original input images)
to a metric space (bottom right) is called embedding the data.

54

"Cat""Cat""Dog""Pre-trained" model"Embedding" spaceObjects in this space act as standard vectors: we can sumthem, compute distances, rank them, etc.
Chapter 3: Datasets and losses

55

input data into clusters such that points in a cluster are
similar and points between clusters are dissimilar
[HTF09]. As another example, in a retrieval system we
may want to search a large database for the top-k most
similar elements to a user-given query.

When dealing with complex data such as images, this is
non-trivial because distances on images are ill-defined if we
operate on pixels (i.e., even small perturbations can modify
millions of pixels). However, assume we have available
some differentiable model that we have already optimized
for some other task which we assume sufficiently generic,
e.g., image classification. We call it a pre-trained model.
As we will see, the internal states of this model can be
interpreted as vectors in a high-dimensional space. In many
cases, these vectors are shown to have useful geometrical
properties, in the sense that objects that are semantically
similar are sent (embedded) into points that are close
in these representations. Hence, we can use these latent
representations with standard clustering models, such as
Gaussian mixture models [HHWW14]. See Figure F.3.1 for
a high-level overview of this idea.

What if we do not have access to a pre-trained model? A
common variation of unsupervised learning is called self-
supervised learning (SSL, [ZJM+21]). The aim of SSL
is to automatically find some supervised objective from a
generic unsupervised dataset, in order to optimize a model
that can be used in a large set of downstream tasks. For
example, if we have access to a large corpus of text, we can
always optimize a program to predict how a small piece of
text is likely to continue [RWC+19]. The realization that
neural networks can also perform an efficient embedding
of text when pre-trained in a self-supervised way had a

55


56

What is a dataset?

Figure F.3.2: Three ways of using trained models. Zero-shot: a
question is directly given to the model. This can be achieved with
generative language models (introduced in Chapter 8). Few-shot
prompting is similar, but a few examples are provided as input.
Both techniques can be employed only if the underlying model
shows a large amount of generalization capabilities. Fine-tuning:
the model is optimized via gradient descent on a small dataset
of examples. This proceeds similarly to training the model from
scratch.

profound impact on the community [MSC+13].1

As we will see in Chapter 8 and Chapter 10, LLMs can be
seen as modern iterations on this basic idea, since
optimizing models such as GPT or Llama [TLI+23] always
start by a basic self-supervised training in terms of
next-token prediction. These models are sometimes called
foundation models.
In the simplest case, they can be
used out-of-the-box for a new task, such as answering a
query: in this case, we say they are used in a zero-shot
fashion. For LLMs, it is also possible to provide a small
number of examples of a new task as input prompt, in
which case we talk about few-shot prompting.
In the

1Large-scale web datasets are also full of biases, profanity, and
vulgar content. Recognizing that models trained on this data internalize
these biases was another important realization [BCZ+16] and it
is one of the major criticisms of closed-source foundation models
[BGMMS21].

56

"Is this a cat?"Pre-trained model"Is this a cat?"Pre-trained model"This is a cat.""This is a racoon."Fine-tuned modelPre-trained modelFine-tuningZero-shot learningFew-shot promptingFine-tuningDataset"cat""racoon""Is this a cat?"
Chapter 3: Datasets and losses

57

most general case, we can take a pre-trained foundation
model and optimize its parameters by gradient descent on
a new task: this is called fine-tuning the model. See
Figure F.3.2 for a comparison of the three approaches. In
this book we focus on building models from scratch, but
fine-tuning can be done by similar means.

Fine-tuning is made especially easy by the presence of
large open-source repositories online.2 Fine-tuning can be
done on the full set of parameters of the starting model, or
by considering only a smaller subset or a small number of
additional parameters: this is called parameter-efficient
fine-tuning (PEFT) [LDR23].3 We will consider PEFT
techniques in the next volume.

Many other variations of supervised learning are possible,
which we do not have space to list in detail here except for
some generic hints. If only parts of a dataset are labeled,
we have a semi-supervised scenario [BNS06]. We will see
some examples of semi-supervised learning in Chapter 12.
Additionally, we can have scenarios with multiple datasets
belonging to “similar” distributions, or
the same
distribution over different period of times, giving rise to
countless problems depending on the order in which the
including domain
tasks or the data are provided,
adaptation, meta-learning [FAL17], continual learning
[PKP+19, BBCJ20], metric learning, unlearning, etc.
Some of these will be treated in the next volume.

2https://huggingface.co/models
3Few-shot learning can also be done by fine-tuning the model. In
cases in which fine-tuning is not needed, we say the model is performing
in-context learning [ASA+23].

57


58

Loss functions

3.2 Loss functions

Once data has been gathered, we need to formalize our
idea of “approximating” the desired behavior, which we do
by introducing the concept of loss functions.

Definition D.3.2 (Loss function)

Given a desired target y and the predicted value ˆy = f (x)
from a model f , a loss function l( y, ˆy) ∈ (cid:82) is a scalar,
differentiable function whose value correlates with the
performance of the model, i.e., l( y, ˆy1
) means
that the prediction ˆy1 is better than the prediction ˆy2
when considering the reference value (target) y.

) < l( y, ˆy2

A loss function embeds our understanding of the task and
our preferences in the solutions’ space on a real-valued scale
that can be exploited in an optimization algorithm. Being
differentiable, it allows us to turn our learning problem into
a mathematical optimization problem that can be solved
via gradient descent by minimizing the average loss on our
dataset.

To this end, given a dataset (cid:83)
)} and a loss
function l(·, ·), a sensible optimization task to solve is the
minimum average loss on the dataset achievable by any
possible differentiable model f :

= {(x i, yi

n

Average over the dataset

f ∗ = arg min

f

1
n

n
(cid:88)

i=1

l( yi,

f (x i

) )

(E.3.1)

Prediction on the i-th sample

58


Chapter 3: Datasets and losses

59

For historical reasons, (E.3.1) is referred to as empirical
risk minimization (ERM), where risk is used as a generic
synonym for loss. See also the box in the next page for
more on the origin of the term.

In (E.3.1) we are implicitly assuming that we are
minimizing across the space of all possible functions
defined on our input x. We will see shortly that our
models can always be parameterized by a set of tensors w
(called parameters of the model), and minimization is
done by searching for the optimal value of
these
parameters via numerical optimization, which we denote
by f (x, w). Hence, given a dataset (cid:83)
n, a loss function l,
and a model space f , we can train our model by
optimizing the empirical risk (E.3.1) via gradient descent
(E.2.27):

w∗ = arg min

w

1
n

n
(cid:88)

i=1

l( yi, f (xi, w))

(E.3.2)

where the minimization is now done with respect to the
parameter’s tensor w.

On the differentiability of the loss

First, note that

Before proceeding, we make two observations on the ERM
framework.
the differentiability
requirement on l
is fundamental. Consider a simple
binary classification task (that we will introduce properly
in the next chapter), where y ∈ {−1, +1} can only take
two values, −1 or 1. Given a real-valued model f (x) ∈ (cid:82),
we can equate the two decisions with the sign of f –
which we denote as sign( f (x)) – and define a 0/1 loss as:

59


60

Loss functions

l( y, ˆy) =

(cid:168)

0

1

if sign( ˆy) = y
otherwise

(E.3.3)

While this aligns with our intuitive notion of “being right”,
it is useless as loss function since its gradient will almost
always be zero (except when the sign of f switches), and
any gradient descent algorithm will remain stuck at
initialization. A less intuitive quantity in this case is the
margin y ˆy, which is positive [negative] depending on
whether the sign of the model aligns [or does not align]
with the desired one, but it varies continuously differently
from 0/1 loss in (E.3.3). A possible loss function in this
case is the hinge loss l( y, ˆy) = max(0, 1 − y ˆy), which is
used to train support-vector models. Details apart, this
shows the inherent
tension between designing loss
functions that encode our notion of performance while at
the same time being useful for numerical optimization.

Risk and loss

Empirical and expected risk minimization framed in
this way are generally associated with the work of the
Russian computer scientist V. Vapnik [Vap13], which gave
rise to the field of statistical learning theory (SLT). SLT
is especially concerned with the behaviour of (E.3.1)
when seen as a finite-sample approximation of (E.3.5)
under some restricted class of functions f and measure of
underlying complexity [PS+03, SSBD14, MRT18]. The
counter-intuitive properties of modern neural networks
(such as strong generalization long after overfitting
should have been expected) have opened many new
avenues of research in SLT [PBL20]. See also the
introduction of Chapter 9.

60


Chapter 3: Datasets and losses

61

3.2.1 Expected risk and overfitting

As a second observation, note that the empirical risk is
always trivial to minimize, by defining:

x is in the training set

f (x) =

(cid:40)

y

if (x, y) ∈ (cid:83)

n

¯y

otherwise

.

(E.3.4)

Default value, e.g., 0

This is a look-up table that returns a prediction y if the pair
(x, y) is contained in the dataset, while it defaults to some
constant prediction ¯y (e.g., 0) otherwise. Assuming that
the loss is lower-bounded whenever y = ˆy, this model will
always achieve the lowest possible value of empirical risk,
while providing no actual practical value.

This shows the difference between memorization and
learning (optimization). Although we search for a model
by optimizing some average loss quantity on our training
data, as in (E.3.1), our true objective is minimizing this
quantity on some unknown, future input yet to be seen.
The elements of our training set are only a proxy to this
end. We can formalize this idea by defining the expected
risk minimization problem.

Definition D.3.3 (Expected risk)

Given a probability distribution p(x, y) and a loss
function l, the expected risk (ER) is defined as:

ER[ f ] = (cid:69)

p(x, y) [l( y, f (x))]

(E.3.5)

Minimizing (E.3.5) can be interpreted as minimizing the

61


62

Loss functions

average (expected) loss across all possible input-output
pairs (e.g., all possible emails) that our model could see.
Clearly, a model with low expected risk would be
guaranteed to work correctly. However, the quantity in
as
(E.3.5)
enumerating and labeling all data points is impossible.
The empirical risk provides an estimate of the expected
risk under the choice of a given dataset and can be seen as
a Monte Carlo approximation of the ER term.

is unfeasible to compute in practice,

The difference in loss between the expected and the
empirical risk is called the generalization gap: a pure
memorization algorithm like (E.3.4) will have poor
generalization or, in other terms, it will overfit to the
specific training data we provided. Generalization can be
tested in practice by keeping a separate test dataset (cid:84)
m
with m data points never used during training,
= (cid:59). Then, the difference in empirical loss
(cid:83)
∩ (cid:84)
between (cid:83)
m can be used as an approximate
measure of overfitting.

n and (cid:84)

m

n

3.2.2 How to select a valid loss function?

If you have not done so already, this is a good time to
study (or skim) the material in Appendix A, especially
probability distributions, sufficient statistics, and
maximum likelihood estimation.

As we will see in the next chapters, the loss encodes our
a priori knowledge on the task to be solved, and it has
a large impact on performance.
In some cases, simple
considerations on the problem are enough to design valid
losses (e.g., as done for the hinge loss in Section 3.2).

However,

it is possible to work in a more principled

62


Chapter 3: Datasets and losses

63

fashion by reformulating the entire training process in
This
purely probabilistic terms, as we show now.
formulation provides an alternative viewpoint on learning,
which may be more intuitive or more useful in certain
scenarios. It is also the preferred viewpoint of many books
[BB23]. We provide the basic ideas in this section, and we
consider specific applications later on in the book.

The key observation is the following. In Section 3.1, we
started by assuming that our examples come from a
distribution p(x, y). By the product rule of probability, we
can decompose p(x, y) as p(x, y) = p(x)p( y | x), such
that p(x) depends on the probability of observing each
input x, and the conditional term p( y | x) describes the
probability of observing a certain output y given an input
x.4 Approximating p( y | x) with a function f (x) makes
sense if we assume that the probability mass is mostly
concentrated around a single point y, i.e., p( y | x) is close
to a so-called Dirac delta function, and it drastically
simplifies the overall problem formulation.

However, we can relax this by assuming that our model
f (x) does not provide directly the prediction, but it is
used instead to parameterize the sufficient statistics of a
conditional probability distribution p( y |
f (x)) over
possible outputs. For example, consider a classification
problem where y ∈ {1, 2, 3} can take three possible values.
We can assume our model has three outputs that
parameterize a categorical distribution over these classes,

4We can also decompose it as p(x, y) = p(x | y)p( y). Methods
that require to estimate p(x) or p(x | y) are called generative, while
methods that estimate p( y | x) are called discriminative. Apart from
language modeling, in this book we focus on the latter case. We
consider generative modeling more broadly in the next volume.

63


64

such that:

p(y | f (x)) =

3
(cid:89)

i=1

(x) yi

fi

Loss functions

where y ∼ Binary(3) is the one-hot encoding of the class
y 5 and f (x) ∼ ∆(3) are the predicted probabilities for
each class. As another example, assume we want to
predict a single scalar value y ∈ (cid:82) (regression). We can
model this with a two-valued function f (x) ∼ (2) such
that the prediction is a Gaussian with appropriate mean
and variance:

p( y | f (x)) = (cid:78) ( y | f1

(x),

(x) )

f 2
2

(E.3.6)

Squared to ensure positivity

where the second output of f (x) is squared to ensure that
the predicted variance remains positive. As can be seen,
this is a very general setup that subsumes our previous
discussion, and it provides more flexibility to the designer,
as choosing a specific parameterization for p( y | x) can
be easier than choosing a specific loss function l( y, ˆy). In
addition, this framework provides a more immediate way
to model uncertainty, such as the variance in (E.3.6).

3.2.3 Maximum likelihood

How can we train a probabilistic model? Remember that we
assumed the samples in our dataset (cid:83)
n to be i.i.d. samples
from a probability distribution p(x, y). Hence, given a
model f (x), the probability assigned to the dataset itself

5Given an integer i, its one-hot representation is a vector of all
zeros except the i-th element, which is 1. This is introduced formally
in Section 4.2.

64


Chapter 3: Datasets and losses

65

by a specific choice of function f is given by the product of
each sample in the dataset:

p((cid:83)

n

| f ) =

n
(cid:89)

i=1

p( yi

| f (x i

))

n

|

The quantity p((cid:83)
f ) is called the likelihood of the
dataset. For a random choice of f (x), the model will
assign probabilities more or less at random across all
possible inputs and outputs, and the likelihood of our
specific dataset will be small. A sensible strategy, then, is
to select the model such that the likelihood of the dataset
is instead maximized. This is a direct application of the
maximum likelihood approach (see Section A.6 in
Appendix A).

Definition D.3.4 (Maximum likelihood)

Given a dataset (cid:83)
)} and a family of
= {(xi, yi
probability distributions p( y | f (x)) parameterized by
f (x), the maximum likelihood solution is given by:

n

f ∗ = arg max

f

n
(cid:89)

i=1

p( yi

| f (x i

)) .

While we are again left with an optimization problem, it
now follows directly from the laws of probability once all
probability distributions are chosen, which is in contrast to
before, where the specific loss was part of the design
space.
The two viewpoints, however, are closely
connected. Working in log-space and switching to a
minimization problem we obtain:

65


66

Bayesian learning

(cid:168)

arg max
f

log

arg min
f

(cid:168) n

(cid:88)

i=1

(cid:171)

p( yi

| f (x i

))

n
(cid:89)

i=1

=

(cid:171)

− log(p( yi

| f (x i

))

(E.3.7)

Hence, the two formulations are identical if we identify
− log(p( y | f (x)) as a “pseudo-loss” to be optimized. As we
will see, all loss functions used in practice can be obtained
under the ML principle for specific choices of this term.
Both viewpoints are interesting, and we urge readers to
keep them in mind as we progress in the book.

3.3 Bayesian learning

We discuss here a further generalization of the probabilistic
formulation called Bayesian neural networks (BNNs),
which is of interest in the literature. We only provide the
general idea and we refer the reader to one of many in-
depth tutorials, e.g., [JLB+22], for more details.

By designing a probability function p( y | f (x)) instead of
f (x) directly, we can handle situations where more than
one prediction is of interest (i.e., the probability function
has more than a single mode). However, our procedure
still returns a single function f (x) out of the space of all
possible functions, while it may happen than more than a
single parameterization across the entire model’s space is
valid. In this case, it could be useful to have access to all
of them for a more faithful prediction.

Once again, we can achieve this objective by designing

66


Chapter 3: Datasets and losses

67

another probability distribution and then letting the rules
of probability guide us. Since we are now planning to
obtain a distribution across all possible functions, we start
by defining a prior probability distribution p( f ) over all
possible functions (recall than in practice f is described
by a finite set of parameters, in which case the prior p( f )
becomes a prior over these weights). For example, we will
see that in many situations functions with smaller norm
are preferred (as they are more stable), in which case we
∥ f ∥ for some norm ∥ f ∥ of f .
could define a prior p( f ) ∝ 1

Once a dataset is observed, the probability over f shifts
depending on the prior and the likelihood, and the update
is given by Bayes’ theorem:

Prior (before observing the dataset)

p( f | (cid:83)

n

) =

p((cid:83)

n

| f ) p( f )
)
p((cid:83)

n

(E.3.8)

Posterior (after observing the dataset)

n

| (cid:83)

The term p( f
) is called the posterior distribution
) in the denominator is called
function, while the term p((cid:83)
n
the evidence and it is needed to ensure that the posterior is
properly normalized. Assume for now that we have access
to the posterior. Differently from before, the distribution
can encode preference for more than a single function
f , which may provide better predictive power. Given an
input x, we can make a prediction by averaging all possible
models based on their posterior’s weight:

67


68

Prediction of f (x)
(cid:90)

Bayesian learning

Weight assigned to f

p( y | x) =

p( y | f (x)) p( f | (cid:83)

)

n

(E.3.9)

f

≈ 1
k

k
(cid:88)

i=1

p( y | fi

(x))p( fi

| (cid:83)

)

n

(E.3.10)

Monte Carlo approximation

n

where in (E.3.10) we have approximated the integral with
a Monte Carlo average over k random samples from the
). The overall beauty
∼ p( f | (cid:83)
posterior distribution fk
of this setup is marred by the fact that the posterior is
in general impossible to compute in closed-form, except
for very specific choices of prior and likelihood [Bis06].
Lacking this, one is forced to approximated solutions, either
by Markov chain Monte Carlo or by variational inference
[JLB+22]. We will see in Section 9.3.1 one example of
Bayesian treatment of the model’s parameters called Monte
Carlo dropout.

We remark on two interesting facts about the posterior
before closing this section. First, suppose we are only
interested about the function having highest posterior
density. In this case, the evidence term can be ignored and
the solution decomposes into two separate terms:

f ∗ = arg max

p((cid:83)

| f )p( f ) =

n

(cid:166)

f
log p((cid:83)

n

arg max
f

| f ) + log p( f ) (cid:169)

(E.3.11)

(E.3.12)

Likelihood term

Regularization term

68


Chapter 3: Datasets and losses

69

This is called the maximum a posteriori (MAP) solution.
If all functions have the same weight a priori (i.e., p( f ) is
uniform over the function’s space), then the second term
is a constant and the problem reduces to the maximum
likelihood solution. In general, however, the MAP solution
will impose a penalty to functions deviating too much
from our prior distribution. We will see this is a useful
idea to combat overfitting and impose specific constraints
on the function f . The term log p( f ) is generally called a
regularizer over the function’s space as it pushes the
solution towards the basin of attraction defined by the
prior distribution.6

Second, the full Bayesian treatment provides a simple way
to incorporate new data, e.g., a new dataset (cid:83) ′
n from the
same distribution. To do that, we replace the prior function
in (E.3.8) with the posterior distribution that we computed
on the first portion of the dataset, which now represents the
starting assumption on the possible values of f which gets
updated by looking at new data.7 This can mitigate issues
when training models online, most notably the so-called
catastrophic forgetting of old information [KPR+17].

6The difference between maximum likelihood and maximum a
posteriori solutions is loosely connected to the difference between
the frequentist and Bayesian interpretation of probability [Bis06],
i.e., probabilities as frequency of events or probabilities as a measure
of uncertainty. From a very high-level point of view, ML sees the
parameters as an unknown fixed term and the data as a random sample,
while a Bayesian treatment sees the data as fixed and the parameters
as random variables.

7Think of the original prior function as the distribution on f after

having observed an initial empty set of values.

69


70

Bayesian learning

70
