9 | Scaling up
the models

About this chapter

We now turn to the task of designing differentiable
models having dozens (or hundreds) of layers. As
we saw, the receptive field of convolutional models
grows linearly with the number of layers, motivating
architectures with such depth. This can be done by
properly stabilizing training using a plethora of methods,
ranging from data augmentation to normalization of the
hidden states.

9.1 The ImageNet challenge

Let us consider again the task of image classification, which
holds a strong interest for neural networks, both practically
and historically. In fact, interest in these models in the
period 2012-2018 can be associated in large part to the
ImageNet Large Scale Visual Recognition Challenge1
(later ImageNet for simplicity). ImageNet was a yearly

1https://image-net.org/challenges/LSVRC/

203


204

The ImageNet challenge

challenge that run from 2010 to 2017 to evaluate state-of-
the-art models for image classification. The challenge was
run on a subset of the entire ImageNet dataset, consisting
of approximately 1M images tagged across 1k classes.

It is instructive to take a look at the early editions of the
In 20102 and in 2011,3 the winners were
challenges.
linear kernels methods built with a combination of
specialized image descriptors and kernels, with a top-5%
error of 28% (2010) and 26% (2011). Despite a number
of promising results,4 convolutional models trained by
gradient descent remained a niche topic in computer
In 2012 the winner model (AlexNet,
vision.
[KSH12]) achieved a top-5% error of 15.3%, 10% lower
than all (non-neural) competitors.

Then,

This was followed by a veritable “Copernican revolution”
(apologies to Copernicus) in the field, since in a matter of
a few years almost all submissions turned to convolutional
models, and the overall accuracy grew at an unprecedented
speed, upward of 95% (leading to the end of the challenge
in 2017), as shown in Figure F.9.1. In a span of 5 years,
convolutional models trained with gradient descent became
the leading paradigm in computer vision, including other
subfields we are not mentioning here, from object detection
to semantic segmentation and depth estimation.

AlexNet was a relatively simple model consisting of 5
convolutional layers and 3 fully-connected layers, totaling
approximately 60M parameters, while the top-performing
models in Figure F.9.1 require up to hundreds of layers.
This is basic example of a scaling law (Chapter 1): adding

2https://image-net.org/challenges/LSVRC/2010/
3https://image-net.org/challenges/LSVRC/2011/
4https://people.idsia.ch/~juergen/computer-vision-contests-won-

by-gpu-cnns.html

204


Chapter 9: Scaling up the models

205

Figure F.9.1: Top-
1 accuracy on the
ImageNet dataset.
Reproduced from Papers
With Code.

layers and compute power for training is proportionally
linked to the accuracy of the model up to a saturation
point given by the dataset.
scaling up
convolutional models beyond a few layers is non-trivial, as
it runs into a number of problems ranging from slow
optimization to gradient issues and numerical instabilities.
As a consequence, a large array of techniques were
developed in 2012-2017 to stabilize training of very large
models.

However,

In this chapter we provide an overview of some of these
techniques. We focus on ideas and methods that are still
fundamental nowadays, even for other architectures (e.g.,
transformers). We begin by three techniques to improve
training that are well-known in machine learning: weight
regularization, data augmentation, and early stopping.
Then, we describe three of the most influential techniques
popularized in 2012-2017: dropout, batch normalization,
and residual connections, more or less in chronological
order of introduction. For each method we describe the
basic algorithm along with some variants that work well in
practice (e.g., layer normalization).

205


206

Data and training strategies

9.2 Data and training strategies

9.2.1 Weight regularization

One possible way to improve training is to penalize
solutions that may seem unplausible, such as having one
or two extremely large weights. Denote by w the vector of
all parameters of our model, and by L(w, (cid:83)
) the loss
function on our dataset (e.g., average cross-entropy). We
can formalize the previous idea by defining a so-called
regularization term R(w) that scores solutions based on
our preference, and penalize the loss by adding the
regularization term to the original loss function:

n

Lreg

= L(w, (cid:83)

n

) + λR(w)

where we assume that a higher value of R(w) corresponds
to a worse solution, and λ ≥ 0 is a scalar that weights the
two terms. For λ = 0 the regularization term has no effect,
while for λ → ∞ we simply select the best function based
on our a priori knowledge.

This can also be justified as performing maximum a-priori
(instead of maximum likelihood) inference based on the
combination of a prior distribution on the weights p(w)
and a standard likelihood function on our data (Section
3.3):

w∗ = arg max

{log p((cid:83)

w

| w) + log p(w)}

n

(E.9.1)

where having a regularization term corresponds to a non-
uniform prior distribution p(w). We have already seen one
example of regularization in Section 4.1.5, i.e., the ℓ
2 norm
of the weights:

206


Chapter 9: Scaling up the models

207

R(w) = ∥w∥2 = (cid:88)

w2
i

i

For the same unregularized loss, penalizing the ℓ
2 norm
will favor solutions with a lower weight magnitude,
corresponding to “less abrupt” changes in the output for a
small deviation in the input.5 Consider now the effect of
the regularization term on the gradient term:

∇Lreg

= ∇L(w, (cid:83)

) + 2λw

n

n

) = 0). For (S)GD, ℓ

Written in this form, this is sometimes called weight
decay, because absent the first term, its net effect is to
decay the weights by a small proportional factor λ
(sending them to 0 exponentially fast in the number of
iterations if ∇L(w, (cid:83)
2 regularization
and weight decay coincide. However, for other types of
optimization algorithms (e.g., momentum-based SGD,
Adam), a post-processing is generally applied on the
gradients. Denoting by g(∇L(w, (cid:83)
)) the post-processed
gradients of the (unregularized) loss, we can write a
generalized weight decay formulation (ignoring the
constant term 2) as:

n

Unregularized gradient

wt

= wt−1

− g(∇L(wt−1, (cid:83)

n

)) − λwt−1

Weight decay term

This is different from pure ℓ
2 regularization, in which case
the gradients of the regularization term would be inside
g(•). This is especially important for algorithms like Adam,

5With respect to (E.9.1), ℓ

2 regularization is equivalent to choosing

a Gaussian prior on the weights with diagonal σ2I covariance.

207


208

Data and training strategies

for which the weight decay formulation (known as AdamW
[LH19]) can work better.

We can also consider other types of regularization terms.
For example, the ℓ

1 norm:

R(w) = ∥w∥

1

= (cid:88)

|x i

|

i

can favor sparse solutions having a high percentage of zero
values (and it corresponds to placing a Laplace prior on
the weights). This can also be generalized to group sparse
variants to enforce structured sparsity on the neurons
[SCHU17].6 Sparse ℓ
1 penalization is less common than
for other machine learning models because it does not
the
interact well with the strong non-convexity of
optimization problem and the use of gradient descent
[ZW23]. However, it is possible to re-parameterize the
optimization problem to mitigate this issue at the cost of a
larger memory footprint. In particular, [ZW23] showed
that we can replace w with two equivalently shaped
vectors a and b, and:

w = a ⊙ b , ∥w∥

1

≈ ∥a∥2 + ∥b∥2

(E.9.2)

where ≈ means that the two problems can be shown to be
almost equivalent under very general conditions [ZW23].

We can gain some geometric insights as to why (and how)
regularization works by considering a convex loss function
L(·, ·) (e.g., least-squares), in which case the regularized

6Training sparse models is a huge topic with many connections
also to efficient hardware execution. See [BJMO12] for a review on
sparse penalties in the context of convex models, and [HABN+21] for
an overview of sparsity and pruning in general differentiable models.

208


Chapter 9: Scaling up the models

209

problem can be rewritten in an explicitly constrained form
as:

)
arg min
n
subject to R(w) ≤ µ

L(w, (cid:83)

(E.9.3)

where µ depends proportionally on λ, with the
unconstrained formulation arising by rewriting (E.9.3)
with a Lagrange multiplier. In this case, ℓ
2 regularization
corresponds to constraining the solution to lie inside a
circle centered in the origin, while ℓ
1 regularization
corresponds to having a solution inside (or on the vertices)
of a regular polyhedron centered in the origin, with the
sparse solutions lying at the vertices intersecting the axes.

9.2.2 Early stopping

From the point of view of optimization, minimizing a
function L(w) is the task of finding a stationary point as
) ≈ 0:
quickly as possible, i.e., a point wt such that ∇L(wt

∥L(wt

) − L(wt−1

)∥2 ≤ ϵ

for some tolerance ϵ > 0. However, this does not
necessarily correspond to what we want when optimizing
a model. In particular, in a low-data regime training for
too long can result in overfitting and, in general, anything
which improves generalization is good irrespective of its
net effect on the value on L(•) or the descent direction
(e.g., weight decay).

Early stopping is a simple example of the difference
between pure optimization and learning. Suppose we
have access to a small supervised dataset, separate from
the training and test dataset, that we call validation

209


210

Data and training strategies

dataset. At the end of every epoch, we track a metric of
interest on the validation dataset, such as the accuracy or
the F1-score. We denote the score at the t-th epoch as at.
The idea of early stopping is to check this metric to see if it
keeps improving: if not, we may be entering an overfitting
regime and we should stop training. Because the accuracy
can oscillate a bit due to random fluctuations, we do this
robustly by considering a window of k epochs (the
patience):

If at

≤ ai, ∀i = t − 1, t − 2, . . . , t − k → Stop training

Wait for k epochs

For a high value of the patience hyper-parameter k, the
algorithm will wait more, but we will be more robust to
possible oscillations. If we have a mechanism to store the
weights of the model (checkpointing) we can also rollback
the weights to the last epoch that showed improvement,
corresponding to the epoch number t − k.

Early stopping can be seen as a simple form of model
selection, where we select the optimal number of epochs
based on a given metric. Differently from the optimization
of the model, we can optimize here for any metric of
interest, such as the F1-score, even if not differentiable.

Interestingly, for large over-parameterized models early
stopping is not always beneficial, as the relation between
epochs and validation error can be non-monotone with
multiple phases of ascent and descent (a phenomenon
called multiple descents [RM22]) and sudden drops in
the loss after long periods of stasis [PBE+22]. Hence, early
stopping is useful mostly when optimizing on small
datasets.

210


Chapter 9: Scaling up the models

211

9.2.3 Data augmentation

Generally speaking, the most effective method to improve
performance for a model is to increase the amount of
available data. However, labelling data can be costly and
time-consuming, and generating data artificially (e.g.,
with the help of
large language models) requires
customized pipelines to work effectively [PRCB24].

In many cases, it is possible to partially mitigate this issue
by virtually increasing the amount of available data by
transforming them according to some pre-specified
number of (semantic preserving) transformations. As a
simple example, consider a vector input x and a
transformation induced by adding Gaussian noise:

x′ = x + ϵ, ϵ ∼ (cid:78) (0, σ2I)

This creates a virtually infinite amount of data comprised
in a small ball centered around x. In addition, this data
must not be stored in the disk, and the process can be
simulated by applying the transformation at runtime every
In fact, it is known
time a new mini-batch is selected.
that training in this way can make the model more robust
and it is connected to ℓ
2 regularization [Bis95]. However,
vectorial data is unstructured, and adding noise with too
high variance can generate points that are invalid.

For images, we can do better by noting that there is in
general a large number of transformations that can change
an image while preserving its semantic: zooms, rotations,
brightness modifications, contrast changes, etc. Denote by
T (x; c) one
rotation),
parameterized by some parameter c (e.g., the rotation
angle). Most transformations include the base image as a
special case (in this case, for example, with a rotation

such transformation (e.g.,

211


212

Data and training strategies

angle c = 0). Data augmentation is the process of
transforming images during training according to one or
more of these transformations:

x ′ = T (x; c), c ∼ p(c)

(E.9.4)

where p(c) denotes the distribution of all valid parameters
(e.g., rotation angles between −20◦ and +20◦). During
training, each element of the dataset is sampled once per
epoch, and each time a different transformation (E.9.4)
can be applied, creating a (virtually) unlimited stream of
unique data points.

Data augmentation is very common for images (or similar
data, such as audio and video), but it requires a number of
design choices: what transformations to include, which
parameters to consider, and how to compose these
transformations. A simple strategy called RandAugment
[CZSL20] considers a wide set of transformations, and for
every mini-batch samples a small number of them (e.g., 2
or 3), to be applied sequentially with the same magnitude.
Still, the user must verify that the transformations are
valid (e.g., if recognizing text, horizontal flipping can
make the resulting image invalid). From a practical point
of view, data augmentation can be included either as part
of the data loading components (see Box C.9.1), or as part
of the model.

Data augmentation pipelines and methods can be more
complex than simple intuitive transformations. Even for
more sophisticated types, the intuition remains that, as long
as the model is able to solve a task in a complex scenario
(e.g., recognizing an object in all brightness conditions)
it should perform even better in a realistic, mild scenario.
Additionally, data augmentation can prevent overfitting by
avoiding the repetition of the same input multiple times.

212


Chapter 9: Scaling up the models

213

# Image tensor (b, c, h, w)
img = torch.randint(0, 256,

size=(32, 3, 256, 256))

# Data augmentation pipeline
from torchvision.transforms import v2
transforms = v2.Compose([

v2.RandomHorizontalFlip(p=0.5),
v2.RandomRotation(10),

])

# Applying the data augmentation pipeline:
# each function call returns a different
# mini-batch starting from the same
# input tensor.
img = transforms(img)

Box C.9.1: Data augmentation pipeline with two transformations
applied in sequence, taken from the torchvision package. In
PyTorch, augmentations can be passed to the data loaders or used
independently. In other frameworks, such as TensorFlow and Keras,
data augmentation can also be included natively as layers inside
the model.

As an example of more sophisticated methods, we describe
mixup [ZCDLP17] for vectors, and its extension cutmix
[YHO+19] for images. For the former, suppose we sample
). The idea of mixup is
) and (x2, y2
two examples, (x1, y1
to create a new, virtual example which is given by their
convex combination:

x = λx1
y = λ y1

+ (1 − λ)x2
+ (1 − λ) y2

(E.9.5)

(E.9.6)

where λ is chosen randomly in the interval [0, 1]. This
procedure should push the model to have a simple (linear)
output in-between the two examples, avoiding abrupt

213


214

Dropout and normalization

changes in output. From a geometric viewpoint, for two
points that are close, we can think of (E.9.6) as slowly
moving on the manifold of the data, by following the line
that connects two points as λ goes from 0 to 1.

Mixup may not work for images, because linearly
interpolating two images pixel-by-pixel gives rise to
blurred images. With cutmix, we sample instead a small
patch of fixed shape (e.g., 32 × 32) on the first image.
Denote by M a binary mask of the same shape as the
images, with 1 for pixels inside the patch, and 0 for pixels
outside the patch. In cutmix, we combine two images x1
and x2 by “stitching” a piece from the first one on top of
the second one:

x = M ⊙ x1

+ (1 − M) ⊙ x2

while the labels are still linearly interpolated as before with
a random coefficient λ. See Figure F.9.2 for an example of
data augmentation using both rotation and cutmix.

9.3 Dropout and normalization

The strategies we have described in the previous section
are very general, in the sense that they imply modifications
to the optimization algorithm or to the dataset itself, and
they can be applied to a wide range of algorithms.

Instead, we now focus on three ideas that were
popularized in the period between 2012 and 2016, mostly
in the context of the ImageNet challenge. All three are
specific to differentiable models, since they can be
implemented as additional layers or connections in the
model that simplify training of very deep models. We list
the methods in roughly chronological order. As we will see

214


Chapter 9: Scaling up the models

215

Figure F.9.2: High-level overview of data augmentation. For every
mini-batch, a set of data augmentations are randomly sampled
from a base set, and they are applied to the images of the mini-
batch. Here, we show an example of rotation and an example of
cutmix. Illustrations by John Tenniel, reproduced from Wikimedia.

in the following chapters,
these methods
fundamental also beyond convolutional models.

remain

9.3.1 Regularization via dropout

When discussing data augmentation, we mentioned that
one insight is that augmentation forces the network to
learn in a more difficult setup, so that its performance in a
simpler environment can improve in terms of accuracy
and robustness. Dropout [SHK+14] extends this idea to
the internal embeddings of the model: by artificially
introducing noise during training to the intermediate
outputs of the model, the solution can improve.

There are many choices of possible noise types:
for
example, training with small amounts of Gaussian noise in

215

Augmentation1Augmentation2...AugmentationnRotationCutmixSample
216

Dropout and normalization

Figure F.9.3: Schematic overview of dropout: starting from a base
model, we add additional units after each layer of interest, shown
in blue. At training time, each dropout unit is randomly assigned a
binary value, masking part of the preceding layers. Hence, we select
one out of exponentially many possible models having a subset of
active hidden units every time a forward pass is made. Dropout
can also be applied at the input level, by randomly removing some
input features.

the activation has always been a popular alternative in the
literature of recurrent models. As the name suggests,
dropout’s idea is to randomly remove certain units
(neurons) during the computation,
reducing the
dependence on any single internal feature and (hopefully)
leading to training robust layers with a good amount of
redundancy.

We define dropout in the case of a fully-connected layer,
which is its most common use case.

Definition D.9.1 (Dropout layer)

Denote by X ∼ (n, c) a mini-batch of internal activations
of the model (e.g., the output of some intermediate fully-
connected layer) with n elements in the mini-batch and
c features. In a dropout layer, we first sample a binary
matrix M ∼ Binary(n, c) of the same size, whose elements

216

Original modelModel with dropout0110011010Sample masks
Chapter 9: Scaling up the models

217

are drawn from a Bernoulli distribution with probability
p (where p ∈ [0, 1] is a user’s hyper-parameter):a

Mi j

∼ Bern(p)

(E.9.7)

The output of the layer is obtained by masking the input:

Dropout(X) = M ⊙ X

The layer has a single hyper-parameter, p, and no
trainable parameters.

aThe samples from Bern(p) are 1 with probability p and 0 with

probability 1 − p.

We call 1 − p the drop probability. Hence, for any
element in the mini-batch, a random number of units
(approximately (1 − p)%) will be set to zero, effectively
removing them. This is shown in Figure F.9.3, where the
additional dropout units are shown in blue. Sampling the
mask is part of the layer’s forward pass: for two different
forward passes, the output will be different since different
elements will be masked, as shown on the right in Figure
F.9.3.

As the figure shows, we can implement dropout as a layer,
which is inserted after each layer that we want to regularize.
For example, consider the fully-connected model with two
layers shown in Figure F.9.3:

y = (FC ◦ FC)(x)

Adding dropout regularization over the input and over the

217


218

Dropout and normalization

model = nn.Sequential(

nn.Dropout(0.3),
nn.Linear(2, 3), nn.ReLU(),
nn.Dropout(0.3),
nn.Linear(3, 1)

)

Box C.9.2: The model in Figure F.9.3 implemented as a sequence
of four layers in PyTorch. During training, the output of the model
will be stochastic due to the presence of the two dropout layers.

output of the first layer returns a new model having four
layers:

y = (FC ◦ Dropout ◦ FC ◦ Dropout)(x)

See Box C.9.2 for an implementation in PyTorch.

While dropout can improve the performance, the output
y is now a random variable with respect to the sampling
of the different masks inside the dropout layers, which
is undesirable after training. For example, two forward
passes of the network can return two different outputs,
and some draws (e.g., with a very large number of zeroes)
can be suboptimal. Hence, we require some strategy to
replace the forward pass with a deterministic operation.

i=1 p(Mi

Suppose we have m dropout layers. Let us denote by Mi
the mask in the i-th dropout layer, by p(M1, . . . , Mm
) =
(cid:81)m
) the probability distribution over the union of
the masks, and by f (x; M) the deterministic output once a
given set of masks M ∼ p(M) are chosen. One choice is to
replace the dropout effect with its expected value during
inference:

218


Chapter 9: Scaling up the models

219

f (x) =

(cid:168)

f (x; M), M ∼ p(M)
p(M) [ f (x; M)]
(cid:69)

[training]
[inference]

We can approximate the expected value via Monte Carlo
sampling (Appendix A) by repeatedly sampling masks
values and averaging:

Ep(M) [ f (x; M)] ≈ 1
k

k
(cid:88)

i=1

f (x; Zi

), Zi

∼ p(M)

which is simply the average of k forward passes. This is
called Monte Carlo dropout [GG16]. The output is still
stochastic, but with a proper choice of k, the variance can
be contained.
In addition, the outputs of the different
forward passes can provide a measure of uncertainty over
the prediction.

However, performing multiple forward passes can be
expensive. A simpler (and more common) option is to
replace the random variables layer-by-layer, which is a
reasonable approximation. The expected value in this case
can be written in closed form:

(cid:69)

p(M) [Dropout(X)] = pX

which is the input rescaled by a constant factor p (the
probability of sampling a 1 in the mask). This leads to an
even simpler formulation, inverted dropout, where this
correction is accounted for during training:

219


220

Dropout and normalization

x = torch.randn((16, 2))

# Training with dropout
model.train()
y = model(x)

# Inference with dropout
model.eval()
y = model(x)

# Monte Carlo dropout for inference
k = 10
model.train()
y = model(x[:, None, :].repeat(1, k, 1))

.mean(1)

Box C.9.3: Applying the model from Box C.9.2 on a mini-batch of
16 examples. For layers like dropout, a framework requires a way
to differentiate between a forward pass executed during training
or during inference. In PyTorch, this is done by calling the train
and eval methods of a model, which set an internal train flag
on all layers. We also show a vectorized implementation of Monte
Carlo dropout.

Dropout(X) =




M ⊙ X
p



X

[training]

[inference]

In this case, the dropout layer has no effect when applied
during inference and can be directly removed. This is the
preferred implementation in most frameworks. See Box
C.9.3 for some comparisons.

As we mentioned, dropout (possibly with a low drop
probability, such as p = 0.8 or p = 0.9) is common for
fully-connected layers.
It is also common for attention
maps (introduced in the next chapter). It is less common

220


Chapter 9: Scaling up the models

221

for convolutional layers, where dropping single elements
of the input tensor results in sparsity patterns which are
too unstructured. Variants of dropout have been devised
which take into consideration the specific structure of
images: for example, spatial dropout [TGJ+15] drops
entire channels of the tensor, while cutout [DT17] drops
spatial patches of a single channel.

For example,
Other alternatives are also possible.
DropConnect [WZZ+13] drops single weights of a
fully-connected layer:

DropConnect(x) = (M ⊙ W)x + b

DropConnect in inference can also be approximated
efficiently with moment matching [WZZ+13]. However,
these are less common in practice, and the techniques
described next are preferred.

9.3.2 Batch (and layer) normalization

When dealing with tabular data, a common pre-processing
operation that we have not discussed yet is normalization,
i.e., ensuring that all features (all columns of the input
matrix) share similar ranges and statistics. For example,
we can pre-process the data to squash all columns in a
[0, 1] range (min-max normalization) or to ensure a zero
mean and unitary variance for each column (called either
standard scaling or normal scaling or z-score scaling).

Batch normalization (BN, [IS15]) replicates these ideas,
but for the intermediate embeddings of the model. This
is non trivial, since the statistics of a unit (e.g., its mean)
will change from iteration to iteration after each gradient
descent update. Hence, to compute the mean of a unit
we should perform a forward pass on the entire training

221


222

Dropout and normalization

dataset at every iteration, which is unfeasible. As the name
implies, BN’s core idea is to approximate these statistics
using only the data in the mini-batch itself.

Consider again the output of any fully-connected layer
X ∼ (n, c), where n is the mini-batch size. We will see
shortly how to extend the ideas to images and other types
of data. In BN, we normalize each feature (each column
of X) to have zero mean and unitary variance, based on
the mini-batch alone. To this end, we start by computing
the empirical column-wise mean µ ∼ (c) and variances
σ2 ∼ (c):

Mean of column j: µ

j

Variance of column j: σ2
j

= 1
n
= 1
n

(cid:88)

X i j

i
(cid:88)

(X i j

i

(E.9.8)

− µ

)2

j

(E.9.9)

We then proceed to normalize the columns:

Set the column mean to 0

X′ =

X − µ

(cid:112)

σ2 + ϵ

Set the column variance to 1

where we consider the standard broadcasting rules (µ and
σ2 are broadcasted over the first dimension), and ϵ > 0 is
a small positive term added to avoid division by zero.
Differently from normalization for tabular data, where this
operation is applied once to the entire dataset before
training, in BN this operation must be recomputed for
every mini-batch during each forward pass.

The choice of zero mean and unitary variance is just a

222


Chapter 9: Scaling up the models

223

convention, not necessarily the best one. To generalize it,
we can let the optimization algorithm select the best choice,
for a small overhead in term of parameters. Consider two
trainable parameters α ∼ (c) and β ∼ (c) (which we can
initialize as 1 and 0 respectively), we perform:

X′′ = αX′ + β

with similar broadcasting rules as above. The resulting
matrix will have mean β
i for the i-th
column. The BN layer is defined as the combination of
these two operations.

i and variance α

Definition D.9.2 (Batch normalization layer)

Given an input matrix X ∼ (n, c), a batch normalization
(BN) layer applies the following normalization:

BN(X) = α

(cid:139)

(cid:129) X − µ
σ2 + ϵ

(cid:112)

+ β

where µ and σ2 are computed according to (E.9.8) and
(E.9.9), while α ∼ (c) and β ∼ (c) are trainable
parameters. The layer has no hyper-parameters. During
inference, µ and σ2 are fixed as described next.

The layer has only 2c trainable parameters, and it can be
shown to greatly simplify training of complex models when
inserted across each block. In particular, it is common to
consider BN placed in-between the linear and non-linear
components of the model:

H = (ReLU ◦ BN ◦ Linear)(X)

Centering the data before the ReLU can lead to better

223


224

Dropout and normalization

exploiting its negative (sparse) quadrant. In addition, this
setup renders the bias in the linear layer redundant (as it
conflates with the β parameter), allowing to remove it.
Finally,
the double linear operation can be easily
optimized by standard compilers in most frameworks.

BN is so effective that is has led to a vast literature on
understanding why [BGSW18]. The original derivation
considered a problem known as internal covariate shift,
i.e., the fact that, from the point of view of a single layer,
the statistics of the inputs it receives will change during
optimization due to the changes in weights of the preceding
layers. However, current literature agrees that the effects of
BN is more evident in the optimization itself, both in terms
of stability and the possibility of using higher learning rates,
due to a combination of scaling and centering effects on
the gradients [BGSW18].7

Extending BN beyond tabular data is simple. For example,
consider a mini-batch of image embeddings X ∼ (n, h, w, c).
We can apply BN on each channel by considering the first
three dimensions together, i.e., we compute a channel-wise
mean as:

µ

z

= 1
nhw

(cid:88)

i, j,k

X i jkz

Mean of channel z (all pixels)

7See

also

https://iclr-blog-track.github.io/2022/03/25/
unnormalized-resnets/ for a nice entry point into this literature
(and the corresponding literature on developing normalizer-free
models.

224


Chapter 9: Scaling up the models

225

Batch normalization during inference

BN introduces a dependency between the prediction over
an input and the mini-batch it finds itself in, which is
unwarranted during inference (stated differently, moving
an image from one mini-batch to another will modify its
prediction). However, we can exploit the fact that the
model’s parameters do not change after training, and we
can freeze the mean and the variance to a preset value.
There are two possibilities to this end:

1. After training, we perform another forward pass on
the entire training set to compute the empirical mean
and variance with respect to the dataset [WJ21].

2. More commonly, we can keep a rolling set of
statistics that are updated after each forward pass of
the model during training, and use these after
training. Considering the mean only for simplicity,
µ = 0,
suppose we initialize another vector (cid:98)
corresponding to the “rolling mean of the mean”.
After computing µ as in (E.9.8), we update the
rolling mean with an exponential moving average:

µ ← λ
(cid:98)

µ + (1 − λ)µ
(cid:98)

where λ is set to a small value, e.g., λ = 0.01.
Assuming training converges, the rolling mean will
also converge to an approximation of the average
given by option (1). Hence, after training we can
use BN by replacing µ with the (pre-computed) (cid:98)
µ,
and similarly for the variance.8

8

µ is the first example of a layer’s tensor which is part of the layer’s
(cid:98)
state, is adapted during training, but is not needed for gradient descent.
In PyTorch, these are referred to as buffers.

225


226

Dropout and normalization

Variants of batch normalization

Despite its good empirical performance, BN has a few
important drawbacks. We have already mentioned the
dependence on the mini-batch, which has other
for example, the variance of µ during
implications:
training will grow large for small mini-batches, and
training can be unfeasible for very small mini-batch sizes.
In addition, training can be difficult in distributed contexts
(where each GPU holds a separate part of the mini-batch).
Finally, replacing µ with a different value after training
creates an undesirable mismatch between training and
inference.

Variants of BN have been proposed to address these issues.
A common idea is to keep the overall structure of the layer,
but to modify the axes along which the normalization is
performed. For example, layer normalization [BKH16]
computes the empirical mean and variance over the rows
of the matrix, i.e., for each input independently:

Mean of row i:

Variance of row i:

µ

i

σ2
i

= 1
c
= 1
c

(cid:88)

X ji

j
(cid:88)

(X ji

j

(E.9.10)

− µ

)2

i

(E.9.11)

Consider Figure F.9.4, where we show a comparison
between BN and LN for tabular and image-like data. In
particular, we show in blue all the samples used to
compute a single mean and variance.
For layer
normalization, we can compute the statistics on h, w, c
simultaneously (variant A) or for each spatial location
separately (variant B). The latter choice is common in

226


Chapter 9: Scaling up the models

227

Figure F.9.4:
Comparison between
BN and LN for tabular
and image data. Blue
regions show the sets
over which we compute
means and variances.
For LN we have two
variants, discussed
better in the main text.

transformer models, discussed in the next chapter. Other
variants are also possible, e.g., group normalization
restricts the operation to a subset of channels, with the
instance
channel
of
case
normalization.9

known as

single

a

In BN, the axes across which we compute the statistics in
(E.9.8) and (E.9.9) are the same as the axes across which
we apply the trainable parameters.
In LN, the two are
decoupled. For example, consider a PyTorch LN layer
applied on mini-batches of dimension (b, 3, 32, 32):

nn.LayerNorm(normalized_shape=[3, 32, 32])

This corresponds to variant A in Figure F.9.4. In this case, α
and β will have the same shape as the axes over which we
are computing the normalization, i.e., α, β ∼ (3, 32, 32),
for a total of 2 × 3 × 32 × 32 = 6144 trainable parameters.
The specific implementation of LN and BN must be checked
for each framework and model.

We close by mentioning another common variant of layer

9See https://iclr-blog-track.github.io/2022/03/25/unnormalized-

resnets/ for a nicer variant of Figure F.9.4.

227

Tabular dataImage dataBatch normalizationLayer normalizationVariant AVariant B
228

Residual connections

normalization, called root mean square normalization
(RMSNorm) [ZS19]. It simplifies LN by removing the mean
centering and shifting, which for a single input vector x ∼
(c) can be written as:

RMSNorm(x) =

x
(cid:80)

i x 2

i

(cid:113) 1
c

⊙ α

(E.9.12)

When β = 0 and the data is already zero-centered, LN and
RMSNorm are identical.

9.4 Residual connections

9.4.1 Residual connections and residual

networks

The combination of all techniques seen in the previous
section is enough to increase significantly the number of
layers in our models, but only up to a certain upper bound.
Consider three generic layers f1, f2, and f3, and two models
g1, g2 with g1 being a subset of g2:

(x) = ( f3

g1
(x) = ( f3

◦ f2

◦ f1

)(x)
◦ f1

)(x)

g2

(x) ≈ g1

Intuitively, by the universal approximation theorem it
should always be possible for the intermediate part, f2, to
(x) ≈ x, in which case
approximate the identity function f2
(x). Hence, there is always a setting of the
g2
parameters in which the second (deeper) model should
perform at least as well as the first (shallower) one.
However, this was not observed in practice, as shown in
Figure F.9.5.

We can solve this by biasing the blocks in the network

228


Chapter 9: Scaling up the models

229

Figure F.9.5:
Bigger models do
not always improve
monotonically in
training error, despite
representing larger
classes of functions.
Reproduced from
[HZRS16].

towards the identity function. This can be done easily by
rewriting a block f (x) with what is called a residual (skip)
connection [HZRS16]:

r(x) = f (x) + x

Hence, we use the block to model deviations from the
identity, f (x) = r(x) − x, instead of modeling deviations
from the zero function. This small trick alone helps in
training models up to hundreds of layers. We call f (x) the
residual path, r(x) a residual block, and a convolutional
model composed of residual blocks a residual network
(abbreviated to ResNet).

Residual connections work well with batch normalization
on the residual path, which can be shown to further bias
the model towards the identity at the beginning of training
[DS20]. However, residual connections can be added only
if the input and output dimensionality of f (x) are identical.
Otherwise, some rescaling can be added to the residual
connection. For example, if x is an image and f (x) modifies
the number of channels, we can add a 1 × 1 convolution:

r(x) = f (x) + Conv2D1×1

(x)

229


230

Residual connections

The benefit of a residual block can be understood also in
terms of its backward pass. Consider the VJP of the residual
block:

VJP of f

vjpr

(v) = vjp f

(v) + v⊤I = vjp f

(v) + v⊤

VJP of the skip connection

Hence, the forward pass lets the input x pass through
unmodified on the skip connection, while the backward
pass adds the unmodified back-propagated gradient v to
the original VJP, which can help mitigating gradient
instabilities.

On the design of the residual block

How to design the block f (x)?
batch-normalized block introduced earlier:

Consider

the

h = (ReLU ◦ BN ◦ Conv2D)
(cid:125)

(cid:124)

(cid:123)(cid:122)
= f (x)

(x) + x

Because the output of ReLU is always positive, we have that
h ≥ x (element-wise). Hence, a stack of residual blocks of
this form can only increase the values of the input tensor, or
set it to zero. For this reason, the original design proposed
in [HZRS16] considered a similar stack of blocks except for
the last activation function. As an example, for two blocks
we obtain the following design:

h = (BN ◦ Conv2D ◦ ReLU ◦ BN ◦ Conv2D)(x) + x

A series of blocks of this form can be preceded by a small
component with non-residual connections to reduce the
image dimensionality, sometimes called the stem. The

230


Chapter 9: Scaling up the models

231

Figure F.9.6: The original ResNet block [HZRS16], and the more
recent ResNeXt [LMW+22] block. As can be seen, the design has
shifted from an early channel reduction to a later compression
(bottleneck). Additional details (not shown) are the switch from
BN to LN and the use of GELU activation functions. Adapted from
[LMW+22].

specific choice of hyper-parameters for this block has varied
significantly over the years.

The original ResNet block proposed a compression in the
number of channels for the first operation, followed by a
standard 3 × 3 convolution and a final upscaling in the
number of channels. Recently, instead, bottleneck layers
like the ResNeXt block [LMW+22] (on the right in Figure
F.9.6) have become popular. To increase the receptive field
of the convolution, the initial layer is replaced by a
depthwise convolution. To exploit the reduced number of
parameters, the number of channels is increased by a given
factor (e.g., 3×, 4×), before being reduced by the last
1 × 1 convolution.

231

1x1 Convolution64 channels3x3 Convolution64 channels3x3 Convolution256 channelsInput (256 channels)Original ResNet designResNeXt design7x7 separable Convolution96 channels1x1 Convolution384 channels1x1 Convolution96 channelsInput (96 channels)
232

Residual connections

Figure F.9.7: Residual
paths: the black, red,
and blue paths are
implemented explicitly;
the green path is only
implicit.

9.4.2 Additional perspectives on residual

connections

We close the chapter by discussing two interesting
perspectives on the use of residual connections, which
have both been explored in-depth in current research.
First, consider a network composed of two residual blocks:

h1
h2

= f1
= f2

(x) + x
) + h1
(h1

(E.9.13)

(E.9.14)

If we unroll the computation:

h2

= f2

( f1

(x) + x) + f1

(x) + x

This corresponds to the sum of several paths in the network,
where the input is either left unmodified, it goes through
only a single transformation ( f1 or f2), or through their
combination.

It should be clear that the number of such paths grows
exponentially with the number of residual blocks. Hence,
deep residual models can be seen as a combination (an
ensemble) of a very large number of smaller models,
implemented through weight-sharing. This view can be
tested to show, for example, that ResNets tend to be robust
to small deletions or modifications of their elements
[VWB16]. This is shown visually in Figure F.9.7.

232


Chapter 9: Scaling up the models

233

Second, consider the following differential equation,
t
expressed in terms of a continuous parameter
representing time:

∂

t x t

= f (x, t)

We are using a neural network with arguments x and t (a
scalar) to parameterize the time derivative of some function.
This is called an ordinary differential equation (ODE). A
common problem with ODEs is integrating from a known
starting value x0 up to some specified time instant T :

x T

= x0

+

(cid:90) T

t=0

f (x, t)d t

Euler’s method10 for computing x T works by selecting a
small step size h and computing iteratively a first-order
discretization:

x t

= x t−1

+ h f (x t−1, t)

each layer corresponds

Merging h into f , this corresponds to a restricted form of
residual model, where all residual blocks share the same
to a discretized
weights,
time-instant, and x T is the output of the network. Under
this point of view, we can directly work with the original
continuous-time equation, and compute the output by
integrating it with modern ODE solvers. This is called a
neural ODE [CRBD18]. Continuous-time variants of
back-propagation can be derived that take the form of
another ODE problem. We will see in the next volume an
interesting connection between neural ODEs and a class of
generative models known as normalizing flows
[PNR+21].

10https://en.wikipedia.org/wiki/Euler_method

233


234

Residual connections

From theory to practice

All the layers we discussed in this
chapter (batch normalization, dropout,
are already implemented in
...)
PyTorch, Equinox, and practically every
other framework. For what concerns
the rest of the techniques we described,
for
it depends on the framework:
example, weight decay in implemented natively in
all PyTorch’s optimizers, data augmentation can be
found as transformations inside torchvision (and other
corresponding libraries), while early stopping must be
implemented manually.11

implementing

1. Before proceeding to the next chapter, I suggest you
try
batch
normalization as a layer using only standard linear
algebra routines, comparing the results with the
built-in layers.

either dropout

or

2. In Chapter 7 you should have implemented a simple
convolutional model for image classification. Try
progressively
adding
normalization, dropout, or residual connections as
needed.

increasing

size,

its

3. Take a standard architecture, such as a ResNet
[HZRS16], or a ResNeXt
Try
implementing the entire model by following the
suggestions from the original papers. Training on

[LMW+22].

11In PyTorch, a common alternative is to use an external library such
as PyTorch Lightning to handle the training process. Modifications to
the training procedure, such as early stopping, are pre-implemented in
the form of callback functions.

234


Chapter 9: Scaling up the models

235

ImageNet-like datasets can be challenging on a
consumer’s GPU – if you do not have access to good
hardware or cloud GPU hours, you can keep
focusing on simpler datasets, such as CIFAR-10.

4. At this point, you may have realized that training
from scratch very large models (e.g., ResNet-50) on
smaller datasets is practically impossible. One
solution is to initialize the weights of the model
from an online repository using, e.g., the weights of
a model trained on ImageNet, and fine-tuning the
model by modifying the last layer, corresponding to
the classification head. By this point of the book,
this should come as relatively easy – I suggest using
one of the many pre-trained models available on
torchvision or on the Hugging Face Hub.12 We will
cover fine-tuning more in-depth in the next volume.

12For an example tutorial: https://pytorch.org/tutorials/beginner/

transfer_learning_tutorial.html.

235


236

Residual connections

236


Part III

Down the rabbit-hole

“It would be so nice if

something made sense for a

change.”

—Alice in Wonderland,

1951 movie

237
