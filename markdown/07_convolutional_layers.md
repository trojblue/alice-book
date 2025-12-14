7 | Convolutional layers

About this chapter

In this chapter we introduce our second core layer, the
convolutional layer, which is designed to work with
images (or, more in general, sequential data of any kind)
by exploiting two key ideas that we call locality and
parameter sharing.

Fully-connected layers are important historically, but less
so from a practical point of view: on unstructured data
(what we also call tabular data, as it can be easily
represented as a table) MLPs are generally outperformed
by other alternatives, such as random forests or well tuned
support vector machines [GOV22]. This is not true,
however, as soon as we consider other types of data,
having some structure that can be exploited in the design
of the layers and of the model.

In this chapter we consider the image domain, while in
the next chapters we also consider applications to time
series, audio, graphs, and videos. In all these cases, the
input has a sequential structure (either temporal, spatial,
or of other type) that can be leveraged to design layers
that are both performant, easily composable, and highly

151


152

Towards convolutional layers

efficient in terms of parameters. Interestingly, we will see
that possible solutions can be designed by taking as starting
point a fully-connected layer, and then suitably restricting
or generalizing it based on the properties of the input.

7.1 Towards convolutional layers

7.1.1 Fully-connected layers are not enough

An image can be described by a tensor X ∼ (h, w, c), where
h is the height of the image, w the width of the image, and
c is the number of channels (which can be 1 for black and
white images, 3 for color images, or higher for, e.g.,
hyper-spectral images). Hence, a mini-batch of images
will generally be of rank 4 with an additional leading
batch dimension (b, h, w, c). The three dimensions are not
identical, since h and w represent a spatial arrangement of
pixels, while the channels c do not have a specific ordering,
in the sense that storing images in an RGB or a GBR
format is only a matter of convention.

On notation, channels, and features

We use the same symbol we used for features in the
tabular case (c) because it will play a similar role in
the design of the models, i.e., we can think of each
pixel as described by a generic set of c features which are
updated in parallel by the layers of the model. Hence, the
convolutional layer will return a generic tensor (h, w, c′)
with an embedding of size c′ for each of the hw pixels.

In order to use a fully-connected layer, we would need to

152


Chapter 7: Convolutional layers

153

“flatten” (vectorize) the image:

h = φ(W · vect(X ) )

(E.7.1)

Flattened image

where vect(x) is equivalent to x.reshape(-1) in PyTorch,
)
and it returns for a generic rank-n tensor x ∼ (i1, i2, . . . , in
an equivalent tensor x ∼ (cid:128)(cid:81)n

(cid:138)

.

j=1 i j

Although it should be clear this is an inelegant approach,
it is worth emphasizing some of its disadvantages. First,
we have lost a very important property from the previous
section, namely, composability: our input is an image,
is a vector, meaning we cannot
while our output
concatenate two of these layers. We can recover this by
reshaping the output vector to an image:

H = unvect(φ(W · vect(X )))

(E.7.2)

where we assume that the layer does not modify the number
of pixels, and unvect reshapes the output to a (h, w, c′)
tensor, with c′ an hyper-parameter.

This leads directly to the second issue, which is that the
layer has a huge number of parameters. Considering, for
example, a (1024, 1024) image in RGB, keeping the same
dimensionality in output results in (1024 ∗ 1024 ∗ 3)2
parameters (or (hw)2cc′ in general), which is in the order
of 1013! We can interpret the previous layer as follows: for
each pixel, every channel in the output is a weighted
combination of all channels of all pixels in the input
image. As we will see, we can obtain a more efficient
solution by restricting this computation.

153


154

Towards convolutional layers

More on reshaping

In order to flatten (or, more in general, reshape) a tensor,
we need to decide an ordering in which to process the
values. In practice, this is determined by the way the
tensors are stored in memory: in most frameworks, the
tensor’s data is stored sequentially in a contiguous block
of memory, in what is called a strided layout. Consider
the following example:

torch.randn(32, 32, 3).stride()
# (96, 3, 1)

The stride is the number of steps that must be taken in
memory to move of 1 position along that axis, i.e., the
last dimension of the tensor is contiguous, while to move
of one position in the first dimension we need 96 (32 ∗ 3)
steps. This is called a row-major ordering or, in image
analysis, a raster order.a Every reshaping operation
works by moving along this strided representation.

ahttps://en.wikipedia.org/wiki/Raster_scan

As a running example to visualize what follows, consider a
1D sequence (we will consider 1D sequences more in-depth
later on; for now, you can think of this as “4 pixels with a
single channel”):

x = (cid:2)x1, x2, x3, x4

(cid:3)

In this case, we do not need any reshaping operations, and
the previous layer (with c′ = 1) can be written as:

154


Chapter 7: Convolutional layers

155

Figure F.7.1: Given
a tensor (h, w, c) and a
maximum distance k, the
(i, j) (shown in
patch Pk
red) is a (2k + 1, 2k + 1, c)
tensor collecting all pixels at
distance at most k from the
pixel in position (i, j).











h1
h2
h3
h4

=






W11 W12 W13 W14
W21 W22 W23 W24
W31 W32 W33 W34
W41 W42 W43 W44
















x1
x2
x3
x4

7.1.2 Local layers

The spatial arrangement of pixels introduces a metric (a
distance) between the pixels. While there are many valid
notions of “distance”, we will find it convenient to work
with the following definition, which defines the distance
between pixel (i, j) and (i′, j′) as the maximum distance
across the two axes:

d((i, j), (i′, j′)) = max(|i − i′|, | j − j′|)

(E.7.3)

How can we exploit this idea in the definition of a layer?
Ideally, we can imagine that the influence of a pixel on
another one decreases with a factor inversely proportional
to their distance. Pushing this idea to its extreme, we can
assume that the influence is effectively zero for a distance
larger than some threshold. To formalize this insight, we
introduce the concept of a patch.

155

WidthHeightChannels
156

Towards convolutional layers

Definition D.7.1 (Image patch)

(i, j) as the
Given an image X , we define the patch Pk
sub-image centered at (i, j) and containing all pixels at
distance equal or lower than k:

(i, j) = [X ]

Pk

i−k:i+k, j−k: j+k,:

where distance is defined as in (E.7.3). This is shown
visually in Figure F.7.1.

The definition is only valid for pixels which are at least k
steps away from the borders of the image: we will ignore
this point for now and return to it later. Each patch is of
shape (s, s, c), where s = 2k + 1, since we consider k pixels
in each direction along with the central pixel. For reasons
that will be clarified later on, we call s the filter size or
kernel size.

Consider a generic layer H = f (X ) taking as input a tensor
of shape (h, w, c) and returning a tensor of shape (h, w, c′).
If the output for a given pixel only depends on a patch of
predetermined size, we say that the layer is local.

Definition D.7.2 (Local layer)

Given an input image X ∼ (h, w, c), a layer f (X ) ∼
(h, w, c′) is local if there exists a k such that:

[ f (X )]

= f (Pk

(i, j))

i j

This has to hold for all pixels of the image.

156


Chapter 7: Convolutional layers

157

We can transform the layer (E.7.1) into a local layer by
setting to 0 all weights belonging to pixels outside the
influence region (receptive field) of each pixel:

Flattened patch (of shape s2c)

(cid:16)

= φ

Hi j

Wi j

· vect(Pk

(i, j))

(cid:17)

Position-dependent weight matrix

We call this class of layers locally-connected. Note that
∼ (c′, ssc) for each
we have a different weight matrix Wi j
output pixel, resulting in hw(s2cc′) parameters.
By
comparison, we had (hw)2cc′ parameters in the initial
s2
layer, for a reduction factor of
hw in the number of
parameters.

Considering our toy example, assuming for example k = 1
(hence s = 3) we can write the resulting operation as:











h1
h2
h3
h4

=






W12 W13
0
0
W21 W22 W23
0
0 W31 W32 W33
0 W41 W42
0
















x1
x2
x3
x4

Our operation is not defined for x1 and x4, in which case
we have considered a “shortened” filter by removing the
weights
operations.
to
Equivalently, you can think of adding 0 on the border
whenever necessary:

corresponding

undefined

157


158

Towards convolutional layers






=











h1
h2
h3
h4

W11 W12 W13
0
0 W21 W22 W23
0
0

0
0
0 W31 W32 W33
0

0
0
0
0 W41 W42 W43
























0
x1
x2
x3
x4
0

This technique is called zero-padding. In an image, for a
kernel size 2k + 1 we need exactly k rows and columns of
0 on each side to ensure that the operation is valid for each
pixel. Otherwise, the output cannot be computed close to
the borders, and the output tensor will have shape (h −
2k, w − 2k, c′). Both are valid options in most frameworks.

On our definition of patches

The definition of convolutions using the idea of
patches is a bit unconventional, but I find it to
greatly simplify the notation.
I provide a more
conventional, signal processing oriented definition later
on. The two definitions are equivalent and can be used
interchangeably. The patch-oriented definition requires
an odd kernel size and does not allow for even kernel
sizes, but these are uncommon in practice.

7.1.3 Translation equivariance and the

convolutional layer

In a locally-connected layer, two identical patches can
result in different outputs based on their location: some
content on pixel (5, 2), for example, will be processed
differently than the same content on pixel (39, 81) because

158


Chapter 7: Convolutional layers

159

the two matrices W5,2 and W39,81 are different. For the
most part, however, we can assume that this information
is irrelevant: informally, “a horse is a horse”, irrespective
of its positioning on the input image. We can formalize
this with a property called translation equivariance.

Definition D.7.3 (Translation equivariance)

We say that a layer H = f (X ) is translation
equivariant if translations of the inputs imply an
equivalent translation of the output:

Identical patches

Pk

(i, j) = Pk

(i′, j′)

implies

f (Pk

(i, j)) = f (Pk

(i′, j′))

Identical outputs

To understand the nomenclature, note that we can
interpret the previous definition as follows: whenever an
object moves (translates) on the image from position (i, j)
to position (i′, j′), the output f (Pk
(i, j)) that we we had in
(i′, j′)). Hence, the
(i, j) will now be found in f (Pk
activations of the layer are moving with the same (èqui in
Latin) translational movement as the input. We will define
more formally equivariance and invariance later on.

A simple way to achieve translation equivariance is given
by weight sharing, i.e., letting every position share the
same set of weights:

Hi j

= φ( W · vect(Pk

(i, j)))

Weight matrix independent of (i, j)

This is called a convolutional layer, and it is extremely

159


160

Towards convolutional layers

efficient in terms of parameters: we only have a single
weight matrix W of shape (c′, ssc), which is independent
from the resolution of the original image (once again,
contrast this with a layer which is only locally-connected
with hw(s2c′c) parameters: we have reduced them by
another factor 1
hw ). We can write a variant with biases by
adding c′ additional parameters in the form of a bias
vector b ∼ (c′). Because of its importance, we restate the
full definition of the layer below.

Definition D.7.4 (Convolutional layer)

Given an image X ∼ (h, w, c) and a kernel size s = 2k +
1, a convolutional layer H = Conv2D(X ) is defined
element-wise by:

Hi j

= W · vect(Pk

(i, j)) + b

(E.7.4)

The trainable parameters are W ∼ (c′, ssc) and b ∼ (c′).
The hyper-parameters are k, c′, and (eventually) whether
to apply zero-padding or not.
In the former case the
output has shape (h, w, c′), in the latter case it has shape
(h − 2k, w − 2k, c′).

The equivalent
See Box C.7.1 for a code example.
object-oriented implementation can be
found in
torch.nn.Conv2D. By comparison, our toy example can
be refined as follows:











h1
h2
h3
h4

=






W2 W3
0
0
W1 W2 W3
0
0 W1 W2 W3
0 W1 W2
0
















x1
x2
x3
x4

(E.7.5)

160


Chapter 7: Convolutional layers

161

from torch.nn import functional as F
x = torch.randn(16, 3, 32, 32)
w = torch.randn(64, 3, 5, 5)
F.conv2d(x, w, padding='same').shape
# [Out]: torch.Size([16, 64, 32, 32])

Box C.7.1: Convolution in PyTorch. Note that the channel
dimension is – by default – the first one after the batch dimension.
The kernel matrix is organized as a (c′, c, k, k) tensor. Padding
can be specified as an integer or a string (‘same’ meaning that the
output must have the same shape as the input, ‘valid’ meaning no
padding).

]⊤
where we now have only three weights W = [W1, W2, W3
(the zero-padded version is equivalent to before and we
omit it for brevity). This weight matrix has a special
structure, where each element across any diagonal is a
constant (e.g., on the main diagonal we only find W2). We
call these matrices Toeplitz matrices,1 and they are
fundamental to properly implement a convolutional layer
on modern hardware. Toeplitz matrices are an example of
structured dense matrices [QPF+24]. Equation (E.7.5)
should also clarify that a convolution remains a linear
operation, albeit with a highly restricted weight matrix
compared to a fully-connected one.

Convolutions and terminology

Our terminology comes (mostly) from signal processing.
We can understand this by rewriting the output of the
convolutional layer in a more standard form. To this end,
we first rearrange the weight matrix into an equivalent
weight tensor W of shape (s, s, c, c′), similar to the PyTorch
implementation in Box C.7.1. For convenience, we also

1https://en.wikipedia.org/wiki/Toeplitz_matrix

161


162

Towards convolutional layers

define a function that converts an integer i′ from the
interval [1, . . . , 2k + 1] to the interval [i − k, . . . , i + k]:

t(i) = i − k − 1

(E.7.6)

where k is left implicit in the arguments of t(•). We now
rewrite the output of the layer with explicit summations
across the axes:

Hi jz

=

2k+1
(cid:88)

2k+1
(cid:88)

c
(cid:88)

i′=1

j′=1

d=1

[W ]

i′, j′,z,d

[X ]

i′+t(i), j′+t( j),d

(E.7.7)

Check carefully the indexing: for a given pixel (i, j) and
output channel z (a free index running from 1 to c′), on the
spatial dimensions W must be indexed along 1, 2, . . . , 2k+1,
while X must be indexed along i − k, i − k + 1, . . . , i + k −
1, i + k. The index d runs instead over the input channels.

From the point of view of signal processing, equation
(E.7.7) corresponds to a filtering operation on the input
signal X through a set of finite impulse response (FIR)
filters [Unc15], implemented via a discrete convolution
(apart from a sign change). Each filter here corresponds to
a slice W:,:,:,i of the weight matrix.
In standard signal
processing, these filters can be manually designed to
perform specific operations on the image. As an example,
a 3 × 3 filter to detect ridges can be written as:2





W =

−1 −1 −1
8 −1
−1
−1 −1 −1





In convolutional layers, instead, these filters are initialized
randomly and trained via gradient descent. We consider

2https://en.wikipedia.org/wiki/Kernel_(image_processing)

162


Chapter 7: Convolutional layers

163

built

convolutional models

the design of
on
convolutional layers in the next section. An interesting
aspect of convolutional layers is that the output maintains
a kind of “spatial consistency” and it can be plotted: we
call a slice H:,:,i of the output an activation map of the
layer, representing how much the specific filter was
“activated” on each input region. We will consider in more
detail the exploration of these maps in the next volume.

7.2 Convolutional models

7.2.1 Designing convolutional “blocks”

With the definition of a convolutional layer in hand, we
now turn to the task of building convolutional models,
also called convolutional neural networks (CNNs). We
consider the problem of image classification, although a lot
of what we say can be extended to other cases. To begin
with, we formalize the concept of receptive field.

Definition D.7.5 (Receptive field)

Denote by X an image, and by H = g(X ) a generic
intermediate output of a convolutional model, e.g., the
result of applying 1 or more convolutional layers. The
receptive field R(i, j) of pixel (i, j) is the subset of X
which contributed to its computation:

[g(X )]

i j

= g(R(i, j)), R(i, j) ⊆ X

For a single convolutional layer, the receptive field of a
pixel is equal to a patch: R(i, j) = Pk
(i, j). However, it is
easy to prove that for two convolutional layers in sequence

163


164

Convolutional models

(i, j), then P3k

with identical kernel size, the resulting receptive field is
(i, j) for three layers, and so
R(i, j) = P2k
on. Hence, the receptive field increases linearly in the
number of convolutional layers. This motivates our notion
of locality: even if a single layer is limited in its receptive
field by the kernel size, a sufficiently large stack of them
results in a global receptive field.

Consider now a sequence of two convolutional layers:

H = Conv(Conv(X ))

Because convolution is a linear operation (see previous
section), this is equivalent to a single convolution with a
larger kernel size (as per the above). We can avoid this
“collapse” in a similar way to fully-connected layers, by
interleaving them with activation functions:

H = (φ ◦ Conv ◦ . . . ◦ φ ◦ Conv)(X )

(E.7.8)

To continue with our design, we note that in (E.7.8) the
channel dimension will be modified by each convolutional
layer, while the spatial dimensions will remain of the same
shape (or will be slightly reduced if we avoid zero-padding).
However, it can be advantageous in practice to eventually
reduce this dimensionality if our aim is something like
image classification.

Consider again the example of a horse appearing in two
different regions across two different images.
The
translation equivariance property of convolutional layers
guarantees that every feature found in region 1 in the first
image will be found, correspondingly, in region 2 of the
second image. However, if our aim is “horse classification”,
we eventually need one or more neurons activating for an
horse irrespective of where it is found in the image itself: if

164


Chapter 7: Convolutional layers

165

we only consider shifts, this property is called translation
invariance.

Many operations that reduce over the spatial dimensions
are trivially invariant to translations, for example:

H ′ = (cid:88)

i, j

Hi j or H ′ = max

i, j

(Hi j

)

In the context of CNNs, this is called a global pooling.
However, this destroys all spatial information present in
the image. We can obtain a slightly more efficient solution
with a partial reduction, called max-pooling.

Definition D.7.6 (Max-pooling layer)

Given a tensor X ∼ (h, w, c), a max-pooling layer, denoted
as MaxPool(X) ∼ ( h

2 , c), is defined element-wise as:

2 , w

[MaxPool(X )]

= max

i jc

(cid:16)

[X ]

2i−1:2i,2 j−1:2 j,c

(cid:17)

2 × 2 image patch

Hence, we take 2×2 windows of the input, and we compute
the maximum value independently for each channel (this
is generalized trivially to larger windows). Max-pooling
effectively halves the spatial resolution while leaving the
number of channels untouched. An example is shown in
Figure F.7.2.

We can build a convolutional “block” by stacking several
convolutional layers with a max-pooling operation (see
Figure F.7.3):

ConvBlock(X ) = (MaxPool ◦ φ ◦ Conv ◦ . . . ◦ φ ◦ Conv)(X )

165


166

Convolutional models

Figure F.7.2:
Visualization of 2x2 max-
pooling on a (4,4,1) image.
For multiple channels,
the operation is applied
independently on each
channel.

Proceeding iteratively, we define a more complex network
by stacking together multiple such blocks:

H = (ConvBlock◦ConvBlock◦. . .◦ConvBlock)(X ) (E.7.9)

This design has a large number of hyper-parameters: the
output channels of each layer, the kernel size of each layer,
etc. It is common to drastically reduce the search space for
the design by making some simplifying assumptions. For
example, the VGG design [SLJ+15] popularized the idea
of maintaining the filter size constant in each layer (e.g.,
k = 3), while keeping the number of channels constant in
each block and doubling them in-between every block.

An alternative way for reducing the dimensionality is to
downsample the output of a convolutional layer: this is
called the stride of the convolution. For example, a
convolution with stride 1 is a normal convolution, while a
convolution with stride 2 will compute only one output
pixel every 2, a convolution with stride 3 will compute one
Large strides and
output every 3 pixels, and so on.
max-pooling can also be combined together depending on
how the entire model is designed.

166

3.2-1.50.20.72.70.5-1.83.00.41.3-2.00.11.25-0.6-0.81.03.23.01.31.25Max-pooling
Chapter 7: Convolutional layers

167

Figure F.7.3:
Abstracting away
from “layers” to
“blocks” to simplify
the design of
differentiable models.

Invariance and equivariance

Informally, if T is a transformation on x from some
set (e.g., all possible shifts), we say a function f
f (T x) = T f (x), and invariant if
is equivariant if
f (T x) = f (x). The space of all transformations form
a group [BBL+17], and the matrix corresponding to
a specific transformation is called a representation
for that group. Convolutional layers are (roughly)
equivariant to translations by design, but other strategies
can be found for more general forms of symmetries, such
as averaging over the elements of the group (frame
averaging, [PABH+21]). We will see other types of
layers’ equivariances in Chapter 10 and Chapter 12.

7.2.2 Designing the complete model

We can now complete the design of our model. By
stacking together multiple convolutional blocks as in
(E.7.9), the output H will be of shape (h′, w′, c′), where w′
and h′ depend on the number of max-pooling operations
(or on the stride of the convolutional layers), while c′ will
depend only on the hyper-parameters of
the last
convolutional layer in the sequence. Note that each
element Hi j will correspond to a “macro-region” in the
original image, e.g., if h′, w′ = 2, H11 will correspond to
the “top-left” quadrant in the original image. We can

167

Original image64 x 64 x 3Convolutional layerMax-poolingConvolutional layerMax-pooling...
168

Convolutional models

Figure F.7.4: Worked-out design of a very simple CNN for image
classification (assuming 10 output classes). We show the output
shape for each layer on the bottom. The global pooling operation
can be replaced with a flattening operation. The last (latent)
representation before the classification head is very useful when
fine-tuning large-scale pre-trained models – it is an embedding of
the image in the sense of Section 3.1.1.

remove this spatial dependency by performing a final
global pooling operation before classification.

The complete model, then, can be decomposed as three
major components: a series of convolutional blocks, a
global average pooling, and a final block for classification.

H = (ConvBlock ◦ . . . ◦ ConvBlock)(X )

(E.7.10)

h = 1
h′w′

(cid:88)

Hi j

i, j
y = MLP(h)

(E.7.11)

(E.7.12)

where MLP(h) is a generic sequence of fully-connected
layers (a flattening operation can also be used in place of
the global pooling). This is a prototypical example of a
CNN. See Figure F.7.4 for a worked-out example.

This design has a few interesting properties we list here:

168

Input shape(64, 64, 3)Convolutional layer32 filtersMax-pooling2 x 2 windowConvolutional layer64 filtersMax-pooling2 x 2 windowGlobal poolingFully-connected layer10 unitsShape(64, 64, 32)Shape(32, 32, 32)Shape(32, 32, 64)Shape(16, 16, 64)Shape(64)Shape(10)Backbone networkClassifier head
Chapter 7: Convolutional layers

169

1. It can be trained like the models described in
for
Chapter 4 and Chapter 5.
classification, we can wrap the output in a softmax
and train all parameters by minimizing the
cross-entropy. The same rules of back-propagation
described in Chapter 6 apply here.

For example,

2. Because of the global pooling operation, it does not
depend on a specific input resolution. However, it is
customary to fix this during training and inference
to simplify mini-batching (more on variable length
inputs in the next chapter).

3. (E.7.11) can be thought of as a “feature extraction”
block, while (E.7.12) as the “classification block”.
This interpretation will be very useful when we
consider transfer learning in the next volume. We
call the feature extraction block the backbone of the
model, and the classification block the head of the
model.

Notable types of convolution

We close the chapter by mentioning two instances of
convolutional layers that are common in practice.

First, consider a convolutional layer with k = 0, i.e., a
so-called 1 × 1 convolution. This corresponds to updating
each pixel’s embedding by a weighted sum of its channels,
disregarding all other pixels:

Hi jz

=

c
(cid:88)

t=1

Wz t X i j t

It is a useful operation for, e.g., modifying the channel
dimension (we will see an example when dealing with

169


170

Convolutional models

residual connections in Chapter 9).
In this case, the
parameters can be compactly represented by a matrix
W ∼ (c′, c). This is equivalent to a fully-connected layer
applied on each pixel independently.

Second, consider an “orthogonal” variant
convolutions,
neighborhood, but disregarding all channels except one:

to 1 × 1
in which we combine pixels in a small

Hi jc

=

2k+1
(cid:88)

2k+1
(cid:88)

i′=1

j′=1

Wi′, j′,c X i′+t(i), j′+t( j),c

where t(•) is the offset defined in (E.7.6). In this case we
have a rank-3 weight matrix W of shape (s, s, c), and each
output channel H:,:,c is updated by considering only the
corresponding input channel X :,:,c.
This is called a
depthwise convolution, and it can be generalized by
considering groups of channels, in which case it is called a
groupwise convolution (with the depthwise convolution
being the extreme case of a group size equal to 1).

We can also combine the two ideas and have a convolution
block made of alternating 1 × 1 convolutions (to mix the
channels) and depthwise convolutions (to mix the pixels).
This is called a depthwise separable convolution and it is
common in CNNs
low-power devices
targeted for
[HZC+17].
the number of
parameters for a single block (compared to a standard
convolution) is reduced from sscc′ to ssc + cc′. We will see
later how these decompositions, where the input is
processed alternatively across
are
fundamental for other types of architectures, such as
transformers, in Chapter 10.

Note that in this case,

separate axes,

170


Chapter 7: Convolutional layers

171

From theory to practice

the layers

All
introduced in this
chapter (convolution, max-pooling)
are implemented in the torch.nn
module.
The torchvision library
provides datasets and functions to load
images, as well as interfaces to apply
transformations to the images that will
be very useful in the next chapter.3

Before proceeding, I suggest you follow and re-implement
one of the many online tutorials on image classification in
torchvision, which should now be relatively easy to
follow.4 Toy image datasets abound, including MNIST
image
(digit classification) and CIFAR-10 (general
classification). Combining the torchvision loader with the
layers in Equinox allows you to replicate the same tutorial
in JAX, e.g.:

https://docs.kidger.site/equinox/examples/mnist/.

Implementing a convolution from scratch is also an
interesting exercise, whose complexity depends on the
level of abstraction.
One possibility is to use the
fold/unfold operations from PyTorch to extract the
patches.5 Premade kernels for convolutions will always be
significantly faster, making this a purely didactic exercise.

If you have some signal processing background, you may
know that convolution can also be implemented as

3https://pytorch.org/vision/stable/transforms.html
4As an example from the official documentation: https://pytorch.

org/tutorials/beginner/blitz/cifar10_tutorial.html

5See for example: https://github.com/loeweX/Custom-ConvLayers-

Pytorch

171


172

Convolutional models

multiplication by moving to the frequency domain. This is
impractical for the small kernels we tend to use, but it can
be useful for very large (also known as long) convolutions,
e.g.:

https://github.com/fkodom/fft-conv-pytorch

PyTorch also provides a differentiable Fast Fourier
transform that you can use as a starting point.

172
