1 | Introduction

Neural networks have become an integral component of
our everyday’s world, either openly (in the guise of large
language models, LLMs), or hidden from view, by
powering countless technologies and scientific discoveries
including drones, cars, search engines, molecular design,
and recommender systems [WFD+23]. As we will see, all
of this has been done by relying on a very small set of
guiding principles and components, forming the core of
this book, while the research focus has shifted to scaling
them up to the limits of what is physically possible.

The power of scaling is embodied in the relatively recent
concept of neural scaling laws, which in turn has been
instrumental in driving massive investments in artificial
intelligence (AI) [KMH+20, HBE+24]:
informally, for
practically any task, simultaneously increasing data,
compute power, and the size of the models – almost
always – results in a predictable increase in accuracy.
Stated in another way, the compute power required to
achieve a given accuracy for a task is decreasing by a
constant factor per period of time [HBE+24].
The
tremendous power of combining simple, general-purpose
tools with exponentially increased computational power in

1


2

Introduction

Figure F.1.1: Training cost (in US dollars) of notable AI models
released from 2016. Training cost is correlated to the three key
factors of scaling laws: size of the datasets, compute power, and
size of the models. As performance steadily increases, variations in
modeling become asymptotically less significant [HBE+24]. Data
reproduced from the Stanford AI Index Report 2024.2

AI was called the bitter lesson by R. Sutton.1

If we take scaling laws as given, we are left with an almost
magical tool. In a nutshell, neural networks are optimized
to approximate some probability distribution given data
drawn from it. In principle, this approximation may fail:
for example, modern neural networks are so large that
they can easily memorize all the data they are shown
[ZBH+21] and transform into a trivial
look-up table.
Instead, trained models are shown to generalize well even
to tasks that are not explicitly considered in the training
data [ASA+23].
In fact, as the size of the datasets
increases, the concept of what is in-distribution and what
is out-of-distribution blurs, and large-scale models show
strong generalization capabilities and a
hints of

1http://www.incompleteideas.net/IncIdeas/BitterLesson.html.
2https://hai.stanford.edu/research/ai-index-report

2

20172018201920202021202220232024Year102103104105106107108Trainingcost(USdollars)
Chapter 1: Introduction

3

fascinating low dependency on pure memorization, i.e.,
overfitting [PBE+22].

The emergence of extremely large models that can be
leveraged for a variety of downstream tasks (sometimes
called foundation models), coupled with a vibrant
open-source community,3 has also shifted how we interact
with these models. Many tasks can now be solved by
simply prompting (i.e., interacting with text or visual
instructions) a pre-trained model found on the web
[ASA+23], with the internals of the model remaining a
complete black-box. From a high-level perspective, this is
similar to a shift from having to programs your libraries in,
e.g., C++, towards relying on open-source or commercial
software whose source code is not accessible.
The
metaphor is not as far fetched as it may seems: nowadays,
few teams worldwide have the compute and the technical
expertise to design and release truly large-scale models
such as the Llama LLMs [TLI+23], just like few companies
have the resources to build enterprise CRM software.

And in the same way,
just like open-source software
provides endless possibilities for customizing or designing
from scratch your programs, customer-grade hardware
and a bit of ingenuity gives you a vast array of options to
experiment with differentiable models, from fine-tuning
them for your tasks [LTM+22] to merging models
[AHS23], quantizing them for low-power hardware,
testing their robustness, or even designing completely new
variants and ideas. For all of this, you need to look ‘under
the hood’ and understand how these models process and
manipulate data internally, with all their tricks and
idiosincrasies
that are born from experience and
debugging. This book is an entry point into this world: if,

3https://huggingface.co/

3


4

Introduction

like Alice, you are naturally curious, I hope you will
appreciate the journey.

About this book

We assume our readers are familiar with the basics of
machine learning (ML), and more specifically supervised
learning (SL). SL can be used to solve complex tasks by
gathering data on a desired behavior, and ‘training’
(optimizing) systems to approximate that behavior. This
for
deceptively simple idea is extremely powerful:
example,
image generation can be turned into the
problem of collecting a sufficiently large collection of
simulating the English
images with their captions;
language becomes the task of gathering a large collection
of text and learning to predict a sentence from the
preceding ones; and diagnosing an X-ray becomes
equivalent to having a large database of scans with the
associated doctors’ decision (Figure F.1.2).

a

a

large number

program with

In general, learning is a search problem. We start by
of
defining
degree-of-freedoms (that we call parameters), and we
manipulate the parameters until the model performance is
satisfying. To make this idea practical, we need efficient
ways of searching for the optimal configuration even in the
presence of millions (or billions, or trillions) of parameters.
As the name implies, differentiable models do this by
restricting the selection of the model to differentiable
components, i.e., mathematical functions that we can
differentiate. Being able to compute a derivative of a
high-dimensional function (a gradient) means knowing
what happens if we slightly perturb their parameters,
which in turn leads to automatic routines for their

4


Chapter 1: Introduction

5

Figure F.1.2: Most tasks can be categorized based on the desired
input - output we need: image generation wants an image (an
ordered grid of pixels) from a text (a sequence of characters),
while the inverse (image captioning) is the problem of generating a
caption from an image. As another example, audio query answering
requires a text from an audio (another ordered sequence, this time
numerical). Fascinatingly, the design of the models follow similar
specifications in all cases.

optimization (most notably, automatic differentiation
and gradient descent). Describing this setup is the topic
of the first part of the book (Part I, Compass and Needle),
going from Chapter 2 to Chapter 6.

By viewing neural networks as simply compositions of
differentiable primitives we can ask two basic questions
(Figure F.1.3): first, what data types can we handle as
inputs or outputs? And second, what sort of primitives can
we use? Differentiability is a strong requirement that does
not allow us to work directly with many standard data
integers, which are
types,
fundamentally discrete and hence discontinuous.
By
contrast, we will see that differentiable models can work
easily with more complex data represented as large arrays
(what we will call tensors) of numbers, such as images,

such as characters or

5

Image captioningParis"What is the capitalof France?"Imagegeneration"An image of theTour Eiffel"Audio queryanswering"An image of theTour Eiffel"
6

Introduction

Figure F.1.3: Neural networks are sequences of differentiable
primitives which operate on structured arrays (tensors): each
primitive can be categorized based on its input/output signature,
which in turn defines the rules for composing them.

which can be manipulated algebraically by basic
compositions of linear and nonlinear transformations.

In the second part of the book we focus on a prototypical
example of differentiable component, the convolutional
operator (Part II, from Chapter 7 until Chapter 9).
Convolutions can be applied whenever our data can be
represented by an ordered sequence of elements: these
include, among others, audio, images, text, and video.
Along the way we also introduce a number of useful
techniques to design deep (a.k.a., composed of many steps
in sequence) models, as well as several important ideas
such as text tokenization, autoregressive generation of
sequences, and causal modeling, which form the basis for
state-of-the-art LLMs.

The third part of the book (Part III, Down the Rabbit
Hole) continues our exploration of differentiable models by
considering alternative designs for sets (most importantly
attention layers and transformer models in Chapter 10
and 11), graphs (Chapter 12), and finally recurrent layers
for temporal sequences (Chapter 13).

6

def my_program(x: tensor) -> tensor:    ...    ...    ...    return yInput typesOutput typesDifferentiableprimitives
Chapter 1: Introduction

7

including generative modeling,

The book is complemented by a website4 where I will
(hopefully) collect additional chapters and material on
topics of interest that do not focus on a specific type of
conditional
data,
computation, transfer learning, and explainability.
These chapters are more research-oriented in nature and
can be read in any order. In addition, I provide a series of
guided lab sessions in notebook form, which cover a large
part of the material from the book as well as advanced
topics such as contrastive learning and model merging.5

In the land of differentiability

Neural networks have a long and rich history. The name
itself is a throwback to early attempts at modeling
(biological) neurons in the 20th century, and similar
terminology has remained pervasive: to be consistent with
existing frameworks, in the upcoming chapters we may
refer to neurons, layers, or, e.g., activations. After multiple
waves of interest, the period between 2012 and 2017 saw
an unprecedented rise in complexity in the networks
spurred by large-scale benchmarks and competitions, most
notably the ImageNet Large Scale Visual Recognition
Challenge (ILSVRC) that we cover in Chapter 9. A second
major wave of interest came from the introduction of
transformers (Chapter 10) in 2017: just like computer
vision was overtaken by convolutional models a few years
before, natural language processing was overtaken by
transformers in a very short period. Further improvements
in these years were done for videos, graphs (Chapter 12),
and audio, culminating in the current excitement around

4https://sscardapane.it/alice-book
5http://tinyurl.com/guided-labs

7


8

Introduction

LLMs, multimodal networks, and generative models.6

This period paralleled a quick evolution in terminology,
from the connectionism of the 80s [RHM86] to the use of
deep learning for referring to modern networks in
opposition to the smaller, shallower models of the past
[Ben09, LBH15]. Despite this, all these terms remain
inexorably vague, because modern (artificial) networks
retain almost no resemblance to biological neural
networks and neurology [ZER+23]. Looking at modern
neural networks, their essential characteristic is being
composed of differentiable blocks: for this reason, in this
book I prefer the term differentiable models when
feasible. Viewing neural networks as differentiable models
leads directly to the wider topic of differentiable
programming, an emerging discipline that blends
computer science and optimization to study differentiable
computer programs more broadly [BR24].7

As we travel through this land of differentiable models, we
are also traveling through history: the basic concepts of
numerical optimization of linear models by gradient
descent (covered in Chapter 4) were known since at least
the XIX century [Sti81];
so-called “fully-connected
networks” in the form we use later on can be dated back
to the 1980s [RHM86]; convolutional models were known

6This is not the place for a complete historical overview of modern
neural networks; for the interested reader, I refer to [Met22] as a great
starting point.

7Like many,

I was inspired by a ‘manifesto’ published by
Y. LeCun on Facebook in 2018:
https://www.facebook.com/yann.
lecun/posts/10155003011462143. For the connection between neural
networks and open-source programming (and development) I am
also thankful to a second manifesto, published by C. Raffel in
2021: https://colinraffel.com/blog/a-call-to-build-models-like-we-
build-open-source-software.html.

8


Chapter 1: Introduction

9

Figure F.1.4: AI hype - except it is 1958, and the US psychologist
Frank Rosenblatt has gathered up significant media attention with
his studies on “perceptrons”, one of the first working prototypes of
neural networks.

and used already at the end of the 90s [LBBH98].8
However, it took many decades to have sufficient data and
power to realize how well they can perform given enough
data and enough parameters.

While we do not have space to go in-depth on all possible
topics (also due to how quickly the research is progressing),
I hope the book provides enough material to allow the
reader to easily navigate the most recent literature.

Notation and symbols

The
fundamental data type when dealing with
differentiable models is a tensor,9 which we define as an

8For a history of NNs up to this period through interviews to some
of the main characters, see [AR00]; for a large opinionated history
there is also an annotated history of neural networks by J. Schmidhuber:
https://people.idsia.ch/~juergen/deep-learning-history.html.

9In the scientific literature, tensors have a more precise definition
as multilinear operators [Lim21], while the objects we use in the book
are simpler multidimensional arrays. Although a misnomer, the use of
tensor is so widespread that we keep this convention here.

9


10

Introduction

Figure F.1.5: Fundamental data types: scalars, vectors, matrices,
and generic n-dimensional arrays. We use the name tensors to
refer to them. n is called the rank of the tensor. We show the vector
as a row for readability, but in the text we assume all vectors are
column vectors.

n dimensional array of objects,
typically real-valued
numbers. With apologies to any mathematician reading
us, we call n the rank of the tensor. The notation in the
book varies depending on n:

1. A single-item tensor (n = 0) is just a single value (a
scalar). For scalars, we use lowercase letters, such
as x or y.10

2. Columns of values (n = 1) are vectors. For vectors
we use a lowercase bold font, such as x. The
corresponding row vector is denoted by x⊤ when we
need to distinguish them. We can also ignore the
transpose for readability, if clear from context.

3. Rectangular array of values (n = 2) are matrices.
We use an uppercase bold font, such as X or Y.

4. No specific notation is used for n > 2. We avoid
calligraphic symbols such as (cid:88) , that we reserve for
sets or probability distributions.

10If you are wondering, scalars are named like this because they can
be written as scalar multiples of one. Also, I promise to reduce the
number of footnotes from now on.

10

ScalarVectorMatrixn-dimensional array
Chapter 1: Introduction

11

For working with tensors, we use a variety of indexing
strategies described better in Section 2.1. In most cases,
understanding an algorithm or an operation boils down
to understanding the shape of each tensor involved. To
denote the shape concisely, we use the following notation:

X ∼ (b, h, w, 3)

This is a rank-4 tensor with shape (b, h, w, 3). Some
dimensions can be pre-specified (e.g., 3), while other
dimensions can be denoted by variables. We use the same
symbol to denote drawing from a probability distribution,
e.g., ϵ ∼ (cid:78) (0, 1), but we do this rarely and the meaning
of the symbol should always be clear from context. Hence,
x ∼ (d) will substitute the more common x ∈ (cid:82)d, and
similarly for X ∼ (n, d) instead of X ∈ (cid:82)n×d. Finally, we
may want to constrain the elements of a tensor, for which
we use a special notation:

1. x ∼ Binary(c) denotes a tensor with only binary

values, i.e., elements from the set {0, 1}.

≥ 0 and (cid:80)

2. x ∼ ∆(a) denotes a vector belonging to the so-called
= 1. For tensors with
simplex, i.e., x i
higher rank, e.g., X ∼ ∆(n, c), we assume the
normalization is applied with respect to the last
dimension (e.g., in this case each row of Xi belongs
to the simplex).

i xi

Additional notation is introduced along each chapter when
necessary. We also have a few symbols on the side:

• A bottle to emphasize some definitions. We have
many definitions, especially in the early chapters,
and we use this symbol to visually discriminate the
most important ones.

11


12

Introduction

• A clock for sections we believe crucial to understand

the rest of the book – please do not skip these!

• On the contrary, a teacup for more relaxed sections –
these are generally discursive and mostly optional in
relation to the rest of the book.

Final thoughts before departing

The book stems from my desire to give a coherent form
to my lectures for Neural Networks for Data Science
Applications, a course I teach in the Master Degree in Data
Science at Sapienza University of Rome since many years.
The core chapters of the book constitute the main part of
the course, while the remaining chapters are topics that I
cover on and off depending on the year. Some parts have
been supplemented by additional courses I have taught
(or I intend to teach), including parts of Neural Networks
for Computer Engineering, an introduction to machine
learning for Telecommunication Engineering, plus a few
tutorials, PhD courses, and summer schools over the years.

There are already a number of excellent (and recent) books
on the topic of modern, deep neural networks, including
[Pri23, ZLLS23, BB23, Fle23, HR22]. This book covers a
similar content to all of these in the beginning, while the
exposition and some additional parts (or a few sections
in the advanced chapters) intersect less, and they depend
I hope I can provide
mostly on my research interests.
an additional (and complementary) viewpoint on existing
material.

understanding
As my choice of name suggests,
differentiable programs comes from both theory and
coding: there is a constant interplay between how we

12


Chapter 1: Introduction

13

software

libraries,

design models and how we implement them, with topics
like automatic differentiation being the best example. The
current resurgence of neural networks (roughly from 2012
onwards) can be traced in large part to the availability of
going from Theano
powerful
[ARAA+16] to Caffe, Chainer, and then directly to the
modern iterations of TensorFlow, PyTorch, and JAX,
among others.
I try whenever possible to connect the
from existing programming
discussion to concepts
frameworks, with a focus on PyTorch and JAX. The book is
not a programming manual, however, and I refer to the
documentation of the libraries for a complete introduction
to each of them.

Before moving on, I would like to list a few additional
things this book is not. First, I have tried to pick up a few
concepts that are both (a) common today, and (b) general
enough to be of use in the near future. However, I cannot
foresee the future and I do not strive for completeness,
and several parts of these chapters may be incomplete or
outdated by the time you read them. Second, for each
concept I try to provide a few examples of variations that
exist in the literature (e.g., from batch normalization to
layer normalization). However, keep in mind that
hundreds more exist: I invite you for this to an exploration
of the many pages of Papers With Code. Finally, this is a
book on the fundamental components of differentiable
models, but implementing them at scale (and making
them work) requires both engineering sophistication and
(a bit of) intuition. I cover little on the hardware side, and
for the latter nothing beats experience and opinionated
blog posts.11

11See for example this blog post by A. Karpathy: http://karpathy.
github.io/2019/04/25/recipe/, or his recent Zero to Hero video series:
https://karpathy.ai/zero-to-hero.html.

13


14

Introduction

Acknowledgments

Equations’ coloring is thanks to a beautiful LaTeX package
by ST John.12 Color images of Alice in Wonderland and the
black and white symbols in the margin are all licensed from
Shutterstock.com. The images of Alice in Wonderland in
the figures from the main text are reproductions from the
original John Tenniel illustrations, thanks to Wikimedia. I
thank Roberto Alma for feedback on a previous draft of the
book and for encouraging me to publish the book. I also
thank Corrado Zoccolo, Emanuele Rodolà, Marcin Słaby,
Konstantin Burlachenko, and Diego Sandoval for providing
extensive corrections and suggestions to the current version,
and everyone who sent me feedback via email.

License

The book is released under CC BY-SA license.13 This license
enables “reusers to distribute, remix, adapt, and build upon
the material in any medium or format, so long as attribution
is given to the creator. The license allows for commercial use.
If you remix, adapt, or build upon the material, you must
license the modified material under identical terms”.

12https://github.com/st--/annotate-equations/tree/main
13https://creativecommons.org/licenses/by-sa/4.0/

14


Part I

Compass and needle

“Would you tell me, please, which way I

ought to go from here?”

“That depends a good deal on where

you want to get to,” said the Cat.

“I don’t much care where” said Alice.

“Then it doesn’t matter which way you

go,” said the Cat.

— Chapter 6, Pig and Pepper

15
