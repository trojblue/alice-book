Alice’s Adventures in a
differentiable wonderland

A primer on designing neural networks

Vol. I - A tour of the land

Simone Scardapane


2


3

“For, you see, so many out-of-the-way

things had happened lately, that Alice

had begun to think that very few things

indeed were really impossible.”

— Chapter 1, Down the Rabbit-Hole


4


Foreword

This book is an introduction to the topic of (deep) neural
networks, the core technique at the heart of large language
models, generative artificial intelligence - and many other
applications. Because the term neural comes with a lot of
historical baggage, and because neural networks are simply
compositions of differentiable primitives, I refer to them
– when feasible – with the simpler term differentiable
models.

In 2009, I stumbled almost by chance upon a paper by
Yoshua Bengio on the power of ‘deep’ networks [Ben09],
at the same time when automatic differentiation libraries
like Theano [ARAA+16] were becoming popular. Like
Alice, I had stumbled upon a strange programming realm -
a differentiable wonderland where simple things, such as
selecting an element, were incredibly hard, and other
things, such as recognizing cats, were amazingly simple.

I have spent more than ten years reading about,
implementing, and teaching these ideas. This book is a
rough attempt at condensing something of what I have
learned in the process, with a focus on their design and
most common components. Because the field is evolving
quickly, I have tried to strike a good balance between

i


ii

theory and code, historical considerations and recent
trends. I assume the reader has some exposure to machine
learning and linear algebra, but I try to cover the
preliminaries when necessary.

Gather round, friends:
it’s time for our beloved
Alice’s Adventures in a
differentiable wonderland


Contents

Foreword

1 Introduction

I Compass and needle

2 Mathematical preliminaries

2.1 Linear algebra . . . . . . . . . . . . . . . . . .
2.2 Gradients and Jacobians . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
2.3 Gradient descent

3 Datasets and losses

3.1 What is a dataset? . . . . . . . . . . . . . . . .
3.2 Loss functions . . . . . . . . . . . . . . . . . .
3.3 Bayesian learning . . . . . . . . . . . . . . . .

4 Linear models

4.1 Least-squares regression . . . . . . . . . . . .
4.2 Linear models for classification . . . . . . . .
4.3 More on classification . . . . . . . . . . . . .

i

1

15

17
18
32
39

51
51
58
66

71
71
83
90

5 Fully-connected models

101
5.1 The limitations of linear models . . . . . . . 101

iii


5.2 Composition and hidden layers . . . . . . . . 103
5.3 Stochastic optimization . . . . . . . . . . . . 110
5.4 Activation functions . . . . . . . . . . . . . . . 114

6 Automatic differentiation

123
6.1 Problem setup . . . . . . . . . . . . . . . . . . 123
6.2 Forward-mode differentiation . . . . . . . . . 129
6.3 Reverse-mode differentiation . . . . . . . . . 132
6.4 Practical considerations . . . . . . . . . . . . 135

II A strange land

149

7 Convolutional layers

151
7.1 Towards convolutional layers . . . . . . . . . 152
. . . . . . . . . . . . . 163
7.2 Convolutional models

8 Convolutions beyond images

173
8.1 Convolutions for 1D and 3D data . . . . . . 173
8.2 1D and 3D convolutional models . . . . . . . 178
8.3 Forecasting and causal models . . . . . . . . 187
. . . . . . . . . . . . . . . 194
8.4 Generative models

9 Scaling up the models

203
9.1 The ImageNet challenge . . . . . . . . . . . . 203
9.2 Data and training strategies . . . . . . . . . . 206
9.3 Dropout and normalization . . . . . . . . . . 214
9.4 Residual connections . . . . . . . . . . . . . . 228

III Down the rabbit-hole

237

10 Transformer models

239
10.1 Long convolutions and non-local models . . 239
10.2 Positional embeddings . . . . . . . . . . . . . 251
. . . . . . . 258
10.3 Building the transformer model


11 Transformers in practice

265
11.1 Encoder-decoder transformers . . . . . . . . 265
11.2 Computational considerations . . . . . . . . 271
11.3 Transformer variants . . . . . . . . . . . . . . 279

12 Graph models

283
12.1 Learning on graph-based data . . . . . . . . 283
12.2 Graph convolutional layers . . . . . . . . . . 293
12.3 Beyond graph convolutional layers . . . . . 305

13 Recurrent models

315
13.1 Linearized attention models . . . . . . . . . . 315
. . . . . . . . . . . 319
13.2 Classical recurrent layers
13.3 Structured state space models . . . . . . . . 327
13.4 Additional variants . . . . . . . . . . . . . . . 334

A Probability theory

341
A.1 Basic laws of probability . . . . . . . . . . . . 341
A.2 Real-valued distributions
. . . . . . . . . . . 344
A.3 Common distributions . . . . . . . . . . . . . 345
A.4 Moments and expected values . . . . . . . . 346
A.5 Distance between distributions . . . . . . . . 347
A.6 Maximum likelihood estimation . . . . . . . 348

B 1D universal approximation

351
B.1 Approximating a step function . . . . . . . . 352
B.2 Approximating a constant function . . . . . 353
B.3 Approximating a generic function . . . . . . 355


vi

vi
