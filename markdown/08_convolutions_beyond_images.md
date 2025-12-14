8 | Convolutions

beyond images

About this chapter

Convolutional models are a powerful baseline model in
many applications, going far beyond image classification.
In this chapter we provide an overview of several such
extensions, including the use of convolutional layers
for 1D and 3D data, text modeling, and autoregressive
generation. Several of the concepts we introduce (e.g.,
masking, tokenization) are fundamental in the rest of
the book and for understanding modern LLMs.

8.1 Convolutions for 1D and 3D data

8.1.1 Beyond images: time series, audio,

video, text

In the previous chapter we focused exclusively on images.
However, many other types of data share similar
characteristics, i.e., one or more “ordered” dimensions
and one dimension
space,
representing time or

173


174

Convolutions for 1D and 3D data

representing features (the channels in the image case). Let
us consider some examples:

1. Time series are collections of measurements of one
or more processes (e.g., stocks prices, sensor values,
energy flows). We can represent a time series as a
matrix X ∼ (t, c), where t is the length of the time
∼ (c) are the c measurements at time t
series, and Xi
(e.g., c sensors from an EEG scan, or c stock prices).
Each time instant is equivalent to a pixel, and each
measurement is equivalent to a channel.

2. Audio files (speech, music) can also be described by
a matrix X ∼ (t, c), where t is the length of the audio
signal, while c are the channels of the recording (1
for a mono audio, 2 for a stereo signal, etc.).

Frequency-analysis

Audios can also be converted to an image-like
format via frequency analysis (e.g., extracting the
MFCC coefficients over small windows), in which
case the resulting time-frequency images represent
the evolution of the frequency content over the
signal - see Figure F.8.1 for an example. With this
preprocessing we can use standard convolutional
models to process them.

3. Videos can be described by a rank-4 tensor
X ∼ (t, h, w, c), where t is the number of frames of
the video, and each frame is an image of shape
(h, w, c). Another example is a volumetric scan in
medicine, in which case t is the volume depth.

Time series, audio signals, and videos can be described by
their sampling rate, which denotes how many samples

174


Chapter 8: Convolutions beyond images

175

Figure F.8.1: Audio can be represented as either a 1D sequence
(left), or a 2D image in a time-frequency domain (middle). In the
second case, we can apply the same techniques described in the
previous chapter.

are acquired per unit of time, sometimes expressed in
samples per second, or hertz (Hz). For example, classical
EEG units acquire signals at 240 Hz, meaning 240 samples
each second. A stock can be checked every minute,
corresponding to 1/60 Hz. By contrast, audio is acquired
with very high frequency to ensure fidelity: for example,
music can be acquired at 44.1e3 Hz (or 44.1 kHz). Typical
acquisition frame rates for video are instead around 24
frames per second (fps) to ensure smoothness to the
human eye.

Image resolution, audio sampling rate, and video frame
rates all play similar roles in determining the precision with
which a signal is acquired. For an image, we can assume
a fixed resolution a priori (e.g., 1024 × 1024 pixels). This
is reasonable, since images can always be reshaped to a
given resolution while maintaining enough consistency,
except for very small resolutions. By contrast, audio and
video durations can vary from input to input (e.g., a song
of 30 seconds vs. a song of 5 minutes), and they cannot
be reshaped to a common dimension,1 meaning that our

1In the sense of having the same duration and resolution.

175

Feature extraction(e.g., MFCC)ConvolutionalNeural NetworkWindow sizeTimeFrequency
176

Convolutions for 1D and 3D data

datasets will be composed of variable-length data.
In
addition, audio resolution can easily grow very large: with
a 44.1 kHz sampling rate, a 3-minute audio will have ≈ 8M
samples.

We also note that the dimensions in these examples can
be roughly categorized as either “spatial dimensions” (e.g.,
images) or “temporal dimensions” (e.g., audio resolution).
While images can be considered symmetric along their
spatial axes (in many cases, an image flipped along the
width is another valid image), time is asymmetric: an audio
sample inverted on its temporal axis is in general invalid,
and an inverted time series represents a series evolving
from the future towards its past. Apart from exploiting this
aspect in the design of our models (causality), we can also
be interested in predicting future values of the signal: this
is called forecasting.

Finally, consider a text sentence, such as “the cat is on
the table”. There are many ways to split this sentence
into pieces. For example, we can consider its individual
syllables: [”the”, “cat”, “i”, “s”, “on”, “the”, “ta”, ble”]. This
is another example of a sequence, except that each element
of the sequence is now a categorical value (the syllable)
instead of a numerical encoding. Hence, we need some way
of encoding these values into features that can be processed
by the model: splitting a text sequence into components is
called tokenization, while turning each token into a vector
is called embedding the tokens.

these aspects
In the next sections we consider all
(variable-length inputs, causality, forecasting, tokenization,
and embedding) in turn,
to see how we can build
convolutional models to address them. Some of the
techniques we introduce, such as masking, are very
general and are useful also for other types of models, such

176


Chapter 8: Convolutions beyond images

177

as transformers. Other techniques, such as dilated
convolutions, are instead specific to convolutional models.

8.1.2 1D and 3D convolutional layers

Let us consider how to define convolutions for 1D signals
(e.g., time series, audio) and their extension to 3D signals
(e.g., videos). Note that the dimensionality refers only to
the number of dimensions along which we convolve (spatial
or time), and does not include the channel dimension.
Recall that, in the 1D case, we can represent the input as a
single matrix:

Length of the sequence

Features

X ∼ ( t , c )

We now replicate the derivation from Chapter 7. Given
(i) ∼ (s, c) as the
a patch size s = 2k + 1, we define Pk
subset of rows in X at distance at most k from i (ignoring
border elements for which zero-padding can be used). A
1D convolutional layer H = Conv1D(X) outputs a matrix
H ∼ (t, c′), with c′ an hyper-parameter that defines the
output dimensionality, defined row-wise as:

[Conv1D(X )]

= φ(W · vect(Pk

i

(i)) + b)

(E.8.1)

with trainable parameters W ∼ (c′, sc) and b ∼ (c′). Like
in the 2D case, this layer is local (for a properly modified
definition of locality) and equivariant to translations of the
sequence.

In the 2D case, we also discussed an alternative notation
with all indices explicitly summed over:

177


178

1D and 3D convolutional models

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

(E.8.2)

where t(i) = i + k − 1 as in (E.7.6). Recall that we use t
to index i′ and j′ differently for the two tensors: from 1 to
2k + 1 for W , and from i − k to i + k for X . The equivalent
variant for (E.8.1) is obtained trivially by removing one
summation index:

=

Hiz

2k+1
(cid:88)

c
(cid:88)

i′=1

d=1

[W ]

i′,z,d

[X ]

i′+t(i),d

(E.8.3)

where the parameters W ∼ (s, c′, c) are now organized in
a rank-3 tensor. By contrast, the 3D variant is obtained by
adding a new summation over the third dimension with
index p:

Hpi jz

=

2k+1
(cid:88)

2k+1
(cid:88)

2k+1
(cid:88)
,

c
(cid:88)

p′=1

i′=1

j′=1

d=1

[W ]

p′,i′, j′,z,d

[X ]

p′+t(p),i′+t(i), j′+t( j),d

We assume that the kernel size is identical across all
dimensions for simplicity. With similar reasonings we can
derive a vectorized 3D variant of convolution, and also 1D
and 3D variants of max pooling.

8.2 1D and 3D convolutional models

We now consider the design of convolutional models in
the 1D case, with a focus on how to handle variable-length
inputs and how to deal with text sequences. Several of the
ideas we introduce are fairly generic for all differentiable
models.

178


Chapter 8: Convolutions beyond images

179

8.2.1 Dealing with variable-length inputs

their

∼ (t1, c) and X2

Consider two audio files (or two time series, or two texts),
corresponding input matrices
described by
∼ (t2, c). The two inputs share the
X1
same number of channels c (e.g., the number of sensors),
but they have different lengths, t1 and t2. Remember from
our discussion in Section 7.1 that convolutions can handle
(in principle) such variable-length inputs. In fact, denote
by g a generic composition of 1D convolutions and
max-pooling operations, corresponding to the feature
extraction part of the model. The output of the block are
two matrices:

H1

= g(X1

) , H2

= g(X2

)

having the same number of columns but a different
number of rows (depending on how many max-pooling
operations or strided convolutions are applied on the
inputs). After global average pooling, the dependence on
the length disappears:

= (cid:88)

h1

H1i , h2

= (cid:88)

H2i

i

i

and we can proceed with a final classification on the vectors
h1 and h2. However, while this is not a problem at the level
of the model, it is a problem in practice, since mini-batches
cannot be built from matrices of different dimensions, and
thus operations cannot be easily vectorized. This can be
handled by zero-padding the resulting mini-batch to the
maximum dimension across the sequence length. Assuming
> t2, we can
for example, without lack of generality, t1
build a “padded” mini-batch as:

179


180

1D and 3D convolutional models

# Sequences with variable length
# (3, 5, 2, respectively)
X1, X2, X3 = torch.randn(3, 8),
torch.randn(5, 8),
torch.randn(2, 8)

# Pad into a single mini-batch
X = torch.nn.utils.rnn.pad_sequence(

[X1, X2, X3],
batch_first=True)

print(X.shape)
# [Out]: torch.Size([3, 5, 8])

Box C.8.1: A padded mini-batch from three sequences of variable
length (with c = 8). When using a DataLoader, padding can be
achieved by overwriting the default collate_fn, which describes
how the loader concatenates the individual samples.

X = stack

(cid:129)

X1,

(cid:152)(cid:139)

(cid:149)X2
0

where stack operates on a new leading dimension, and the
resulting tensor X has shape (2, t1, c). We can generalize
this to any mini-batch by considering the largest length with
respect to all elements of the mini-batch. For a convolution,
this is not very different from zero-padding, and operating
on the padded input will not influence significantly the
operation (e.g., in audio, zero-padding is equivalent to
adding silence at the end). See Box C.8.1 for an example
of building a padded mini-batch.

Alternatively, we can build a masking matrix describing
valid and invalid indexes in the mini-batched tensor:

(cid:149)

M =

(cid:152)

1t1
0t1

−t2

1t2

180


Chapter 8: Convolutions beyond images

181

where the index denotes the size of the vectors. These
masking matrices can be helpful to avoid invalid operations
on the input tensor.

8.2.2 CNNs for text data

Let us consider now the problem of dealing with text data.
As we mentioned previously, the first step in dealing with
text is tokenization, in which we divide the text (a string)
into a sequence of known symbols (also called tokens in
this context). There are multiple types of tokenizers:

1. Character tokenizer: each character becomes a

symbol.

2. Word tokenizer: each (allowed) word becomes a

symbol.

3. Subword tokenizer:

intermediate between a
character tokenizer and a word tokenizer, each
symbol is possibly larger than a character but also
smaller than a word.

This is shown schematically in Figure F.8.2. In all three
cases, the user has to define a dictionary (vocabulary) of
allowed tokens, such as all ASCII characters for a character
tokenizer. In practice, one can select a desired size of the
dictionary, and then look at the most frequent tokens in the
text to fill it up, with every other symbol going into a special
“out-of-vocabulary” (OOV) token. Subword tokenizers have
many specialized algorithms to this end, such as byte-pair
encoding (BPE) [SKF+99].2

2This is a short exposition focused on differentiable models, and
we are ignoring many preprocessing operations that can be applied to
text, such as removing stop words, punctuation, “stemming”, and so
on. As the size of the models has grown, these operations have become

181


182

1D and 3D convolutional models

Figure F.8.2: Starting from a text, multiple types of tokenizers
are possible. In all cases, symbols are then embedded as vectors
and processed by a generic 1D model.

text can have a wide
Because large collections of
variability, pre-trained subword tokenizers are a standard
choice nowadays. As a concrete example, OpenAI has
released an open-source version of its own tokenizer,3
which is a subword model consisting of approximately
100k subwords (at the time of writing). Consider for
example the encoding of “This is perplexing!” with this
tokenizer, shown in Figure F.8.3. Some tokens correspond
to entire words (e.g., “This”), some to pieces of a word
(e.g, “perplex”), while others to punctuation marks. The
sequence can be equivalently represented by a sequence of
integers:

[2028, 374, 74252, 287, 0]

(E.8.4)

Each integer spans between 0 and the size of the vocabulary
(in this case, roughly 100k), and it uniquely identifies the
token with respect to that vocabulary. In practice, nothing
prevents us from adding “special” tokens to the sequence,
such as tokens representing the beginning of the sentence

less common.

3https://github.com/openai/tiktoken

182

Classify this text!Charactertokenizer['c', 'l', 'a', ..., 't', '!']Sub-wordtokenizer['clas', 'si', ..., 'text']Wordtokenizer['classify', 'this', 'text']EmbeddingNeural network
Chapter 8: Convolutions beyond images

183

Figure F.8.3: Example
of applying the tiktoken
tokenizer to a sentence.

(sometimes denoted as [BOS]), OOV tokens, or anything
else. The [BOS] token will be of special significance in the
next section.

Subword tokenization with very large dictionaries can be
counter-intuitive at times: for example, common digits
such as 52 have their unique token, while digits like 2512
can be split into a “251” token and a “2” token, so that
visualizing the tokenization process is always important to
debug the models’ behaviour.4 Given the importance of the
tokenization step, this is a very active research topic – we
mention here, for example, byte-level tokenizers [PPR+24]
and tokenizers that can be trained end-to-end [HWG25].

After the tokenization step, the tokens must be embedded
into vectors to be used as inputs for a CNN. A simple one-
hot encoding strategy here works poorly, since vocabularies
are large and the resulting vectors would be significantly
sparse. Instead, we have two alternative strategies: the first
is to use pretrained networks that perform the embedding
for us; we will consider this option later on, when we
introduce transformers. In order to build some intuition
for it, we consider here the second alternative, training the
embeddings together with the rest of the network.

Suppose we fix an embedding dimension e as a hyper-
parameter. Since the size n of the dictionary is also fixed,
we can initialize a matrix of embeddings E ∼ (n, e). We now

4For applications where processing numbers
specialized numerical tokenizers can be applied [GPE+23].

is

important,

183


184

1D and 3D convolutional models

Figure F.8.4: A lookup table to convert a sequence of tokens’ IDs
to their curresponding embeddings: the input is a list, the output
is a matrix. The embeddings (shown inside the box) can be trained
together with all the other parameters via gradient descent. We
assume the size of the vocabulary is n = 16.

define a look-up operation that replaces each integer with
the corresponding row in E. Denoting by x the sequence
of IDs we have:

Row x1 in the embedding matrix

LookUp(x) = X =















Ex1

Ex2...

Exm

The resulting input matrix X will have shape (m, e), where
m is the length of the sequence. We can now apply a
generic 1D convolutional model for, e.g., classifying the
text sequence:

ˆy = CNN(X)

This model can be trained in a standard way depending on

184

CheckOutThis...BowlingCheck this outTokenizer[2, 16, 3]1D CNN
Chapter 8: Convolutions beyond images

185

class TextCNN(nn.Module):

def __init__(self, n, e):

super().__init__()
self.emb = nn.Embedding(n, e)
self.conv1 = nn.Conv1d(e, 32, 5,

padding='same')

self.conv2 = nn.Conv1d(32, 64, 5,

padding='same')

self.head = nn.Linear(64, 10)

def forward(self, x):

# (*, m)
# (*, m, e)
# (*, e, m)

x = self.emb(x)
x = x.transpose(1, 2)
x = relu(self.conv1(x)) # (*, 32, m)
# (*, 32, m/2)
x = max_pool1d(x, 2)
x = relu(self.conv2(x)) # (*, 64, m/2)
x = x.mean(2)
return self.head(x)

# (*, 64)
# (*, 10)

Box C.8.2: A 1D CNN with trainable embeddings. n is the size
of the dictionary, e is the size of each embedding. We use two
convolutional layers with 32 and 64 output channels. The shape
of the output for each operation in the forward pass is shown as a
comment.

the task, except that gradient descent will be performed
jointly on the parameters of the model and the embedding
matrix E. This is shown visually in Figure F.8.4, and an
example of model’s definition is given in Box C.8.2.

This idea is extremely powerful, especially because in
many cases we find that the resulting embeddings can be
manipulated algebraically as vectors, e.g., by looking at
the closest embeddings in an Euclidean sense to find
“semantically similar” words or sentences. This idea is at
the core of the use of differentiable models in many
sectors that necessitate retrieval or search of documents.

185


186

1D and 3D convolutional models

Differentiable models and embeddings

Once again, the idea of embedding is very general: any
procedure that converts an object into a vector with
algebraic characteristics is an embedding. For example,
the output of the backbone of a trained CNN after global
pooling can be understood as a high-level embedding of
the input image, and it can be used to retrieve “similar”
images by comparing it to all other embeddings.

8.2.3 Dealing with long sequences

Many of the sequences described before can be very long.
In this case, the locality of convolutional layers can be a
drawback, because we need a linearly increasing number
of layers to process larger and larger receptive fields. We
will see in the next chapters that other classes of models
(e.g., transformers) can be designed to solve this problem.
For now we remain in the realm of convolutions and we
show one interesting solution, called dilated (or atrous,
from the French à trous) convolutions, popularized in the
WaveNet model for speech generation [ODZ+16].

We introduce an additional hyper-parameter called the
dilation rate. A convolution with dilation rate of 1 is a
standard convolution. For a dilation rate of 2, we modify
the convolution operation to select elements for our patch
by skipping one out of two elements in the sequence.
Similarly, for a dilation rate of 4, we skip three elements
over four, etc. We stack convolutional
layers with
exponentially increasing dilation rates, as shown in Figure
F.8.5. The number of parameters does not change, since
the number of neighbors remains constant irrespective of
the dilation rate. However, it is easy to show that the

186


Chapter 8: Convolutions beyond images

187

Figure F.8.5: Convolutional layers with increasing dilation rates.
Elements selected for the convolution are in red, the others are
greyed out. We show the receptive field for a single output element.

resulting receptive field in this case grows exponentially
fast in the number of layers.

8.3 Forecasting and causal models

8.3.1 Forecasting sequences

One important aspect of working with sequences is that
we can build a model to predict future elements, e.g.,
energy prices, turbulence flows, call center occupations,
etc. Predicting tokens is also the fundamental building
block for large language models and other recent
breakthroughs. In a very broad sense, much of the current
excitement around neural networks revolves around the
question of how much a model can be expected to infer
from next-token prediction on large corpora of text, and
how much this setup can be replicated across different
[WFD+23].
modalities (e.g., videos) and dynamics
Formally, predicting the next element of a sequence is
called forecasting in statistics and time series analysis.
From now on, to be consistent with modern literature, we
will use the generic term token to refer to each element of
the sequence, irrespective of whether we are dealing with
an embedded text token or a generic vector-valued input.

187


188

Forecasting and causal models

The reason forecasting is an important problem is that we
can train a forecasting model by just having access to a set
of sequences, with no need for additional target labels: in
modern terms, this is also called a self-supervised learning
task, since the targets can be automatically extracted from
the inputs.

Stationarity and forecasting

Just like text processing, forecasting real-world time
series has a number of associated problems (e.g., the
possible non-stationarity of the time series, trends and
seasonalities) that we do not consider here.a In practice,
audio, text, and many other sequences of interest
can be considered stationary and do not need special
preprocessing. Like for text, for very large forecasting
datasets and correspondingly large models, the impact
of preprocessing tend to diminish [AST+24].

ahttps://filippomb.github.io/python-time-series-

handbook/

To this end, suppose we fix a user-defined length t, and
we extract all possible subsequences of length t from the
dataset (e.g., with t = 12, all consecutive windows of 12
elements, or all sentences composed of 12 tokens, etc.). In
the context of LLMs, the size of the input sequence is called
the context of the model. We associate to each subsequence
a target value which is the next element in the sequence
itself. Thus, we build a set of pairs (X, y), X ∼ (t, c) , y ∼ (c)
and our forecasting model is trained in a supervised way
over this dataset:

f (X) ≈ y

Note that a standard 1D convolutional model can be used
as forecasting model, trained with either mean-squared

188


Chapter 8: Convolutions beyond images

189

error (for continuous time series) or cross-entropy (for
categorical sequences, such as text). While the model is
trained to predict a single step-ahead, we can easily use
it to generate as many steps as we want by what is called
an autoregressive approach, meaning that the model is
predicting (regressing) on its own outputs. Suppose we
predict a single step, (cid:98)y = f (X), and we create a “shifted”
input by adding our predicted value to the input (removing
the first element to avoid exceeding t elements):

Window of t − 1 input elements

X′ =









X2:t
(cid:98)y

(E.8.5)

Predicted value at time t + 1

Forecasting discrete sequences

For a continuous time series this is trivial. For a time
series with discrete values, f will return a probability
vector over the possible values (i.e., possible tokens),
and we can obtain (cid:98)y by taking its arg max, i.e., the
token associated to the highest probability. Alternatively,
we can sample a token proportionally to the predicted
probabilities: see Section 8.4.1.

We can now run f (X′) to generate the next input value in
the sequence, and so on iteratively, by always updating
our buffered input in a FIFO fashion. This approach is
extremely powerful, but it requires us to fix a priori the
input sequence length, which limits its applicability. To
overcome this
limitation, we need only a minor
modification to our models.

189


190

Forecasting and causal models

8.3.2 Causal models

Suppose we only have available a short sequence of 4
elements collected into a matrix X ∼ (4, c), but we have
trained a forecasting model on longer sequences with
t = 6. In order to run the model on the shorter sequence,
we can zero-pad the sequence with two zero vectors 0 at
the beginning, but these will be interpreted by the model
as actual values of the time series unless we mask its
operations. Luckily, there is a simpler and more elegant
approach in the form of causal models.

Definition D.8.1 (Causal layer)

A layer H = f (X) is causal if Hi
), i.e., the output
value corresponding to the i-th element of the sequence
depends only on elements “from its past”.

= f (X:i

A model composed only of causal layers will, of course,
be causal itself. For example, a convolutional layer with
kernel size 1 is causal, since each element is processed
considering only itself. However, a convolutional layer with
kernel size 3 is not causal, since it is processed considering
in addition one element to the left and one element to the
right. We can convert any convolution into a causal variant
by partially zero masking the weights corresponding to
non-causal connections:

= φ (cid:128)(cid:148)

W ⊙ M

(cid:151)

hi

vect(Pk

(i)) + b

(cid:138)

Masked weight matrix

= 0 if the weight corresponds to an element in
where Mi j
the input such that j > i, 1 otherwise. Causal 1D
convolutions can be combined with dilated kernels to

190


Chapter 8: Convolutions beyond images

191

Figure F.8.6: Overview of a 1D causal convolutional layer with
(original) kernel size of 3 and exponentially increasing dilation
rates. Zeroed out connections are removed, and we show the
receptive field for a single output element.

obtain autoregressive models for audio, such as in the
WaveNet model [ODZ+16] - see Figure F.8.6 for an
example.

Masking is easier to understand in the case of a single
channel, in which case M is simply a lower-triangular binary
matrix. The masking operation effectively reduces the
number of parameters from (2k + 1)cc′ to (k + 1)cc′.

By stacking several causal convolutional layers, we can
obtain a causal 1D model variant. Suppose we apply it on
our input sequence, with a model that has no max-pooling
operations. In this case, the output sequence has the same
length as the input sequence:

(cid:98)Y = fcausal

(X)

In addition, any element in the output only depends on
input elements in the same position or preceding it. Hence,
we can define a more sophisticated forecasting model by
predicting a value for each element of the input sequence.

191

Time dimension
192

Forecasting and causal models

(a) Non-causal model

(b) Causal model

Figure F.8.7: Comparison between (a) a non-causal model for
forecasting (predicting only a single element for the entire input
sequence) and (b) a causal model trained to predict one output
element for each input element in the sequence.

Practically, consider now a matrix output defined as:

Y =

(cid:152)

(cid:149)X2:t
y

This is similar to the shifted input from (E.8.5), except
that we are adding the true value as last element of the
sequence. We can train this model by minimizing a loss on
all elements, e.g., a mean-squared error:

l((cid:98)Y, Y) = ∥(cid:98)Y − Y∥2 =

t
(cid:88)

i=1

∥(cid:98)Yi

− Yi

∥2

(E.8.6)

Loss when predicting Xi+1

We simultaneously predict the second element based on
the first one, the third one based on the first two, etc.
For a single input window, we have t separate loss terms,
greatly enhancing the gradient propagation. A comparison
between the two approaches is shown in Figure F.8.7: in

192

Convolutional modelGlobal Average PoolingMSECausal convolutional modelMSE
Chapter 8: Convolutions beyond images

193

Figure F.8.8: Inference with a causal CNN, generating a sequence
step-by-step in an autoregressive way. Unused input tokens are
greyed out. Generated tokens are shown with different colors to
distinguish them.

Figure F.8.7a we show a non-causal convolutional model
trained to predict the next element in the sequence, while
in Figure F.8.7b we show a causal model trained according
to (E.8.6).

More importantly, we can now use the model in an
autoregressive way with any sequence length up to the
maximum length of t. This can be seen easily with an
example. Suppose we have t = 4, and we have observed
two values x1 and x2. We call the model a first time by
zero-padding the sequence to generate the third token:






−
(cid:98)x3
−
−










= f

















x1
x2
0
0

We are ignoring all output values except the second one
(in fact, the third and fourth outputs are invalid due to the
zero-padding). We add (cid:98)x3 to the sequence and continue
calling the model autoregressively (we show in color the
predicted values):

193

Causal modelCausal modelCausal model
194

Generative models






= f






−
−
(cid:98)x4
−











x1
x2
(cid:98)x3
0











,






= f






−
−
−
(cid:98)x5











x1
x2
(cid:98)x3
(cid:98)x4











,






= f






−
−
−
(cid:98)x6











x2
(cid:98)x3
(cid:98)x4
(cid:98)x5











. . .

In the last step we removed one of the original inputs
to keep the constraint on the size of the input. This is
also shown in Figure F.8.8. Note that the model is trained
only on real values, not on its own predictions: this is
called teacher forcing. A variant of teacher forcing is to
progressively replace some of the values in the mini-batches
with values predicted by the model, as training proceeds
and the model becomes more accurate.

Causal autoregressive models are especially interesting in
the case of text sequences (where we only have a single
channel, the index of the tokens), since we can start from
a single [BOS] token representing the beginning of the
sequence and generate text sentences from scratch, or
condition the generation on a specific prompt by the user
which is appended to the [BOS] token.
A similar
reasoning can be applied to audio models to generate
speech or music [ODZ+16].

8.4 Generative models

8.4.1 A probabilistic formulation

is a simple example of a
An autoregressive model
generative model.5 We will talk at length about other

5Remember from Chapter 3 that we assume our supervised pairs
(x, y) come from some unknown probability distribution p(x, y). By
the product rule of probability we can decompose it equivalently as
p( y | x)p(x), or p(x | y)p( y). Any model which approximates p(x)

194


Chapter 8: Convolutions beyond images

195

types of generative models in the next volume. For now,
we provide some insights specific to autoregressive
algorithms. We consider sequences with a single channel
and discrete values, such as text. Autoregressive models
over text tokens are the foundation of LLMs, and they can
be used as the basis for multimodal architectures (Chapter
11).

Generative models are more naturally framed in the
context of probabilities, so we begin by reframing our
previous discussion with a probabilistic formalism. Denote
by (cid:88) the space of all possible sequences (e.g., all possible
combinations of text tokens). In general, many of these
sequences will be invalid, such as the sequence [“tt”, “tt”]
in English. However, even very uncommon sequences may
appear at least once or twice in very large corpora of text
(imagine a character yelling “Scotttt!”).

We can generalize this by considering a probability
distribution p(x) over all possible sequences x ∈ (cid:88) . In
the context of text, this is also called a language model.
Generative modeling is the task of learning to sample
efficiently from this distribution:6

x ∼ p(x)

To see how this connects to our previous discussion, note
that by the product rule of probability we can always

or p(x | y) is called generative, because you can use it to sample new
input points. By contrast, a model that only approximates p( y | x),
like we did in the previous chapters, is called discriminative.

6In this section ∼ is used to denote sampling from a probability

distribution instead of the shape of a tensor.

195


196

Generative models

rewrite p(x) as:

p(x) = (cid:89)

p(x i

| x:i

)

i

(E.8.7)

where we condition each value x i to all preceding values.
If we assume that our model input length is large enough
to accommodate all possible sequences, we can use a
causal forecasting model to parameterize the probability
distribution in (E.8.7):

p(x i

| x:i

) = Categorical(x i

| f (x:i

))

where we use a single, shared model for all time-steps.
Maximum likelihood over this model is then equivalent to
minimizing a cross-entropy loss over the predicted
probabilities, as in Section 4.2.2.

8.4.2 Sampling in an autoregressive model

In general, sampling from a probability distribution is non-
trivial. However, for autoregressive models we can exploit
the product decomposition in (E.8.7) to devise a simple
iterative strategy:

1. Sample x1

∼ p(x1

). This is equivalent to conditioning
| {}). In practice, we always
on the empty set p(x1
condition on an initial fixed token, such as the [BOS]
token, so that our input is never empty.

2. Sample x2

∼ p(x2

) by running again the network
with the value we sampled at step (1), as in Figure
F.8.8.

| x1

3. Sample x3

∼ p(x3

| x1, x2

).

4. Continue until we reach a desired sequence length

196


Chapter 8: Convolutions beyond images

197

or until we get to an end-of-sentence token.

We did this implicitly before by always sampling the
element of highest probability:

x i

= arg max
i

f (x:i

)

However, we can also generalize this by sampling a value
according to the probabilities predicted by f . Remember
(Section 4.2.1) that the softmax can be generalized by
considering an additional temperature parameter. By
varying this parameter during inference, we can vary
smoothly between always taking the argmax value (very
low temperature)
to having an almost uniform
distribution over tokens (very high temperature).

In the context of probabilistic modeling, sampling in this
way from this class of models is called ancestral
sampling, while in the context of language modeling we
sometimes use the term greedy decoding. The use of the
term “greedy” and this brief discussion is enough to
highlight one potential drawback of this approach: while
the product decomposition of p(x) is exact, greedy
decoding is not guaranteed to provide a sample
corresponding to high values of p(x).

To see this, note that f provides an estimate of the
probability for a single token, but the probability of a
sequence is given by a product of many such terms. Hence,
sampling a token with high (local) probability at the
beginning of a sequence may not correspond to a
sequence having large (global) probability as a sentence.
This is easy to visualize if you imagine the choice of the
first token letting the decoding stage being “stuck” in a
low-probability path.

197


198

Generative models

A common mitigation to this problem is beam search (or
beam decoding).
In beam search, in the first step we
sample k different elements (called the beams, with k being
a user-defined parameter). In the second step, for each of
our k beams we sample k possible continuations. Out of
these k2 pairs, we keep only the top-k values in terms of
their product probability p(x1
) (or, equivalently,
their log probability). We continue iteratively in this way
until the end of the sequence.

)p(x2

| x1

is

from our

autoregressive model

Viewed under this lens, sampling the most probable
sequence
a
combinatorial search problem (think of a tree, where for
each token we expand across all possible next tokens, and
so on). From the point of view of computer programming,
beam search is then an example of breadth-first search
over this tree.
In a sense, beam search is trading off a
simple training procedure for a more expensive inference
stage – many other techniques exist to this end, including
the possibility of guiding the decoding to satisfy an
external reward function [WBF+24].

8.4.3 Conditional modeling

As we mentioned earlier,
in general we may not be
interested so much in generating sequences from scratch,
but in generating continuations of known sequences, such
as a user’s question or interaction. This can be formalized
by considering conditional probability distributions in the
form p(x | c), where c is the conditioning argument, such
as a user’s prompt. Our previous discussion extends
almost straightforwardly to this case. For example, the
product decomposition is now written as:

p(x | c) = (cid:89)

p(x i

| x:i, c)

i

198


Chapter 8: Convolutions beyond images

199

where we condition on the previous inputs and the user’s
context. Sampling and decoding are extended in a similar
way.

To perform conditional generation we parameterize p(xi
x:i, c) with a neural network f (x, c) such that:

|

p(xi

| x:i, c) ≈ Categorical(x i

| f (x:i, c))

Hence, the major difference with the unconditional case is
that we need a function f (x, c) having two input arguments
and which satisfies causality in the first argument. When
working with autoregressive models, if both x and c are
texts we can do this easily be considering c as part of the
input sequence and working with a single concatenated
input x ′ = [c∥x]. For example, with the user’s prompt “The
capital of France”, taking for simplicity a word tokenizer
we might have:7

fcausal

([The, capital, of, France]) = is
([The, capital, of, France, is]) = Paris

fcausal

Hence, we can handle unconditional and conditional
modeling simultaneously with a single model.8 In the next
volume we will see other examples of conditional

7We ignore the presence of an end-of-sequence token (EOS) to stop

the autoregressive generation.

8We will see in Chapter 11 that almost any type of data can be
converted into a sequence of tokens. Suppose we are generating
image
a text sequence conditioned on an image prompt (e.g.,
captioning). If both text and images are converted to tokens having
the same embedding size, we can apply an autoregressive model
by concatenating the tokens from the two input types (also called
modalities in this context), where we view the image tokens as the
conditioning set c.

199


200

Generative models

generative models in which more sophisticated strategies
are needed. We will also extend upon this topic when we
discuss decoder-only transformer models in Chapter 11.

200


Chapter 8: Convolutions beyond images

201

From theory to practice

Working with text data is more complex
than image classification, due to many
subtleties involved with tokenization,
data formatting, weird characters, and
variable-length sequences.
PyTorch
has its own text library, torchtext,
which at the time of writing is less
documented than the main library and relies on another
beta library (torchdata) to handle the data pipelines.
Thus, we ignore it here, but we invite you to check it out
on your own.

Hugging Face Datasets is probably the most versatile tool
in this case, as it provides a vast array of datasets and pre-
trained tokenizers, which can be exported immediately to
PyTorch.9 Familiarize yourself a bit with the library before
proceeding with the exercise.

1. Choose a text classification dataset, such as the classic

IMDB dataset.10

2. Tokenize it to obtain a dataset of the form (x, y),
where x is a list of integers as in (E.8.4) and y is the
text label.

3. Build and train a 1D CNN model similar to Box C.8.2.
Experiment a bit with the model’s design to see its
impact on the final accuracy.

PyTorch does not have a quick way to make a 1D
convolution causal, so we will postpone our autoregressive

9See this tutorial
datasets/use_dataset.

for a guide:

https://huggingface.co/docs/

10https://huggingface.co/datasets/stanfordnlp/imdb

201


202

Generative models

experiments for when we introduce transformers.11
Training your own tokenizer is a very good didactic
exercise, although it is far beyond the scope of the book.
For an introduction, you can check this minimalistic BPE
implementation: https://github.com/karpathy/minbpe.

11If you want to try, you can emulate a causal convolution with
proper padding; see Lecture 10.2 here: https://fleuret.org/dlc/. The
entire course is really good if you are looking for streamed lectures.

202
