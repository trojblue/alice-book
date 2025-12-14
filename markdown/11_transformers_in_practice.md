11 | Transformers

in practice

About this chapter

the basic
We now consider a few variations of
transformer model,
encoder-decoder
architectures, causal MHA layers, and applications to
the image and audio domains.

including

11.1 Encoder-decoder transformers

designed for what

The model we described in Chapter 10 can be used to
perform regression or classification of a given sequence.
However, the original transformer [VSP+17] was a more
complex model,
called
sequence-to-sequence (seq2seq) tasks. In a seq2seq task,
both input and output are sequences, and there is no
trivial correspondence between their tokens. A notable
example is machine translation, in which the output is
the translation of the input sequence in a different
language.

are

265


266

Encoder-decoder transformers

the

input

to a

sequence

One possibility to build a differentiable model for seq2seq
tasks is an encoder-decoder (ED) design [SVL14]. An ED
is composed of two blocks: an encoder that
model
processes
transformed
representation (possibly of a fixed dimensionality), and a
that autoregressively generates the output
decoder
sequence conditioned on the output of the encoder. The
transformer model we described before can be used to
this type for
build the encoder:
classification are called encoder-only transformers.
In
order to build the decoder we need two additional
components: a way to make the model causal (to perform
autoregression), and a way to condition its computation to
a separate input (the output of the encoder).

transformers of

11.1.1 Causal multi-head attention

Let us consider first
the problem of making the
transformer block causal. The only component in which
tokens interact is the MHA block. Hence, having a causal
variant of MHA is enough to make the entire model causal.
Remember that, for convolutions, we designed a causal
variant by appropriately masking the weights in the
convolutional filter. For MHA, we can mask instead all
interactions between tokens that do not satisfy the
causality property:

Masked-SA(X) = softmax

(cid:18) QK⊤⊙ M
(cid:112)

(cid:19)

k

V

It is essential to perform the masking inside the softmax.
Consider the following (wrong) variant:

(cid:18)

Wrong:

softmax

(cid:19)

(cid:18) QK⊤
(cid:112)
k

(cid:19)

⊙ M

V

266


Chapter 11: Transformers in practice

267

Figure F.11.1: Visual depiction of causal attention implemented
with attention masking.

Because of the denominator in the softmax, all tokens
participate in the computation of each token, irrespective
= 0 for non-causal
of masking. Also note that setting Mi j
links does not work, because exp(0) = 1. Hence, the
correct implementation is to select an upper triangular
matrix with −∞ on the upper part, since exp(−∞) = 0
as desired:1

=

Mi j

(cid:168)−∞ if i > j
1

otherwise

Practically, the values can be set to a very large, negative
number instead (e.g., −109).

11.1.2 Cross-attention

Second, let us consider the problem of making the output
of the MHA layer conditional on a separate block of inputs.
To this end, let us rewrite the MHA operation by explicitly
separating the three appearances of the input matrix:

1We are using the weird convention that any finite number
multiplied by −∞ is −∞ to simplify the notation. In practice we
would use a replacing function (e.g., masked_fill in PyTorch) to override
the values, not perform the multiplication.

267

Softmax
268

Encoder-decoder transformers

SA(X1, X2, X3

) = softmax

k X⊤

2

(cid:18) X1WqW⊤
k

(cid:112)

(cid:19)

X3Wv

= X3

= X2

= X (which,
The SA layer corresponds to X1
coincidentally, explains the name we gave to it). However,
the formulation also works if we consider keys, values, and
queries belonging to separate sets. One important case is
cross-attention (CA), in which we assume that the keys
and values are computed from a second matrix Z ∼ (m, e):

Cross-attention between X and Z

CA(X, Z) = softmax





XWqW⊤
(cid:112)

k Z⊤

k



 ZWv

(E.11.1)

such that CA(X, Z) = SA(X, Z, Z). The interpretation is that
the embeddings of X are updated based on their similarity
with a set of external (key, values) pairs provided by Z: we
say that X is cross-attending on Z. Note that this formulation
is very similar to a concatenation of the two sets of input
tokens followed by an appropriate masking of the attention
matrix.

Comparison with feedforward layers

Consider a simplified variant of
the cross-attention
operation in (E.11.1), in which we parameterize explicitly
the keys and values matrices:2

2See also the discussion on the perceiver network in Section 11.2.1.

268


Chapter 11: Transformers in practice

269

NeuralMemory(X) = softmax

(cid:129) XWqK
(cid:112)

(cid:139)

k

V

(E.11.2)

The layer is now parameterized by a query projection
matrix Wq and by the two matrices K and V. (E.11.2) is
called a memory layer [SWF+15], in the sense that rows
of the key and value matrices are used by the model to
store interesting patterns to be retrieved dynamically by
an attention-like operation. If we further simplify the layer
k, and
by setting Wq
replacing the softmax with a generic activation function φ,
we obtain a two-layer MLP:

= I, ignoring the normalization by

(cid:112)

MLP(X) = φ (XK) V

(E.11.3)

Hence, MLPs in transformer networks can be seen as
approximating an attention operation over trainable keys
and values. Visualizing the closest tokens in the training
data shows human-understandable patterns [GSBL20].

11.1.3 The encoder-decoder transformer

With these two components in hand, we are ready to
discuss the original transformer model, shown in Figure
F.11.2.3 First, the input sequence X is processed by a
(called the encoder),
standard transformer model
providing an updated embedding sequence H. Next, the
output sequence is predicted autoregressively by another
transformer model (called the decoder). Differently from

3A pedantic note: technically, Transformer (upper-cased) is a proper
noun in [VSP+17]. In the book, I use transformer (lower-cased) to
refer to any model composed primarily of attention layers.

269


270

Encoder-decoder transformers

Figure F.11.2: Encoder-decoder architecture, adapted from
[VSP+17]. Padded tokens in the decoder are greyed out.

the encoder, the decoder has three components for each
block:

1. A masked variant of the MHA layer (to ensure

autoregression is possible).

2. A cross-attention layer where the queries are given

by the input sequence embedding H.

3. A standard token-wise MLP.

Decoder-only models are also possible, in which case the
second block of the decoder is removed and only masked
MHA and MLPs are used. Most modern LLMs are built by
decoder-only models trained to autoregressively generate
text tokens [RWC+19], as discussed below.
In fact,
encoder-decoder models have become less common with
the realization that many seq2seq tasks can be solved
directly with decoder-only models by concatenating the
input sequence to the generated output sequence, as
described in Section 8.4.3.

270

MHAMLPCausal MHACross MHAMLP[BOS]Input sequenceEncoder block(n times)Encoder outputOutput sequenceDecoder block(m times)Predicted nextoutput token
Chapter 11: Transformers in practice

271

def self_attention(Q: Float[Array, "n k"],
K: Float[Array, "n k"],
V: Float[Array, "n v"]
) -> Float[Array, "n v"]:

return nn.softmax(Q @ K.T) @ V

Box C.11.1: Simple implementation of the SA layer, explicitly
parameterized in terms of the query, key, and value matrices.

11.2 Computational considerations

11.2.1 Time complexity and linear-time

transformers

The MHA performance does not come without a cost:
since every token must attend to all other tokens, its
complexity is higher
than a simpler convolutional
operation. To understand this, we look at its complexity
from two points of view: memory and time. We use a
naive implementation of the SA layer for reference, shown
in Box C.11.1.

Let us look first at the time complexity. The operation inside
the softmax scales as (cid:79) (n2k) because it needs to compute
n2 dot products (one for each pair of tokens). Compare
this to a 1D convolutional layer, which scales only linearly
in the sequence length. Theoretically, this quadratic growth
in complexity can be problematic for very large sequences,
which are common in, e.g., LLMs.

This has led to the development of several strategies for
speeding up autoregressive generation (e.g., speculative
decoding [LKM23]), as well as linear or sub-quadratic
variants of transformers. As an example, we can replace
the SA layer with a cross-attention layer having a trainable

271


272

Computational considerations

set of tokens Z, where the number of tokens can be chosen
as hyper-parameter and controlled by the user. This
strategy was popularized by the Perceiver architecture
[JGB+21] to distill the original set of tokens into smaller
latent bottlenecks. There are many alternative strategies
for designing linearized transformers: we discuss a few
variants in Section 11.3 and Chapter 13.

Importantly, an implementation such as the one in Box
C.11.1 can be shown to be heavily memory-bound on
modern hardware [DFE+22], meaning that its compute
cost is dominated by memory and I/O operations. Hence,
the theoretical gains of linear-time attention variants are
not correlated with actual
speedup on hardware.
Combined with a possible reduction in performance, this
makes them less attractive than a strongly-optimized
implementation of MHA, such as the one described next.

11.2.2 Online softmax

In terms of memory, the implementation in Box C.11.1 has
also a quadratic n2 complexity factor because the attention
matrix QK⊤ is fully materialized during computation.
However, this is unnecessary and this complexity can be
drastically reduced to a linear factor by chunking the
computation in blocks and only performing the softmax
normalization at the end [RS21].

To understand this, consider a single query vector q, and
suppose we split our keys and values into two blocks, which
are loaded in turn in memory:

K =

(cid:152)

(cid:149)K1
K2

, V =

(cid:152)

(cid:149)V1
V2

(E.11.4)

272


Chapter 11: Transformers in practice

273

If we ignore the denominator in the softmax, we can
decompose the SA operation, computing the output for
each chunk in turn:

SA(q, K, V) =

1
+ L2

L1

[h1

+ h2

]

(E.11.5)

where for the two chunks i = 1, 2 we have defined two
auxiliary quantities:

hi
= (cid:88)

Li

= exp (Kiq) Vi

[ exp (Kiq) ]

j

(E.11.6)

(E.11.7)

j

Remember we are loading the chunks in memory
separately, hence for chunk 1 we compute h1 and L1; then
we offload the previous chunk and we compute h2 and L2
for chunk 2.
the operation is not
fully-decomposable unless we keep track of the additional
statistics
the
normalization coefficients of the softmax operation). More
in general, for multiple chunks i = 1, . . . , m we will have:

(which is needed to compute

Note that

Li

SA(q, K, V) =

1
(cid:80)m
i=1 Li

(cid:150) m
(cid:88)

(cid:153)

hi

i=1

(E.11.8)

Hence, we can design a simple iterative algorithm where
for every block of keys and values loaded in memory, we
update and store the cumulative sum of the numerator
and denominator in (E.11.8), only performing the
normalization at the end. This trick (sometimes called
online
IO-aware
implementation and kernel fusion has led to highly
memory- and compute- efficient implementations of

combined with

softmax),

an

273


274

Computational considerations

Figure F.11.3: Official benchmark of FlashAttention and
FlashAttention-2 on an NVIDIA A100 GPU card, reproduced from
https://github.com/Dao-AILab/flash-attention.

attention such as FlashAttention-2.4
Distributed
implementations of attention (e.g., RingAttention
[LZA23]) can also be devised by assigning groups of
queries to different devices and rotating chunks of keys
and queries among the devices. Optimizing the operation
for specific hardware can lead to some counter-intuitive
behaviours, such as increased speed for larger sequence
lengths - see Figure F.11.3.

11.2.3 The KV cache

An important implementative aspect of MHA occurs when
dealing with autoregressive generation in decoder-only
models. For each new token to be generated, only a new
row of the attention matrix and one value token must be
computed, meaning that the previous keys and values can
be stored in memory, as shown in Figure F.11.4. This is

4https://github.com/Dao-AILab/flash-attention

274


Chapter 11: Transformers in practice

275

Figure F.11.4: To
compute masked self-
attention on a new token,
most of the previous
computation can be
reused (in gray). This
is called the KV cache.

called the KV cache and it is a standard in most optimized
implementations of MHA.

The size of the KV cache is linearly increasing in the
sequence length. Once again, you can compare this to an
equivalent implementation of a causal convolutional layer,
where memory is upper-bounded by the size of the
receptive field. Designing expressive layers with a fixed
memory cost in autoregressive generation is a motivating
factor for Chapter 13.

11.2.4 Transformers for images and audio

Transformers were originally developed for text, and they
soon became the default choice for language modeling. In
particular, the popular GPT-2 model [RWC+19] (and later
variants)
is a decoder-only architecture which is
pre-trained by forecasting tokens in text sequences. Most
open-source LLMs, such as LLaMa [TLI+23], follow a
similar architecture. By constrast, BERT [DCLT18] is
another popular family of pre-trained word embeddings
based on an encoder-only architecture trained to predict
masked tokens (masked language modeling). Differently
from GPT-like models, BERT-like models cannot be used to
generate text but only to perform text embedding or as the
first part of a fine-tuned architecture. Encoder-decoder
models for language modeling also exist (e.g., the T5

275

KV Cache
276

Computational considerations

Figure F.11.5: Image tokenization: the image is split into non-
overlapping patches of shape p × p (with p an hyper-parameter).
Then, each patch is flattened and undergoes a further linear
projection to a user-defined embedding size e. c is the number
of channels of the input image.

family [RSR+20]), but they have become less popular.5

From a high-level point of view, a transformer is composed
of three components: a tokenization / embedding step,
which converts the original input into a sequence of
vectors; positional embeddings to encode information
about the ordering of the original sequence; and the
transformer blocks themselves. Hence, transformers for
other types of data can be designed by defining the
appropriate tokenization procedure and positional
embeddings.

Let us consider first computer vision. Tokenizing an image
at the pixel level is too expensive, because of the quadratic
growth in complexity with respect to the sequence length.
The core idea of Vision Transformers (ViTs, [DBK+21]) is
to split the original input into non-overlapping patches of
fixed length, which are then flattened and projected to an
embedding of pre-defined size, as shown in Figure F.11.5.

5Diffusion language models [YTL+25] (DLMs) are a recent
alternative to autoregressive LLMs, and they are trained with a
denoising objective vaguely reminiscent of BERT models. All types
of LLMs undergo several post-training steps beyond token prediction
(e.g., instruction tuning), which we do not have space to cover here.

276

Patch extractionFlatteningProjectionPatch size: p x pEmbedding size: ppcEmbedding size: e
Chapter 11: Transformers in practice

277

from einops import rearrange
# A batch of images
xb = torch.randn((32, 3, 64, 64))

# Define the operation: differently from
# standard einsum, we can split the output
# in blocks using brackets
op = 'b c (h ph) (w pw) \

-> b (h w) (ph pw c)'

# Run the operation with a given patch size
patches = rearrange(xb, op, ph=8, pw=8)
print(patches.shape) # [Out]: (32, 64, 192)

Box C.11.2: einops can be used to decompose an image into
patches with a simple extension of the einsum syntax.

The embedding step in Figure F.11.5 can be achieved with
a convolutional layer, having stride equal to the kernel
size. Alternatively, libraries like einops6 extend the einsum
operation (Section 2.1) to allow for grouping of elements
into blocks of pre-determined shape. An example is shown
in Box C.11.2.

The original ViT used trainable positional embeddings
along with an additional class token to perform image
classification. ViTs can also be used for image generation
by predicting the patches in a row-major or column-major
order. In this case, we can train a separate module that
converts each patch into a discrete set of tokens using, e.g.,
a vector-quantized variational autoencoder [CZJ+22],
or we can work directly with continuous outputs [TEM23].
For image generation, however, other non-autoregressive
approaches such as diffusion models and flow matching
tend to be preferred; we cover them in the next volume.

6http://einops.rocks

277


278

Computational considerations

Figure F.11.6: An example of a bimodal transformer that
operates on both images and text: the outputs of the two tokenizers
are concatenated and sent to the model.

(with pooling) as

By developing proper tokenization mechanisms and
positional embeddings,
transformers have also been
developed for audio, in particular for speech recognition.
In this case, it is common to have a small 1D convolutional
the tokenization block
model
[BZMA20, RKX+23]. For example, Wav2Vec [BZMA20] is
an encoder-only model whose output is trained with an
extension of the cross-entropy loss, called connectionist
temporal classification loss [GFGS06],
to align the
output embeddings to the transcription. Because labeled
data with precise alignments is scarce, Wav2Vec models
are pre-trained on large amounts of unlabeled audio with
a variant of a masked language modeling loss. By contrast,
Whisper [RKX+23] is an encoder-decoder model where
the decoder is trained to autoregressively generate the
transcription. This provides more flexibility to the model
and reduces the need for strongly labeled data, but at the
cost of possible hallucinations in the transcription phase.
Neural audio codecs can also be trained to compress
audio into a sequence of discrete tokens [DCSA23], which
in turn form the basis for generative applications such as
text-to-speech generation [WCW+23].

Transformers can also be defined for time-series [AST+24],

278

"Describe the image"ImagetokenizationText tokenizationAutoregressivetransformer"An illustration of Alice"
Chapter 11: Transformers in practice

279

graphs (covered in the next chapter), and other types of
data. The decoupling between data and architecture is also
the basis for multimodal variants, which can take as input
(or provide as output) different modalities. This is achieved
by tokenizing each modality (image, audio, ...) with its
tokenizer, and concatenating the different tokens together
into a single sequence [BPA+24]. We show an example for
an image-text model in Figure F.11.6.

11.3 Transformer variants

We close the chapter by discussing a few interesting
variation on the basic transformer block. First, several
variants have been devised for very large transformers to
slightly reduce the computational time or parameter’s
count. As an example, parallel blocks [DDM+23] perform
the MLP and MHA operation in parallel:

H = H + MLP(H) + MHA(H)

In this way, the initial and final linear projections in the
MLP and MHA layers can be fused for a more efficient
implementation. As another example, multi-query MHA
[Sha19] shares the same key and value projection matrix
for each head, varying only the queries.

More in general, we can replace the MHA layer with a
simpler (linear complexity in the sequence length)
operation, while keeping the overall structure of the
transformer block, i.e., alternating token and channel
mixing with layer normalization and residual connections.
As an example, suppose the sequence length is fixed (e.g.,
for computer vision, the number of patches can be fixed a
priori). In this case, the MHA layer can be replaced by an
MLP operating on a single input channel, corresponding to

279


280

Transformer variants

Figure F.11.7: Mixer
block, composed of
alternating MLPs on
the rows and columns of
the input matrix.

one dimension of the embedding. This type of model is
called a mixer model [THK+21] - see Figure F.11.7.
Ignoring the normalization operations, this can be written
as alternating MLPs on transpositions of the input matrix:

H = MLP(H) + H
H = (cid:2)MLP(H⊤) + H⊤(cid:3)⊤

(E.11.9)

(E.11.10)

Other variants of the mixer model are also possible using,
e.g., 1D convolutions, Fourier transforms, or pooling. In
particular, in the S2-MLP [YLC+22] model the token mixing
operation is replaced by an even simpler MLP applied on
a shifted version of its input. The general class of such
models has been called MetaFormers by [YLZ+22].

Gated (multiplicative) interactions can also be used in the
composition of the block. In this case, several blocks are
executed in parallel but their output is combined via
Hadamard multiplication. We can write a generic gated
unit as:

f (X) = φ

(X) ⊙ φ

(X)

2

1

(E.11.11)

1 and φ
(X) = σ(XA) and φ

where φ
φ
unit (GLU) described in Section 5.4.

2 are trainable blocks. For example, with
(X) = XB we obtain the gated linear

1

2

the gMLP model
As a few representative examples,
[LDSL21] uses gated units instead of a channel mixing

280

MLPMLP
Chapter 11: Transformers in practice

281

block in a mixer model; the LLaMa family of models
[TLI+23] uses GLU-like units instead of the standard MLP
block; while the gated attention unit (GAU) [HDLL22]
uses a simpler attention-like model having a single head
for φ
2. These designs are
especially popular in some recent variants of recurrent
models, discussed later on in Chapter 13.

1 and a linear projection for φ

To simplify the design even further, the multilinear operator
network (MONet) removes all activation functions to define
a block which is composed only of linear projections and
element-wise multiplications [CCGC24]:

H = E(AX ⊙ BX + DX)

where E is similar to the output projection in the
transformer block, DX acts as a residual connection, and B
is implemented via a low-rank decomposition to reduce
the number of parameters [CCGC24].
In order to
introduce token mixing, a token-shift operation is
implemented in all odd-numbered blocks in the model.

From theory to practice

There are many interesting exercises
this point –
that can be done at
you are almost a master in designing
differentiable models! To begin, using
any image classification dataset, you
can try implementing from scratch a Vision Transformer
as described in Section 11.2.4, following [DBK+21] for
choosing the hyper-parameters. Training a ViT from scratch
on a small dataset is quite challenging [LLS21, SKZ+21],
so be ready for some disappointment unless you have
sufficient computational power to consider million-size

281


282

Transformer variants

datasets. You can also try a simpler variant, such as the
Mixer model described in Section 11.3. All these exercises
should be relatively simple.

1. For tokenizing the image you can use Einops as in
Box C.11.2 or other strategies (e.g., a convolution
with large stride). For small images you can also try
using each pixel as token.

2. For positional embeddings, all strategies described in
Section 10.2 are valid. The simplest one for a ViT is
to initialize a matrix of trainable embeddings, but I
suggest you experiment with sinusoidal and relative
embeddings as practice.

You can also try implementing a small GPT-like model.
There are many sophistications in the tokenization of text
data that we do not cover. However,
the minGPT
repository7 is a fantastic didactic implementation that you
can use as starting point.

7https://github.com/karpathy/minGPT

282
