B | 1D universal

approximation

About this chapter

While formally proving the universal approximation
theorem is beyond the scope of this book, it is helpful
to get an intuitive feeling for how such proofs can be
constructed. In this appendix we follow and extend the
visual intuitions from a 2019 online book chapter by M.
Nielsen,a to which we refer for an extended discussion
(and some interactive visualizations), especially for the
case of multi-dimensional inputs.

ahttp://neuralnetworksanddeeplearning.com/chap4.html

We focus on the original approximation theorem by
Cybenko [Cyb89] which considers models having one
hidden layer with sigmoid activation functions. We also
restrict the analysis to functions with a single input and a
single output, that can be visualized easily. The reasoning
can be extended to other activation functions and to
higher dimensions.

The outline of this visual proof is relatively simple:

351


352

Approximating a step function

1. As a first step, we show how to manually set the
weights of a model with a single neuron in the hidden
layer to approximate a step function.

2. Then, we proceed to show how adding another unit in
the hidden layer allows to approximate any function
which is constant over a small interval, and zero
everywhere else (we call these interval functions “bin”
functions).

3. Finally, we describe a simple procedure to
approximate a generic function by first binning it to
the desired accuracy, and then adding as many
neurons as needed to approximate all bins in turn.
For m bins we obtain a network with 2m neurons.
For a generic function with multiple inputs, this
number would grow exponentially in the number of
dimensions, making the proof non constructive in a
practical case.

B.1 Approximating a step function

To begin, let us consider a single neuron in the hidden
layer, in which case we can write the network’s equation
as (ignoring the output bias term, as it is not helpful in our
derivation):

f (x) = aσ(wx + s)

For the purposes of visualization, we rewrite this by adding
a minus sign on the bias, and we factor the multiplication
term on the entire input of σ (the two variants are clearly
equivalent):

352


Appendix B: 1D universal approximation

353

Figure F.B.1: A network
with a single neuron in
the hidden layer can be
visualized as a sigmoid with
controllable slope, center,
and amplitude. We show
here an example where
we fix the amplitude and
the center, but we vary the
slope.

f (x) = a σ( w (x − s ))

(E.B.1)

Amplitude

Slope

Shift

This is similar to the “tunable” variant of sigmoid we
introduce in Section 5.4. In particular, in this formulation
a controls the amplitude of the sigmoid, w controls the
slope, and s shifts the function by a fixed amount.

We show in Figure F.B.1 several plots of (E.B.1), where we
fix a and s while varying w. As can be seen, by increasing
w the slope gets steeper. Fixing it to a very large constant
(say, w = 104), we are left with a very good approximation
to a step function, of which we can control the location
of the step (the s parameter) and the amplitude (the a
parameter), as shown in Figure F.B.2a.

B.2 Approximating a constant

function

If we add a second neuron with opposite amplitude (and
slightly shifted position), we can approximate a function
which is constant over a small interval (we call it a “bin”

353

sxaaσ(w(x−s))w=0.1w=1w=5
354

Approximating a constant function

(a) 1 neuron

(b) 2 neurons

(c) 4 neurons

Figure F.B.2: (a) A neural network with one input, one hidden
neuron, and one output can approximate any step function (here
shown with a = 1 and s = 0.3). (b) With two hidden neurons and
one output we can approximate any function which is constant over
a small interval. (c) With four neurons, we can approximate any
function which is piecewise constant over two non-zero intervals.
Note that bins can be negative by defining a negative amplitude.

function). Defining a width ∆ we can write:

(cid:130)

(cid:130)

f (x) = aσ

w

x − s −

∆

2

(cid:140)(cid:140)

(cid:130)

(cid:130)

− aσ

w

x − s +

(cid:140)(cid:140)

∆

2

(E.B.2)

Go up [down] at s − ∆
2

Go down [up] at s + ∆
2

where we recall that w is now a large constant, e.g., 104.
(E.B.2) describes a function (equivalent to a model with
one hidden layer having two neurons) which increases
by a at s − ∆
2 , is constant with value f (x) = a over the
interval (cid:2)s − ∆
(cid:3), and then decreases to 0 afterwards.
2 , s + ∆
2
An example is shown in Figure F.B.2b.

For the following, we can rewrite the previous function
as f (x; a, s, ∆) to highlight the dependence on the three
parameters a, s, and ∆.

354

0.00.20.40.60.81.0x0.00.20.40.60.81.0Output0.00.20.40.60.81.0x0.00.20.40.60.8Output0.00.20.40.60.81.0x−0.4−0.20.00.20.40.60.8Output
Appendix B: 1D universal approximation

355

B.3 Approximating a generic

function

Because fa,s,∆(x) is effectively 0 outside the corresponding
two functions defined over non-intersecting
interval,
intervals will not influence each other,
i.e., the “bin”
function we just defined is highly localized. Hence, by
adding two additional neurons in the hidden layer we can
define a function which is constant over two separate
intervals (an example of which is shown in Figure F.B.2c):

f (x) = f (x; a1, s2, ∆

1

) + f (x; a2, s2, ∆

2

)

The rest of the proof is now trivial and proceeds by
binning the function we want to approximate in many
small intervals. Given any (continuous) function g(x)
over an interval (which we assume [0, 1] for simplicity),
we first bin the input domain into m equispaced intervals,
where m controls the accuracy of the approximation (the
higher m, the better the approximation). Hence, the i-th
bin spans the interval:

=

Bi

(cid:149) i
m

−

∆

2

,

i
m

+

∆

(cid:152)

2

where ∆ is the size of each bin. For each bin, we compute
the average value of g(x) inside the interval itself:

(cid:90)

gi

= 1
∆

x∈Bi

g(x)d x

Finally, we define a network with 2m neurons in the hidden

355


356

Approximating a generic function

(a) 5 bins

(b) 15 bins

(c) 50 bins

Figure F.B.3: Approximating g(x) = sin(x)
in [0, 10] with (a)
m = 5, (b) m = 15, and (c) m = 50 bins. The original function is
in red, the approximation (E.B.3) in green. The average squared
error in the three cases decreases exponentially (approximately
0.02, 0.002, and 0.00016).

x

layer, two for each bin. Each bin function is centered in
the bin and takes value gi:

f (x) =

m
(cid:88)

i=1

(cid:18)

f

x; gi ,

(cid:19)

, ∆

i
m

(E.B.3)

(Approximated) constant value

The i-th bin is centered in i
m

We show in Figure F.B.3 an example of such approximation
in the case of g(x) = sin(x)
for increasing number of bins
(m = 5, m = 15, m = 50). It should be clear that the MSE is
inversely proportional to m, and we can decrease the error
as much as desired by simply increasing the resolution of
the approximation.

x

Similar reasonings can be applied to multi-dimensional
inputs and different activation functions.1

1http://neuralnetworksanddeeplearning.com/chap4.html

356

0246810x−0.20.00.20.40.60.81.0Output0246810x−0.20.00.20.40.60.81.0Output0246810x−0.20.00.20.40.60.81.0Output
Bibliography

[AB21]

[ADIP21]

[AHS23]

[AHSB14]

+

[AJB

17]

[AR00]

+
[ARAA

A. N. Angelopoulos and S. Bates. A gentle introduction to conformal
prediction and distribution-free uncertainty quantification. arXiv preprint
arXiv:2107.07511, 2021. 98

A. Apicella, F. Donnarumma, F. Isgrò, and R. Prevete. A survey on modern
trainable activation functions. Neural Networks, 138:14–32, 2021. 118

S. K. Ainsworth, J. Hayase, and S. Srinivasa. Git re-basin: Merging models
modulo permutation symmetries. In ICLR, 2023. 3

F. Agostinelli, M. Hoffman, P. Sadowski, and P. Baldi.
Learning
activation functions to improve deep neural networks. arXiv preprint
arXiv:1412.6830, 2014. 118

D. Arpit, S. Jastrz˛ebski, N. Ballas, D. Krueger, E. Bengio, M. S. Kanwal,
T. Maharaj, A. Fischer, A. Courville, Y. Bengio, et al. A closer look at
memorization in deep networks. In ICML, 2017. 108

J. A. Anderson and E. Rosenfeld. Talking nets: An oral history of neural
networks. MIT Press, 2000. 9

16] R. Al-Rfou, G. Alain, A. Almahairi, C. Angermueller, D. Bahdanau,
N. Ballas, F. Bastien, J. Bayer, A. Belikov, A. Belopolsky, et al. Theano:
A Python framework for fast computation of mathematical expressions.
arXiv preprint arXiv:1605.02688, pages 1–19, 2016. i, 13

[ASA

+

23]

+
[AST

24]

[AZLL19]

[BAY22]

[BB23]

E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou. What learning
algorithm is in-context learning? investigations with linear models. In
ICLR, 2023. 2, 3, 57

A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen,
O. Shchur, S. S. Rangapuram, S. P. Arango, S. Kapoor, et al. Chronos:
Learning the language of time series. arXiv preprint arXiv:2403.07815,
2024. 188, 278

Z. Allen-Zhu, Y. Li, and Y. Liang.
Learning and generalization in
overparameterized neural networks, going beyond two layers. In NeurIPS,
2019. 110

S. Brody, U. Alon, and E. Yahav. How attentive are graph attention
networks? In ICLR, 2022. 307

C. M. Bishop and H. Bishop. Deep learning: Foundations and concepts.
Springer Nature, 2023. 12, 63, 341, 349

357


358

[BBCJ20]

[BBL

+

17]

[BCB15]

+
[BCZ

16]

[Ben09]

[BGHN24]

Bibliography

M. Biesialska, K. Biesialska, and M. R. Costa-Jussa. Continual lifelong
learning in natural language processing: A survey. In COLING, 2020. 57

M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst.
IEEE Signal
Geometric deep learning: going beyond Euclidean data.
Processing Magazine, 34(4):18–42, 2017. 167, 296

D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly
learning to align and translate. In ICLR, 2015. 240

T. Bolukbasi, K.-W. Chang, J. Y. Zou, V. Saligrama, and A. T. Kalai. Man is
to computer programmer as woman is to homemaker? debiasing word
embeddings. In NeurIPS, 2016. 56

Y. Bengio. Learning deep architectures for AI. Foundations and Trends®
in Machine Learning, 2(1):1–127, 2009. i, 8, 109

J. Blasiok, P. Gopalan, L. Hu, and P. Nakkiran. When does optimizing a
proper loss yield calibration? In NeurIPS, 2024. 95

[BGLA21]

F. M. Bianchi, D. Grattarola, L. Livi, and C. Alippi. Graph neural networks
with convolutional ARMA filters. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 44(7):3496–3507, 2021. 298
[BGMMS21] E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell. On the
dangers of stochastic parrots: Can language models be too big? In ACM
FAccT. ACM, 2021. 56

[BGSW18]

[BHB

+

18]

[Bis95]

[Bis06]

[BJMO12]

[BKH16]

[BKK19]

[Ble90]

[BN24]

[BNS06]

[BP20]

+

[BPA

24]

N. Bjorck, C. P. Gomes, B. Selman, and K. Q. Weinberger. Understanding
batch normalization. In NeurIPS, 2018. 224

P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi,
M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, et al.
Relational inductive biases, deep learning, and graph networks. arXiv
preprint arXiv:1806.01261, 2018. 309

C. M. Bishop. Training with noise is equivalent to Tikhonov regularization.
Neural Computation, 7(1):108–116, 1995. 211

C. Bishop. Pattern recognition and machine learning. Springer, 2006. 68,
69, 81, 97, 341, 349

F. Bach, R. Jenatton, J. Mairal, and G. Obozinski. Optimization with
sparsity-inducing penalties. Foundations and Trends® in Machine Learning,
4(1):1–106, 2012. 208

J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. arXiv preprint
arXiv:1607.06450, 2016. 226, 325

S. Bai, J. Z. Kolter, and V. Koltun. Deep equilibrium models. In NeurIPS,
2019. 321

G. E. Blelloch. Prefix sums and their applications. School of Computer
Science, Carnegie Mellon University Pittsburgh, PA, USA, 1990. 330

J. Bernstein and L. Newhouse. Old optimizer, new norm: An anthology.
arXiv preprint arXiv:2409.20325, 2024. 47

M. Belkin, P. Niyogi, and V. Sindhwani. Manifold regularization: A
geometric framework for learning from labeled and unlabeled examples.
Journal of Machine Learning Research, 7(11), 2006. 57, 292, 293

J. Bolte and E. Pauwels.
differentiation in machine learning. In NeurIPS, 2020. 145

A mathematical model for automatic

F. Bordes, R. Y. Pang, A. Ajay, A. C. Li, A. Bardes, S. Petryk, O. Mañas, Z. Lin,
A. Mahmoud, B. Jayaraman, et al. An introduction to vision-language
modeling. arXiv preprint arXiv:2405.17247, 2024. 240, 279

358


Appendix B: Bibliography

359

[BPRS18]

+
[BPS

24]

[BR07]
[BR24]

[BZMA20]

[CCGC24]

[CMMB22]

[CPPM22]

[CRBD18]

A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind. Automatic
differentiation in machine learning: a survey. Journal of Marchine Learning
Research, 18:1–43, 2018. 123

M. Beck, K. Pöppel, M. Spanring, A. Auer, O. Prudnikova, M. Kopp,
G. Klambauer, J. Brandstetter, and S. Hochreiter. xLSTM: Extended long
short-term memory. arXiv preprint arXiv:2405.04517, 2024. 327

V. I. Bogachev and M. A. S. Ruas. Measure theory. Springer, 2007. 344

M. Blondel and V. Roulet. The elements of differentiable programming.
arXiv preprint arXiv:2403.14606, 2024. 8, 39, 124, 135

A. Baevski, Y. Zhou, A. Mohamed, and M. Auli. wav2vec 2.0: A framework
for self-supervised learning of speech representations. In NeurIPS, 2020.
278

Y. Cheng, G. G. Chrysos, M. Georgopoulos, and V. Cevher. Multilinear
operator networks. In ICLR, 2024. 281

F. Cinus, M. Minici, C. Monti, and F. Bonchi. The effect of people
In AAAI ICWSM,
recommenders on echo chambers and polarization.
2022. 53

E. Chien, C. Pan, J. Peng, and O. Milenkovic. You are AllSet: A multiset
function framework for hypergraph neural networks. In ICLR, 2022. 309

R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud. Neural
ordinary differential equations. In NeurIPS, 2018. 233

[CS21]

+
[CVMG

S. Chung and H. Siegelmann. Turing completeness of bounded-precision
recurrent neural networks. In NeurIPS, 2021. 320

14] K. Cho, B. Van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares,
H. Schwenk, and Y. Bengio. Learning phrase representations using rnn
encoder-decoder for statistical machine translation. In EMNLP. ACL, 2014.
327

[CW82]

[Cyb89]

+

[CZJ

22]

[CZSL20]

[DBK

+

21]

[DCLT18]

[DCSA23]

D. Coppersmith and S. Winograd. On the asymptotic complexity of matrix
multiplication. SIAM Journal on Computing, 11(3):472–492, 1982. 27

G. Cybenko. Approximation by superpositions of a sigmoidal function.
Mathematics of Control, Signals and Systems, 2(4):303–314, 1989. 109,
351

H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman. MaskGIT:
Masked generative image transformer. In IEEE/CVF CVPR, 2022. 277
E. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le. RandAugment: Practical
automated data augmentation with a reduced search space. In IEEE/CVF
CVPR Workshops, 2020. 212

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al.
An image is worth 16x16 words: Transformers for image recognition at
scale. In ICLR, 2021. 261, 276, 281

J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of
deep bidirectional transformers for language understanding. In NAACL.
ACL, 2018. 275

A. Défossez, J. Copet, G. Synnaeve, and Y. Adi. High fidelity neural audio
compression. Transactions on Machine Learning Research, 2023. 278

[DDM

+

23] M. Dehghani, J. Djolonga, B. Mustafa, P. Padlewski, J. Heek, J. Gilmer,
A. P. Steiner, M. Caron, R. Geirhos, I. Alabdulmohsin, et al. Scaling vision
transformers to 22 billion parameters. In ICML, 2023. 279

359


360

[DFAG17]

+
[DFE

22]

[DLL

+

22]

[DOMB24]

[DS20]

[DT17]

[DZPS19]

[EHB23]

[FAL17]

[Fle23]
[GBGB21]

[GD23]

[GDE

+

20]

[GFGS06]

[GG16]

[GGR22]

+
[GJG

21]

[GM17]

[GMS05]

[GOV22]

[GPE+23]

Bibliography

Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier. Language modeling with
gated convolutional networks. In ICML, 2017. 119

T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré. FlashAttention: Fast and
memory-efficient exact attention with IO-awareness. In NeurIPS, 2022.
272

V. P. Dwivedi, A. T. Luu, T. Laurent, Y. Bengio, and X. Bresson. Graph
neural networks with learnable structural and positional representations.
In ICLR, 2022. 311, 312

T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski. Vision transformers
need registers. In ICLR, 2024. 262

S. De and S. Smith. Batch normalization biases residual blocks towards
the identity function in deep networks. In NeurIPS, 2020. 229

T. DeVries and G. W. Taylor. Improved regularization of convolutional
neural networks with cutout. arXiv preprint arXiv:1708.04552, 2017. 221

S. S. Du, X. Zhai, B. Poczos, and A. Singh. Gradient descent provably
optimizes over-parameterized neural networks. In ICLR, 2019. 110
F. Eijkelboom, R. Hesselink, and E. J. Bekkers. E (n) equivariant message
passing simplicial networks. In ICML, 2023. 309

C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast
adaptation of deep networks. In ICML, 2017. 57

F. Fleuret. The Little Book of Deep Learning. Lulu Press, Inc., 2023. 12

D. J. Gauthier, E. Bollt, A. Griffith, and W. A. Barbosa. Next generation
reservoir computing. Nature Communications, 12(1):5564, 2021. 322

A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective
state spaces. arXiv preprint arXiv:2312.00752, 2023. 337, 338

A. Gu, T. Dao, S. Ermon, A. Rudra, and C. Ré. Hippo: Recurrent memory
with optimal polynomial projections. In NeurIPS, 2020. 328

A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber. Connectionist
temporal classification:
labelling unsegmented sequence data with
recurrent neural networks. In ICML, 2006. 278

Y. Gal and Z. Ghahramani. Dropout as a bayesian approximation:
Representing model uncertainty in deep learning. In ICML, 2016. 219

A. Gu, K. Goel, and C. Ré. Efficiently modeling long sequences with
structured state spaces. In ICLR, 2022. 328, 332

A. Gu, I. Johnson, K. Goel, K. Saab, T. Dao, A. Rudra, and C. Ré. Combining
recurrent, convolutional, and continuous-time models with linear state
space layers. In NeurIPS, 2021. 329, 338

C. Gallicchio and A. Micheli. Echo state property of deep reservoir
computing networks. Cognitive Computation, 9:337–350, 2017. 325

M. Gori, G. Monfardini, and F. Scarselli. A new model for learning in
graph domains. In IEEE IJCNN. IEEE, 2005. 321

L. Grinsztajn, E. Oyallon, and G. Varoquaux. Why do tree-based models
still outperform deep learning on typical tabular data? In NeurIPS, 2022.
151

S. Golkar, M. Pettee, M. Eickenberg, A. Bietti, M. Cranmer, G. Krawezik,
F. Lanusse, M. McCabe, R. Ohana, L. Parker, et al.
xVal: A
continuous number encoding for large language models. arXiv preprint
arXiv:2310.02989, 2023. 183

360


Appendix B: Bibliography

361

[GPSW17]

[Gri12]

[GSBL20]

+
[GSR

17]

[GW08]

C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern
neural networks. In ICML, 2017. 98

A. Griewank. Who invented the reverse mode of differentiation?
Documenta Mathematica, Extra Volume ISMP, 389400, 2012. 124

M. Geva, R. Schuster, J. Berant, and O. Levy. Transformer feed-forward
layers are key-value memories. In EMNLP. ACL, 2020. 269

J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural
message passing for quantum chemistry. In ICML, 2017. 308

A. Griewank and A. Walther. Evaluating derivatives: principles and
techniques of algorithmic differentiation. SIAM, 2008. 124, 139

[GWFM+13] I. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio.

Maxout networks. In ICML, 2013. 119

[GZBA22]

D. Grattarola, D. Zambon, F. M. Bianchi, and C. Alippi. Understanding
pooling in graph neural networks. IEEE Transactions on Neural Networks
and Learning Systems, 2022. 298

[HABN

+

21] T. Hoefler, D. Alistarh, T. Ben-Nun, N. Dryden, and A. Peste. Sparsity in
deep learning: Pruning and growth for efficient inference and training in
neural networks. Journal of Machine Learning Research, 22(241):1–124,
2021. 208

+
[HBE

24]

[HDLL22]

[HG16]

A. Ho, T. Besiroglu, E. Erdil, D. Owen, R. Rahman, Z. C. Guo, D. Atkinson,
N. Thompson, and J. Sevilla. Algorithmic progress in language models.
arXiv preprint arXiv:1710.05941, 2024. 1, 2

W. Hua, Z. Dai, H. Liu, and Q. Le. Transformer quality in linear time. In
ICML, 2022. 281

D. Hendrycks and K. Gimpel. Gaussian error linear units (GELUs). arXiv
preprint arXiv:1606.08415, 2016. 117

[HHWW14] P. Huang, Y. Huang, W. Wang, and L. Wang. Deep embedding network for

clustering. In ICPR. IEEE, 2014. 55

[Hoc98]

[Hor91]

[HR22]

[HS97]

[HSS08]

[HTF09]

[HWG25]

[HYL17]

+
[HZC

17]

S. Hochreiter. Recurrent neural net learning and vanishing gradient.
International Journal Of Uncertainity, Fuzziness and Knowledge-Based
Systems, 6(2):107–116, 1998. 325

K. Hornik. Approximation capabilities of multilayer feedforward networks.
Neural Networks, 4(2):251–257, 1991. 109

M. Hardt and B. Recht. Patterns, predictions, and actions: Foundations of
machine learning. Princeton University Press, 2022. 12

S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural
Computation, 9(8):1735–1780, 1997. 327

T. Hofmann, B. Schölkopf, and A. J. Smola. Kernel methods in machine
learning. The Annals of Statistics, 36(3):1171–1220, 2008. 108, 316, 317

T. Hastie, R. Tibshirani, and J. H. Friedman. The elements of statistical
learning: data mining, inference, and prediction. Springer, 2009. 55, 95

S. Hwang, B. Wang, and A. Gu. Dynamic chunking for end-to-end
hierarchical sequence modeling. arXiv preprint arXiv:2507.07955, 2025.
183

W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning
on large graphs. In NeurIPS, 2017. 304

A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand,
M. Andreetto, and H. Adam. MobileNets: Efficient convolutional neural

361


362

[HZRS15]

[HZRS16]

[ICS22]

[IS15]

[JGB

+

21]

+
[JK

17]

+
[JLB

22]

[KB15]

[KG21]

[KL18]

[KMH

+

20]

+
[KPR

17]

[KSH12]

[KVPF20]

[KW17]

[Lau19]

[LBBH98]

[LBH15]

Bibliography

networks for mobile vision applications. arXiv preprint arXiv:1704.04861,
2017. 170

K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
Surpassing human-level performance on ImageNet classification. In IEEE
ICCV, 2015. 116

K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image
recognition. In IEEE/CVF CVPR, 2016. 229, 230, 231, 234
K. Irie, R. Csordás, and J. Schmidhuber. The dual form of neural
networks revisited: Connecting test time predictions to training patterns
via spotlights of attention. In ICML, 2022. 79

S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network
training by reducing internal covariate shift. In ICML, 2015. 221

A. Jaegle, F. Gimeno, A. Brock, O. Vinyals, A. Zisserman, and J. Carreira.
Perceiver: General perception with iterative attention. In ICML, 2021.
272

P. Jain, P. Kar, et al. Non-convex optimization for machine learning.
Foundations and Trends® in Machine Learning, 10(3-4):142–363, 2017.
44

L. V. Jospin, H. Laga, F. Boussaid, W. Buntine, and M. Bennamoun. Hands-
on Bayesian neural networks—a tutorial for deep learning users. IEEE
Computational Intelligence Magazine, 17(2):29–48, 2022. 66, 68

D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In
ICLR, 2015. 46

P. Kidger and C. Garcia.
Equinox: neural networks in JAX via
callable PyTrees and filtered transformations. Differentiable Programming
Workshop, NeurIPS, 2021. 121

S. M. Kakade and J. D. Lee. Provably correct automatic sub-differentiation
for qualified programs. In NeurIPS, 2018. 145

J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child,
S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural
language models. arXiv preprint arXiv:2001.08361, 2020. 1, 240

J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins,
A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al.
Overcoming catastrophic forgetting in neural networks. Proceedings of the
National Academy of Sciences, 114(13):3521–3526, 2017. 69

A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with
deep convolutional neural networks. In NeurIPS, 2012. 115, 204

A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are
RNNs: Fast autoregressive transformers with linear attention. In ICML,
2020. 315, 317

T. N. Kipf and M. Welling. Semi-supervised classification with graph
convolutional networks. In ICLR, 2017. 295

S. Laue. On the equivalence of automatic and symbolic differentiation.
arXiv preprint arXiv:1904.02990, 2019. 129

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning
applied to document recognition. Proceedings of the IEEE, 86(11):2278–
2324, 1998. 9

Y. LeCun, Y. Bengio, and G. Hinton.
521(7553):436–444, 2015. 8

Deep learning.

Nature,

362


Appendix B: Bibliography

363

[LCX+23]

[LDR23]

[LDSL21]

[LH19]

[Lim21]
[LJ09]

[LKM23]

[LLLG22]

[LLS21]

+
[LMW

22]

[LPW

+

17]

+
[LRZ

23]

[LTM

+

22]

+
[LWV

24]

[LZA23]

[MCT

+]

[Met22]

[MGMR24]

+
[MKS

20]

[MRF+19]

J. Li, Y. Cheng, Z. Xia, Y. Mo, and G. Huang. Generalized activation via
multivariate projection. arXiv preprint arXiv:2309.17194, 2023. 118

V. Lialin, V. Deshpande, and A. Rumshisky. Scaling down to scale up: A
guide to parameter-efficient fine-tuning. arXiv preprint arXiv:2303.15647,
2023. 57

H. Liu, Z. Dai, D. So, and Q. V. Le. Pay attention to MLPs. In NeurIPS,
2021. 280

I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In
ICLR, 2019. 47, 208

L.-H. Lim. Tensors in computations. Acta Numerica, 30:555–764, 2021. 9

M. Lukoševiˇcius and H. Jaeger. Reservoir computing approaches to
recurrent neural network training. Computer Science Review, 3(3):127–
149, 2009. 322

Y. Leviathan, M. Kalman, and Y. Matias. Fast inference from transformers
via speculative decoding. In ICML, 2023. 271

Y. Li, B. Lin, B. Luo, and N. Gui. Graph representation learning
beyond node and homophily. IEEE Transactions on Knowledge and Data
Engineering, 35(5):4880–4893, 2022. 305

S. H. Lee, S. Lee, and B. C. Song. Vision transformer for small-size datasets.
arXiv preprint arXiv:2112.13492, 2021. 281

Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A ConvNet
for the 2020s. In IEEE/CVF CVPR, 2022. 231, 234
Z. Lu, H. Pu, F. Wang, Z. Hu, and L. Wang. The expressive power of neural
networks: A view from the width. In NeurIPS, 2017. 110

D. Lim, J. Robinson, L. Zhao, T. Smidt, S. Sra, H. Maron, and S. Jegelka.
Sign and basis invariant networks for spectral graph representation
learning. In ICLR, 2023. 312

H. Liu, D. Tam, M. Muqeeth, J. Mohta, T. Huang, M. Bansal, and C. A.
Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper
than in-context learning. In NeurIPS, 2022. 3

Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljaˇci´c, T. Y. Hou,
and M. Tegmark. KAN: Kolmogorov-Arnold networks. arXiv preprint
arXiv:2404.19756, 2024. 119

H. Liu, M. Zaharia, and P. Abbeel. Ring attention with blockwise
transformers for near-infinite context. In Foundation Models for Decision
Making Workshop, NeurIPS, 2023. 274

H. Mao, Z. Chen, W. Tang, J. Zhao, Y. Ma, T. Zhao, N. Shah, M. Galkin,
and J. Tang. Position: Graph foundation models are already here. In
ICML. 312

C. Metz. Genius makers: the mavericks who brought AI to Google, Facebook,
and the world. Penguin, 2022. 8

L. Müller, M. Galkin, C. Morris, and L. Rampášek. Attending to graph
transformers. Transactions on Machine Learning Research, 2024. 310, 312

J. Mukhoti, V. Kulharia, A. Sanyal, S. Golodetz, P. Torr, and P. Dokania.
Calibrating deep neural networks using focal loss. In NeurIPS, 2020. 98

C. Morris, M. Ritzert, M. Fey, W. L. Hamilton, J. E. Lenssen, G. Rattan, and
M. Grohe. Weisfeiler and Leman go neural: Higher-order graph neural
networks. In AAAI Conference on Artificial Intelligence, volume 33, pages
4602–4609, 2019. 308

363


364

[MRT18]

[MSC

+

13]

[MZBG18]

[NCN

+

23]

[ODG

+

23]

+
[ODZ

16]

+
[OSG

23]

+
[PAA

23]

Bibliography

M. Mohri, A. Rostamizadeh, and A. Talwalkar. Foundations of machine
learning. MIT Press, 2018. 60

T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed
representations of words and phrases and their compositionality.
In
NeurIPS, 2013. 56

G. Marra, D. Zanca, A. Betti, and M. Gori. Learning neuron non-linearities
with kernel-based deep neural networks. arXiv preprint arXiv:1807.06302,
2018. 118

V. Niculae, C. F. Corro, N. Nangia, T. Mihaylova, and A. F. Martins. Discrete
latent structure in neural networks. arXiv preprint arXiv:2301.07473,
2023. 125

A. Orvieto, S. De, C. Gulcehre, R. Pascanu, and S. L. Smith. On the
universality of linear recurrences followed by nonlinear projections. In
HLD 2023 Workshop, ICML, 2023. 328

A. v. d. Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves,
N. Kalchbrenner, A. Senior, and K. Kavukcuoglu. WaveNet: A generative
model for raw audio. In ISCA SSW Workshop, 2016. 186, 191, 194

A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and
S. De. Resurrecting recurrent neural networks for long sequences. In
ICML, 2023. 328, 333

B. Peng, E. Alcaide, Q. Anthony, A. Albalak, S. Arcadinho, H. Cao, X. Cheng,
M. Chung, M. Grella, K. K. GV, et al. RWKV: Reinventing RNNs for the
transformer era. In EMNLP. ACL, 2023. 335, 336

+
[PABH

21] O. Puny, M. Atzmon, H. Ben-Hamu, I. Misra, A. Grover, E. J. Smith, and
Y. Lipman. Frame averaging for invariant and equivariant network design.
arXiv preprint arXiv:2110.03336, 2021. 167

+
[PBE

22]

[PBL20]

[PGCB14]

[PKP

+

19]

[PNR

+

21]

[PP08]

[PPR

+

24]

[PPVF21]

A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra. Grokking:
Generalization beyond overfitting on small algorithmic datasets. In 1st
Mathematical Reasoning in General Artificial Intelligence Workshop, ICLR,
2022. 3, 210

T. Poggio, A. Banburski, and Q. Liao. Theoretical issues in deep networks.
Proceedings of the National Academy of Sciences, 117(48):30039–30045,
2020. 60

R. Pascanu, C. Gulcehre, K. Cho, and Y. Bengio. How to construct deep
recurrent neural networks. In ICLR, 2014. 320

G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter. Continual
lifelong learning with neural networks: A review. Neural Networks,
113:54–71, 2019. 57

G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, and
B. Lakshminarayanan. Normalizing flows for probabilistic modeling and
inference. Journal of Machine Learning Research, 22(57):1–64, 2021. 233

K. B. Petersen and M. S. Pedersen. The matrix cookbook. Technical
University of Denmark, 2008. 36

A. Pagnoni, R. Pasunuru, P. Rodriguez, J. Nguyen, B. Muller, M. Li, C. Zhou,
L. Yu, J. Weston, L. Zettlemoyer, et al. Byte latent transformer: Patches
scale better than tokens. arXiv preprint arXiv:2412.09871, 2024. 183

S. Pesme, L. Pillaud-Vivien, and N. Flammarion. Implicit bias of SGD for
diagonal linear networks: a provable benefit of stochasticity. In NeurIPS,
2021. 108

364


Appendix B: Bibliography

365

[PRCB24]

[Pri23]
+
03]
[PS

[PSL22]

+
[QPF

24]

[RBOB18]

+
[RGD

22]

[RHM86]

[RKG+22]

[RKX

+

23]

[RM22]

[RS21]

[RSR

+

20]

[RWC+19]

[RZL17]

+
[SAL

24]

[Sch15]

[SCHU17]

A. Patel, C. Raffel, and C. Callison-Burch. Datadreamer: A tool for
In ACL.
synthetic data generation and reproducible LLM workflows.
ACL, 2024. 211

S. J. Prince. Understanding Deep Learning. MIT Press, 2023. 12

T. Poggio, S. Smale, et al. The mathematics of learning: Dealing with
data. Notices of the AMS, 50(5):537–544, 2003. 60

O. Press, N. A. Smith, and M. Lewis. Train short, test long: Attention with
linear biases enables input length extrapolation. In ICLR, 2022. 258

S. Qiu, A. Potapczynski, M. Finzi, M. Goldblum, and A. G. Wilson. Compute
better spent: Replacing dense layers with structured matrices. arXiv
preprint arXiv:2406.06248, 2024. 161

M. Ravanelli, P. Brakel, M. Omologo, and Y. Bengio. Light gated recurrent
IEEE Transactions on Emerging Topics in
units for speech recognition.
Computational Intelligence, 2(2):92–102, 2018. 326

L. Rampášek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and D. Beaini.
Recipe for a general, powerful, scalable graph transformer. In NeurIPS,
2022. 310

D. E. Rumelhart, G. E. Hinton, and J. L. McClelland. A general framework
for parallel distributed processing. In Parallel Distributed Processing Volume
1. MIT Press, 1986. 8

D. W. Romero, D. M. Knigge, A. Gu, E. J. Bekkers, E. Gavves, J. M. Tomczak,
and M. Hoogendoorn. Towards a general purpose CNN for long range
dependencies in nD. arXiv preprint arXiv:2206.03398, 2022. 242

A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever.
Robust speech recognition via large-scale weak supervision. In ICML,
2023. 278

J. W. Rocks and P. Mehta. Memorizing without overfitting: Bias, variance,
and interpolation in overparameterized models. Physical Review Research,
4(1):013201, 2022. 210
M. N. Rabe and C. Staats. Self-attention does not need (cid:79) (n2) memory.
arXiv preprint arXiv:2112.05682, 2021. 272

C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou,
W. Li, and P. J. Liu. Exploring the limits of transfer learning with a
unified text-to-text transformer. The Journal of Machine Learning Research,
21(1):5485–5551, 2020. 276

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al.
Language models are unsupervised multitask learners. OpenAI blog, 2019.
55, 270, 275

P. Ramachandran, B. Zoph, and Q. V. Le. Searching for activation functions.
arXiv preprint arXiv:1710.05941, 2017. 117

Roformer:
J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu.
Enhanced transformer with rotary position embedding. Neurocomputing,
568:127063, 2024. 258

J. Schmidhuber. Deep learning in neural networks: An overview. Neural
Networks, 61:85–117, 2015. 327

S. Scardapane, D. Comminiello, A. Hussain, and A. Uncini. Group sparse
regularization for deep neural networks. Neurocomputing, 241:81–89,
2017. 208

365


366

[SGT+08]

[Sha19]

[Sha20]

[SHK

+

14]

[SHW21]

+

[SKF

99]

[SKZ

+

21]

+
[SLJ

15]

[SMDH13]

[SP97]

[SSBD14]

[Sti81]

[SVL14]

[SVVTU19]

[SW17]

+
[SWF

15]

[SWL23]

+
[TCB

24]

[TEM23]

[TGJ+15]

Bibliography

F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini.
The graph neural network model. IEEE Transactions on Neural Networks,
20(1):61–80, 2008. 321

N. Shazeer. Fast transformer decoding: One write-head is all you need.
arXiv preprint arXiv:1911.02150, 2019. 279

N. Shazeer.
arXiv:2002.05202, 2020. 119

GLU variants improve transformer.

arXiv preprint

N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov.
Dropout: a simple way to prevent neural networks from overfitting. The
Journal of Machine Learning Research, 15(1):1929–1958, 2014. 215

V. G. Satorras, E. Hoogeboom, and M. Welling. E(n) equivariant graph
neural networks. In ICML, 2021. 309

Y. Shibata, T. Kida, S. Fukamachi, M. Takeda, A. Shinohara, T. Shinohara,
and S. Arikawa. Byte pair encoding: A text compression scheme that
accelerates pattern matching. 1999. 181

A. Steiner, A. Kolesnikov, X. Zhai, R. Wightman, J. Uszkoreit, and L. Beyer.
How to train your ViT? data, augmentation, and regularization in vision
transformers. Transactions on Machine Learning Researc, 2021. 281

C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan,
V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In
IEEE CVPR, 2015. 166

I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of
initialization and momentum in deep learning. In ICML, 2013. 46

M. Schuster and K. K. Paliwal. Bidirectional recurrent neural networks.
IEEE Transactions on Signal Processing, 45(11):2673–2681, 1997. 320

S. Shalev-Shwartz and S. Ben-David. Understanding machine learning:
From theory to algorithms. Cambridge University Press, 2014. 60

S. M. Stigler. Gauss and the invention of least squares. The Annals of
Statistics, pages 465–474, 1981. 8

I. Sutskever, O. Vinyals, and Q. V. Le. Sequence to sequence learning with
neural networks. In NeurIPS, 2014. 266

S. Scardapane, S. Van Vaerenbergh, S. Totaro, and A. Uncini. Kafnets:
Kernel-based non-parametric activation functions for neural networks.
Neural Networks, 110:19–32, 2019. 118

S. Scardapane and D. Wang. Randomness in neural networks: an overview.
Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery,
7(2):e1200, 2017. 317

S. Sukhbaatar, J. Weston, R. Fergus, et al. End-to-end memory networks.
In NeurIPS, 2015. 269

J. T. Smith, A. Warrington, and S. W. Linderman. Simplified state space
layers for sequence modeling. In ICLR, 2023. 328, 330, 331

M. Tiezzi, M. Casoni, A. Betti, M. Gori, and S. Melacci. State-space
modeling in long sequence processing: A survey on recurrence in the
transformer era. arXiv preprint arXiv:2406.09062, 2024. 315

M. Tschannen, C. Eastwood, and F. Mentzer. GIVT: Generative infinite-
vocabulary transformers. arXiv preprint arXiv:2312.02116, 2023. 277

J. Tompson, R. Goroshin, A. Jain, Y. LeCun, and C. Bregler. Efficient object
localization using convolutional networks. In IEEE/CVF CVPR, 2015. 221

366


Appendix B: Bibliography

367

[THK+21]

+

[TLI

23]

[TNHA24]

[Unc15]

[Vap13]

[VCC

+

18]

[Vel22]

+
[VSP

17]

[VWB16]

[WBF

+

24]

I. O. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai, T. Unterthiner,
J. Yung, A. Steiner, D. Keysers, J. Uszkoreit, et al. MLP-Mixer: An all-MLP
architecture for vision. In NeurIPS, 2021. 280

H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix,
B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient
foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3,
56, 275, 281

D. Teney, A. M. Nicolicioiu, V. Hartmann, and E. Abbasnejad. Neural
redshift: Random networks are not random functions. In IEEE/CVF CVPR,
2024. 108

A. Uncini. Fundamentals of adaptive signal processing. Springer, 2015.
162

V. Vapnik. The nature of statistical learning theory. Springer Science &
Business Media, 2013. 60

P. Veliˇckovi´c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio.
Graph attention networks. In ICLR, 2018. 306

P. Veliˇckovi´c. Message passing all the way up.
arXiv:2202.11097, 2022. 308

arXiv preprint

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin. Attention is all you need. In NeurIPS, 2017.
239, 240, 255, 256, 257, 265, 269, 270

A. Veit, M. J. Wilber, and S. Belongie. Residual networks behave like
ensembles of relatively shallow networks. In NeurIPS, 2016. 232

S. Welleck, A. Bertsch, M. Finlayson, H. Schoelkopf, A. Xie, G. Neubig,
I. Kulikov, and Z. Harchaoui.
From decoding to meta-generation:
Inference-time algorithms for large language models. arXiv preprint
arXiv:2406.16838, 2024. 198

[WCW

+

23] C. Wang, S. Chen, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu,
H. Wang, J. Li, et al. Neural codec language models are zero-shot text to
speech synthesizers. arXiv preprint arXiv:2301.02111, 2023. 278

[WFD

+

23]

[WGGH18]

[WJ21]

[WZZ

+

13]

+
[XYH

20]

[YCC+24]

[YHO

+

19]

H. Wang, T. Fu, Y. Du, W. Gao, K. Huang, Z. Liu, P. Chandak, S. Liu,
P. Van Katwyk, A. Deac, et al. Scientific discovery in the age of artificial
intelligence. Nature, 620(7972):47–60, 2023. 1, 187

X. Wang, R. Girshick, A. Gupta, and K. He. Non-local neural networks. In
IEEE/CVF CVPR, 2018. 243

Y. Wu and J. Johnson. Rethinking "Batch" in BatchNorm. arXiv preprint
arXiv:2105.07576, 2021. 225

L. Wan, M. Zeiler, S. Zhang, Y. Le Cun, and R. Fergus. Regularization of
neural networks using DropConnect. In ICML, 2013. 221

R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang,
Y. Lan, L. Wang, and T. Liu. On layer normalization in the transformer
architecture. In ICML, 2020. 259

L. Yuan, Y. Chen, G. Cui, H. Gao, F. Zou, X. Cheng, H. Ji, Z. Liu, and
M. Sun. Revisiting out-of-distribution robustness in NLP: Benchmarks,
analysis, and llms evaluations. In NeurIPS, 2024. 53

S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo. CutMix:
Regularization strategy to train strong classifiers with localizable features.
In IEEE/CVF ICCV, 2019. 213

367


368

[YLC+22]

+
[YLZ

22]

+
[YTL

25]

[YYZ17]

[ZBH

+

21]

Bibliography

T. Yu, X. Li, Y. Cai, M. Sun, and P. Li. S2-MLP: Spatial-shift MLP architecture
for vision. In WACV, 2022. 280

W. Yu, M. Luo, P. Zhou, C. Si, Y. Zhou, X. Wang, J. Feng, and S. Yan.
Metaformer is actually what you need for vision. In IEEE/CVF CVPR, 2022.
280

L. Yang, Y. Tian, B. Li, X. Zhang, K. Shen, Y. Tong, and M. Wang.
Mmada: Multimodal large diffusion language models. arXiv preprint
arXiv:2505.15809, 2025. 276

B. Yu, H. Yin, and Z. Zhu. Spatio-temporal graph convolutional networks:
A deep learning framework for traffic forecasting. In IJCAI, 2017. 309

C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding
deep learning (still) requires rethinking generalization. Communications
of the ACM, 64(3):107–115, 2021. 2

[ZCDLP17] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz. mixup: Beyond

empirical risk minimization. In ICLR, 2017. 213

[ZER

+

23]

+
[ZJM

21]

[ZKR

+

17]

[ZLLS23]

[ZS19]

[ZTS+21]

[ZW23]

A. Zador, S. Escola, B. Richards, B. Ölveczky, Y. Bengio, K. Boahen,
M. Botvinick, D. Chklovskii, A. Churchland, C. Clopath, et al.
Catalyzing next-generation artificial intelligence through neuroAI. Nature
Communications, 14(1):1597, 2023. 8

J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny. Barlow twins: Self-
supervised learning via redundancy reduction. In ICML, 2021. 55

M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. R. Salakhutdinov, and
A. J. Smola. Deep sets. In NeurIPS, 2017. 264

A. Zhang, Z. C. Lipton, M. Li, and A. J. Smola. Dive into deep learning.
Cambridge University Press, 2023. 12, 45, 46

B. Zhang and R. Sennrich. Root mean square layer normalization. In
NeurIPS, 2019. 228

S. Zhai, W. Talbott, N. Srivastava, C. Huang, H. Goh, R. Zhang,
arXiv preprint
and J. Susskind.
arXiv:2105.14103, 2021. 334, 335

An attention free transformer.

L. Ziyin and Z. Wang. spred: Solving l1 penalty with SGD. In ICML, 2023.
208

368
