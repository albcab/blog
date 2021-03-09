---
title: "Inverse Autoregressive (Normalizing) Flows using JAX: a MADE implementation"
layout: post
toc: true
comments: true
categories: [Variational inference, Normalizing flows, JAX]
---

## Introduction

Recently I've dwelled in the world of approximate Bayesian inference, or more generally speaking: Variational inference. I find it interesting because it's fast, which usually doesn't go very much in accordance with the whole Bayesian thing. It achives this by stating the problem as an optimization one and exploiting the automatic differentiation libraries availiable. Essentially, we are interested in approximating a function (in the Bayesian case it is usually a posterior distribution) with another function with parameters $\phi$, our objective is to find $\phi$ that minimizes the distance or divergence between the two. A (much) better, detailed explanation can be found in [this review paper](https://arxiv.org/abs/1601.00670) or in [this blog post](https://ermongroup.github.io/cs228-notes/inference/variational/), I also find the good old [wikipedia page](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) to be a great resource. 

As in all there are tradeoffs, in this case the speed comes at the price of inexactness. Recent literature has focused (at least from what I've read) on improving the family of parametrized functions we use to approximate the distribution we are interested in (I'll talk from a Bayesian perspective). What started from mean field stuff I don't understand grew to the current vanguard of normalizing flows, a fairly simple concept that builds upon itself to increase its complexity, allowing us (theoretically) to express a pretty big class of density functions. For completness I will state that any density function with strictly positive values on its entire domain and with differentiable conditional probabilities can be built from a base (uniform/normal) distribution, see section 2.2 of [this paper](https://arxiv.org/abs/1912.02762). The beauty of a simple yet effective method has attracted much attention and it seems to have become the weapon of choice for the Bayesian variational inference-er (at least when dealing with continuous variables). 

However, it is still only an approximation. And you know we (ok, me and my Bayesian lads) care about that EXACT inference. Thus, [I've](https://arxiv.org/abs/1903.03704) had the [great idea](https://arxiv.org/abs/1602.05023) of supplementing good old Monte Carlo with an as-close-as-possible-to-gaussian approximation of the density I'm interested in using variational inference with normalizing flows. But first, I must master the flow. Hence this post.


## Normalizing Flows

No point in explaining the details when you can get it from the very best: [this is a great review paper](https://arxiv.org/abs/1912.02762) on the subject covering theory and applications with great references, it goes well beyond the scope of variational inferece; [this paper](https://arxiv.org/abs/1505.05770) focuses on flows for variational inference; [this blog post](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html) is similar to the latter paper while [this one](https://blog.evjang.com/2018/01/nf1.html) has more intution, detail and code. I will only write the gist of it, so we can agree on notation: we are interested on a density function $p(x) \propto \tilde{p}(x)$ for $x \in \mathbb{R}^d$, we approximate it by transforming a random variable $u$, i.e. flow, and correcting for the changes in density caused by the transformation, i.e. normalize, in other words we use variable transformations in the following sense

$$
x = f_{\phi}(u) \quad for \quad u \sim q(u).
$$

Notice that the function transforming the variable is parametrized by a vector $\phi$, our objective is to optimize this vector. The base density $q(u)$ is gaussianlly normal and it can be parametrized by its own vector $\varphi$. However, we will assume that the transformation from a standard multivariate normal distribution to a multivariate normal with paramters $\varphi$ can be absorved by $f_{\phi}$, i.e. making $\varphi$ part of $\phi$, see [section 2.3.2](https://arxiv.org/abs/1912.02762). This way we have an approximation of our density of interest 

$$
q_{\phi}(x) = q(f_{\phi}^{-1}(x))\left|\det\frac{df_{\phi}}{du}(f_{\phi}^{-1}(x))\right|^{-1},
$$

where $\frac{df_{\phi}}{du}(f_{\phi}^{-1}(x))$ is the Jacobian matrix of $f_{\phi}(u)$ evaluated at $f_{\phi}^{-1}(x)$. Notice, for all of this to work, $f_{\phi}$ must be invertible and differentiable. 

The fact that allows us to complicate this simple idea indefinetly is that for any amount of invertible and differentiable transformations $f_1,...,f_K$ their composition $f_K \circ \cdots \circ f_1$ is also invertible and differentiable, and the determinant of the Jacobian matrix for the composition can be easily derived by the identity 

$$
\det\frac{d(f_K \circ \cdots \circ f_1)}{du}(u) = \det\frac{df_K}{du}((f_{K-1} \circ \cdots \circ f_1)(u)) \cdots \det\frac{df_1}{du}(u).
$$

Now that we know our approximation and the parameters it depends on, we can optimize them to make this new density as close as possible to $p(x)$. To do this we need a measure of distance or divergence between the two. The [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) is the preferred measure in the variational inferece literature, the normalizing flows literature (for some generalizations of it see [section 2.3.4](https://arxiv.org/abs/1912.02762)) and fairly prevalent in the Bayesian nonparametric [posterior consistency](https://staff.fnwi.uva.nl/b.j.k.kleijn/NPBayes-LecNotes-2015.pdf) literature, so lets stick to that for now. It also lets us use Monte Carlo estimates of its integrals if we have access to samples from either $u$ or $x$, this will come in handy on the implementation. The Kullback-Leibler divergence we'll use to optimize our parameters $\phi$ can be simplified to

$$
KL(q_{\phi}(x) || p(x)) = \int \log \left( \frac{q_{\phi}(x)}{p(x)} \right) q_{\phi}(x)dx $$
$$ = \int \log \left( \frac{q(u) \left| \det\frac{df_{\phi}}{du} \right|^{-1}} {p(f_{\phi}(u))} \right) q(u)du $$
$$ = \mathbb{E}_ {q} \left[ \log(q(u)) - \log(\left| \det\frac{df_{\phi}}{du} \right|) - \log(p(f_{\phi}(u))) \right] $$
$$\propto -\mathbb{E}_ {q} \left[ \log(\left|\det\frac{df_{\phi}}{du}\right|) + \log(\tilde{p}(f_{\phi}(u))) \right],
$$

where the first equality is by definition, the second is proved in [appendix A](https://arxiv.org/abs/1912.02762), and the proportionality is w.r.t. $\phi$. In the usual Bayesian inference set up, we know the function $p(x)$ up to a constant of proportionality but have no samples from it; setting up the divergence as above makes it practical for Bayesian analysis, assuming that $q(u)$ is chosen such that we can easily sample from it. 


## Inverse Autoregressive Flow

The main burden in deriving our approximate density is that of computing the determinant of the Jacobian matrix $\frac{df_{\phi}}{du}$ at some given value. Each transformation in the flow would require to compute and store a $d x d$ square matrix to then compute its determinant. Even for relativeley small $d$ this could be computationally expensive and for a large $d$ it might be imposible to store in memory. Solving this issue is the goal of the Inverse Autoregressive flows: the Jacobian matrix of this transformation is lower triangular, i.e. the determinant is the product of its diagonal. Again, learn it from somebody that is smarter than me: [this is the paper that introduced it](https://arxiv.org/abs/1606.04934), [this blog post](https://www.ritchievink.com/blog/2019/11/12/another-normalizing-flow-inverse-autoregressive-flows/) goes straight to the point, while [this one](https://bjlkeng.github.io/posts/variational-autoencoders-with-inverse-autoregressive-flows/) provides more detail and intution behind the idea. 

The concept is simple, at each step of the flow we scale and shift each dimension of the input variable with parameters determined by previous dimensions of the input variable. Formally, given an input variable $u$ we derive the output variable $x$ of the same dimension such that

$$
x_i = (f_{\phi}(u))_i =  \frac{u_i - \mu_i(u_{1:i-1}, \phi)}{\sigma_i(u_{1:i-1}, \phi)} \quad i = 1,\dots,d,
$$

where $\sigma_1(\phi)$ and $\mu_1(\phi)$ depend only on the flow's parameters. Notice that the product of the elements on the diagonal of $\frac{df_{\phi}}{du}$ is $\prod_{i=1}^d \sigma_i(u_{1:i-1}, \phi)^{-1}$, avoiding computing the whole matrix and finding its determinant. 

Now, regarding the derivation of parameters $(\sigma_i(u_{1:i-1}, \phi), \mu_i(u_{1:i-1}, \phi))_{i=1}^d$ the possibilities are endless. Black box deep learning practices allow for complex distributions to be transformed in relativley few steps of the flow into gaussian distributions (or at least it seems to be that way empirically) and, since we don't need its functional form for any other derivation of the flow (only its value given some fixed parameters), they are our best choice in exploiting the potential of these methods. To implement this efficiently and allowing for a general class of architectures we use Masked Autoencoders for Distribution Estimation (MADE). Here is the [original paper](https://arxiv.org/abs/1502.03509) and some [blog](https://www.ritchievink.com/blog/2019/10/25/distribution-estimation-with-masked-autoencoders/) [posts](https://bjlkeng.github.io/posts/autoregressive-autoencoders/). 

The autoencoder background is irrelevant for this particular application, all that provides value here is the addition of "masks" to the standard neural network architecture which allows for autoregressive outputs. With the masked architecture we can choose the number of hidden layers $h \geq 0$ and the activation functions for the output and each hidden layer $(g_i)_ {i=1}^{h+1} $ to define autoregressive means (or log scales) given input $u$ and letting $H_0(u) = u$,

$$
H_i(u) = g_i(b_i+(W_i \odot M_i)H_{i-1}(u)) \quad i=1,\dots,h \\
\mu(u, \phi) = g_{h+1}(b_{h+1}+(W_{h+1} \odot M_{h+1})H_{h}(u))
$$ 

where $\odot$ indicates element wise product, $(M_i)_ {i=1}^{h+1}$ are $d x d$ matrices with elements $m_{i,j} \in \{0,1\}$ which are always (for Inverse Autoregressive flows) upper traingular and $M_1$ has zeros also on its diagonal, $(b_i)_ {i=1}^{h+1}$ and $(W_i)_ {i=1}^{h+1}$ are $d x 1$ vectors and $d x d$ matrices respectivley and the parameters of our model, i.e. $(b_i, W_i)_ {i=1}^{h+1} \subset \phi$ for each mean and log scale in our flow. Notice that this architecture ensures its output has the autoregressive characteristics we are interested in to simplify the calculation of the Jacobian of our flow, i.e. $(\mu(u, \phi))_ i = \mu_i(u_{1:i-1}, \phi)$.


## JAX code

[JAX](https://github.com/google/jax) is a growing functional python library that is able to compile and perform automatic differentiation on GPUs and TPUs (brrrrrrrr). Their documentation provides a [great introduction](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) to their syntax and a [cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) that can come in handy, also try to have their [API documentation](https://jax.readthedocs.io/en/latest/jax.html) always at hand. [This blog post](https://blog.evjang.com/2019/07/nf-jax.html) provides example code in JAX for normalizing flows, also get a feel for the mechanics of JAX with some of their [examples](https://github.com/google/jax/tree/master/examples).

```python
import jax.numpy as jnp
import jax.random as rand
from jax import vmap
```

The objective is to build and algorithm that composes any finite number $K$ of Inverse Autoregressive transformations into a flow $f_{\phi} = f_K \circ \cdots \circ f_1\$, minimizes the Kullback-Lieber divergence between our density of interest $p(x)$ and the approximation from the flow $q_{\phi}(x)$, and efficiently transforms input observations $u$ into approximate observations $x$ with density $p(x)$.

### Step 1: Masked Neural Architecture

The JAX functional programming style makes the parameters the central part of the algorithm. We will essentially create a series of functions that together randomly initialize the parameters, calculate the Kullback-Leiber divergence as a function of the parameters $\phi$ (and other fixed values), differentiate it w.r.t. $\phi$ and use these derivatives to minimize. Hence, it's important how we feed these parameters to our functions, taking into consideration the autoregressive restrictions we need to impose and the JAX interface. Using [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) is probably a good idea, to keep JAX happy, then one autoregressive transformation with $h = 1$ hidden layer for its log scale and $h=0$ hidden layers for its mean would require the parameters in pytree fashion

$$
[ (W_1 \odot M_1 , b_1) , (W_2 \odot M_2 , b_2) ]
$$

for the log scale and 

$$
[ (W_1 \odot M_1 , b_1) ]
$$

for the mean. To implement the masked restrictions in code we simply feed them as a list of row vectors, ignoring the zeros. Consider a dimension $d=2$ to keep things simple, then our paramters would have the form
```python
[ ([ [w_{1,2}], [] ], [b_{1}, b_{2}]), ([ [w_{1,1}, w_{1,2}], [w_{2,2}] ], [b_{1}, b_{2}]) ]	#log scale
[ ([ [w_{1,2}], [] ], [b_{1}, b_{2}]) ]								#mean
```
then inside the functions we simply add the necessary zeros and we can do our linear combinations, activations and repeat until the complexity creates an intelligent being. Notice that the first layer always has zeros in the diagonals. In the same way we input the activation functions as a list `[activation_fun1, activation_fun2]` (for log scale) or a single activation function on a list `[activation_fun]` which the function will understand to mean `[activation_fun, ..., activation_fun, identity_fun]` for $h \geq 1$ (`[activation_fun1, identity_fun]` for log scale). 

Thus, the (Masked) Autoregressive neural network function that calculates the mean or log scale parameters of the model:
```python
def DenseAutoReg (z, parameters, activations):
    if len(activations) == 1 and len(parameters) > 1:
        h = len(parameters)
        activations = [activations[0] for _ in range(h-1)] + [None]
    else:
        assert len(parameters) == len(activations)
        
    for j, (W, b) in enumerate(parameters):
        if j == 0:
            W = [jnp.concatenate([jnp.zeros(i+1), w]) for i, w in enumerate(W)]
        else:
            W = [jnp.concatenate([jnp.zeros(i), w]) for i, w in enumerate(W)]
        W = jnp.array(W)
        z = jnp.dot(z, W) + b
        if activations[j] is not None:
            z = activations[j](z)
    
    return z
```

### Step 2: Inverse Autoregressive Flow

An inverse autoregressive transformation would require as input parameters and activation functions for both the mean and log scale.
```python
def InvAutoRegFlow (z, mu_param, log_sd_param, mu_act, log_sd_act):
    mu = DenseAutoReg(z, mu_param, mu_act)
    log_sd = DenseAutoReg(z, log_sd_param, log_sd_act)
    
    return (z - mu)/jnp.exp(log_sd)
    # return jnp.exp(log_sd)*z + mu
```

Then we repeat this transformation $K$ times, this time extending the pytree for the paramters to a list with $K$ elements `[ (mean_parameters1, log_scale_parameters1), ..., (mean_parametersK, log_scale_parametersK)]`, same for the activation functions. The function will also allow for the ordering of the inputs to be reversed at each step of the flow, because I'm a sheep ([section 4.1.1](https://arxiv.org/abs/1903.03704)), and output the log of the product of the determinant of the Jacobian matrices of each Inverse Autoregressive layer of the flow.
```python
def MakeFlow (z, parameters, activations, invert = True):
    log_det_jac = 0.
    for param, act in zip(parameters, activations):
        log_det_jac -= jnp.sum(DenseAutoReg(z, param[1], act[1]))
        # log_det_jac += jnp.sum(DenseAutoReg(z, param[1], act[1]))
        z = InvAutoRegFlow(z, *param, *act)
        if invert:
            z = z[::-1]
    if invert and len(parameters) % 2 > 0:
        z = z[::-1]
    
    return z, log_det_jac
```

### Step 3: Minimizing the Kullback-Leiber divergence

The integral needed to compute the Kullback-Leiber divergence (or at least the part which concerns $\phi$) is approximated by Monte Carlo, assuming that we can generate random variables with density $q(u)$.
```python
def MCKLDiv (Z, parameters, activations, log_target, invert = True):
    X, LDJ = vmap(MakeFlow, (0, None, None, None), 0)(Z, parameters, activations, invert)
    KL = -jnp.sum( vmap(log_target, 0, 0)(X) + LDJ )
    
    return KL/(jnp.shape(Z)[0])
```

The last fundamental function needed withing the JAX framework is one that (randomly) initializes the parameters of the model to their aforementioned pytree structure. To do this we use a slight modification of the [variance_scaling](https://jax.readthedocs.io/en/latest/_modules/jax/_src/nn/initializers.html#variance_scaling) function of `jax.nn.initializers` that outputs vectors of different sizes for a specific dimension and is used to initialize the (masked) row vectors of $W \odot M$ and $b$.
```python
def variance_scaling(scale, distribution, dtype=jnp.float32):
  def init(key, denominator, shape, dtype=dtype):
    variance = jnp.array(scale / denominator, dtype=dtype)
    if distribution == "truncated_normal":
      # constant is stddev of standard normal truncated to (-2, 2)
      stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
      return rand.truncated_normal(key, -2, 2, shape, dtype) * stddev
    elif distribution == "normal":
      return rand.normal(key, shape, dtype) * jnp.sqrt(variance)
    elif distribution == "uniform":
      return rand.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")
  return init
```
Then, the initializer function randomly builds the pytrees for a given dimension `d`, number of repetitions of the Inverse Autoregressive transformation `K`, and pytree of length `K` with the number of hidden layers at each transformation of the form `[ (#hidden_layes_mean, #hidden_layes_log_scale), ...]`.
```python
def init_rand_param (d, K, hidden_layers, seed = 123, rng = variance_scaling(1., "truncated_normal")):
    assert K == len(hidden_layers)
    keys = [rand.PRNGKey(seed)]
    parameters = []
    for f in range(K):
        keys = rand.split(keys[0], 1+d+1)
        mu_layers = [([rng(keys[i+1], d/1., (d-(i+1), )) for i in range(d)], rng(keys[d+1], d/1., (d, )))]
        keys = rand.split(keys[0], 1+d+1)
        log_sd_layers = [([rng(keys[i+1], d/1., (d-(i+1), )) for i in range(d)], rng(keys[d+1], d/1., (d, )))]
        for h in range(hidden_layers[f][0]):
            keys = rand.split(keys[0], 1+d+1)
            mu_layers.append(([rng(keys[i+1], d/1., (d-i, )) for i in range(d)], rng(keys[d+1], d/1., (d, ))))
        for h in range(hidden_layers[f][1]):
            keys = rand.split(keys[0], 1+d+1)
            log_sd_layers.append(([rng(keys[i+1], d/1., (d-i, )) for i in range(d)], rng(keys[d+1], d/1., (d, ))))
        parameters.append((mu_layers, log_sd_layers))
        
    return parameters
```

And that is all, now we simply choose an optimization algorithm `from jax.experimental import optimizers`, given some function of interest `posterior`, choosing paramters `K, activations, hidden_layers` we define our loss function,
```python
loss = lambda param, Z: MCKLDiv(Z, param, activations, posterior)
```
generate randomly `Z` from a standard multivariate normal, and, assuming we are using sgd (choose a `step_size`), optimize that bitch.
```python
from jax import value_and_grad, jit

opt_init, opt_update, get_params = optimizers.sgd(step_size)

@jit
def update (i, opt_state, Z):
    params = get_params(opt_state)
    KL, grad_params = value_and_grad(loss)(params, Z)
    return KL, opt_update(i, grad_params, opt_state)

params = init_rand_param(d, K, hidden_layers, rng = variance_scaling(1., "normal"))
opt_state = opt_init(params)
for i in range(num_batches):
    KL, opt_state = update(i, opt_state, Z)
    print("KL Divergence = {}".format(KL))
```

## Example

There is much to be said about the Monte Carlo approximation of the gradient used in optimization, here is a [review paper](http://www.optimization-online.org/DB_FILE/2013/06/3920.pdf), and about the different variations of gradient descent that could be used, here is a [review blog post](https://ruder.io/optimizing-gradient-descent/index.html). This example is only meant to prove that the method described above works and that it can build complex neural architectures. The objective is to approximate a multivariate normal distribution with a non diagonal covariance matrix and a non zero mean. This will be achieved using an Inverse Autoregressive flow of two transformations ($K=2$), one hidden layer for both the mean and log scale at each transformation, [log sigmoid](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.log_sigmoid.html#jax.nn.log_sigmoid) activation function for the hidden layers in the first transformation and [ELU](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.elu.html#jax.nn.elu) in the second, and a standard multivariate normal distribution as a base measure.

Consider an objective posterior density for $x \in \mathbb{R}^2$ s.t.
$$
x \sim \mathcal{N}_2\left(  \begin{bmatrix} 2 \\ -2 \end{bmatrix},  \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} \right),
$$
i.e. transform ![standard normal]({{ site.baseurl }}/images/norm_hex.png) into ![nonstandard normal]({{ site.baseurl }}/images/normNF_hex.png)

```python
import itertools

from jax import value_and_grad, jit
from jax.experimental import optimizers as optim
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as rand
import jax.scipy as jsp

hidden_layers = [(1, 1), (1, 1)]
K = 2
activations = [([jnn.log_sigmoid], [jnn.log_sigmoid]), ([jnn.elu], [jnn.elu])]
MC_n = 100
d = 2

mean = jnp.array([2, -2])
var = jnp.array([[2, -1], [-1, 2]])
posterior = lambda x: jsp.stats.multivariate_normal.logpdf(x, mean, var)

loss = lambda param, Z: MCKLDiv(Z, param, activations, posterior)

opt_init, opt_update, get_params = optim.adagrad(step_size = 0.1, momentum = 0.9)

@jit
def update (i, opt_state, Z):
    params = get_params(opt_state)
    KL, grad_params = value_and_grad(loss)(params, Z)
    return KL, opt_update(i, grad_params, opt_state)

params = init_rand_param(d, K, hidden_layers, rng = variance_scaling(1., "normal"))
opt_state = opt_init(params)
itercount = itertools.count()
key = rand.PRNGKey(123)
Z = rand.normal(key, (MC_n, d))
Z_ = Z
count = 0
while True:
    ind, div, param = -1, 1e10, []
    stop = False
    while not stop:
        i = next(itercount) - count
        KL, opt_state = update(i+count, opt_state, Z)
        print("\rKL Divergence = {}".format(KL), end = "\r")
        if KL < div:
            ind, div, param = i, KL, opt_state
        elif i > 1000:
            stop = True
    count += i
    Z_ = jnp.concatenate([Z_, Z])
    Z_div = loss(get_params(param), Z_)
    print("\nKL batch = {} (at {}), KL all = {}\n".format(div, ind+1, Z_div))
    yn = input("write anything to stop")
    if yn != '':
        break
    key, subkey = rand.split(key)
    Z = rand.normal(subkey, (MC_n, d))
```

After a couple of epochs we get parameters that transform 1100 observations with mean $$\begin{bmatrix} 0.03882353 \\ 0.01266078 \end{bmatrix}$$ and covariance $$\begin{bmatrix} 0.9359905 & -0.00276474 \\ -0.00276474 & 0.9968354 \end{bmatrix}$$ to mean $$\begin{bmatrix} 2.234608 \\ -2.1736727 \end{bmatrix}$$ and covariance $$\begin{bmatrix} 1.6226339 & -0.88107896 \\ -0.88107896 & 2.2406228 \end{bmatrix}$$.
Visually ![standard normal]({{ site.baseurl }}/images/norm.png) to ![nonstandard normal]({{ site.baseurl }}/images/normNF.png) :rocket:
