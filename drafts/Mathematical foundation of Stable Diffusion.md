# Mathematical foundation of Stable Diffusion

## Appetizer: What is a generative model?


## Overview and math prerequisites


## Variational Autoencoder and Bayesian Learning

Autoencoder is a simple idea in neural network to compress data/learn a latent representation by artificially making a bottleneck in the network design. However there are many variations and details.

On the other hand, variational autoencoder (VAE) actually have more relation to Bayesian learning than autoencoder. The core of the diffusion process can be interpreted within the setup/framework of VAE. We should also not confuse this use of VAE as a theoretical foundation with the VAE as one of the neural network component in Stable diffusion itself, which actually incorporates many more elements/engineering than a vanilla VAE. (Though this is out of scope of this post)

Let's setup the math. Assume a Bayesian framework, so that everything is a (jointly distributed) random variable. The classical framework of theoretical statistics with parametric inference (which also becomes the framework of Statistical Learning/ML/AI with some modification) can be recast as:

Let $(X, Z)$ be jointly random, $Z$ is the latent/underlying model parameter, while $X$ is the random observable/data. The probability model is initially specified as the conditional probability distribution $p(x | z)$, which in a Bayesian framework is extended to requiring specifying the full joint probability distribution $p(x, z)$. Notice that this subsume the information contained by just $p(x | z)$ and indeed is enough to solve the inference problem by maximum likelihood - we can compute the marginal of the observable $p(x) = \int p(x, z) dz$, then apply definition of conditional probability to get $p(z | x) = \frac{p(x, z)}{p(x)} \left( = \frac{p(x | z)}{p(x)} p(z) \right)$. Of course in an actual course of Bayesian inference we will be more serious and discuss how this is different from the frequtist prardigm in that we have the additional assumption of the **piror** in the form of a probability distribution over $Z$, and arrange the formula to put more emphasis on updating the belief through the ratio as we've shown in the bracketed part above. We'd also discuss how $p(x)$ act as a **normalization factor** that is often a source of difficulty in Bayesian inference, and then go on to talk about the **likelihood ratio**.

Under this paradigm, there is a duality between inference and optimization - to perform inference of parameter, we just need to be able to optimize the objective function $p(z | x)$.

The generative modelling problem is an interesting converse (?) of the problem above. Here instead of starting with a latent creating an observable, and then trying to infer/recover the latent, we instead starts with the observables and try to model/infer a latent that capture its essential characeristic, after which we can generate random samples of $X$ by first sampling from the latent $Z$, and then artificially performing the conditional random generation of $X$ given $Z$. There are at least two points about their differences:

- The order of conditional probability is flipped, as well as which of $X$ or $Z$ is regarded as the primary objective
- The fact that the probabilistic modelling of $Z$ is considered an actual model (imperfect representation of real world) in the inference case, but can be a completely artificial creation for the generative modelling problem.

Nonetheless, the Bayesian + joint probability framework is flexible enough to accomodate them both as "order" doesn't matter (because we can just specify the joint probability). This flexibility is convinent mathematically, but can be problematic for engineers as it means you are responsible for keeping track of the interpretations yourself, which is not included in the model.

With the basic framework in place, we can move on to the actual problem. In practise, the normalization factor is often computationally intractable, and so we cannot just use the theoretically exact distribution to answer problems. An engineering approach would instead find some way to produce a learned approximation/proxy of the that function. In this context, the original/theoretically exact distribution is called the **ground truth**.

Here's the example: For an autoencoder, the compressed representation after passing through the encoder is the latent, the encoder is also a recognizer (to detect the latent features) while the decoder is a generator (to produce/reconstruct plausible data given the feature). Such an autoencoder can be used for the generative modelling problem by deducing a marginal distribution for the latent, then use the decoder to produce sample from the random latent. The objective here is for the marginal distribution of the produced data $\hat X$ to match the ground truth of the actual, **empirical** probability distribution of data $X$. Using autoencoder in this way is a bit tricky because:

- The latent space is designed by us, or in the case here, both encoder and decoder are jointly learned
- TODO?

Suppose $p$ is the ground truth and we have access to $p(x | z)$ directly as a decoder. Now we want to learn $q_\theta(z | x)$ to approximate the ground truth $p(z | x)$. One way to do this is ask to minimze the KL divergence $D_\text{KL}( q_\theta(z | x) || p(z | x) )$.

TODO: actual VLB.

## Diffusion Process in Stable diffusion

Here's the math following the framework above. Let $q$ be the ground truth distribution, $x_0$ be the data/observable, which in our case here is the distribution of images (whose value is a 2D "vector"). We artificially define our own latents, $x_1, \cdots, x_T$, through the forward diffusion process, by defining our artificial ground truth conditional distribution $q(x_t | x_{t-1} )$. We define it as a Markov Chain. Furthermore let $\beta_t \in (0, 1)$, a predetermined fixed sequence of constants, be the noise/variance schedule (will explain).

Also a notation: we will use $x_{i\ :\ j}$ (note the Colon) to denote a slice of the sequence, i.e. $x_i, x_{i+1}, \cdots, x_j$ (inclusive).

> Forward diffusion model: Define a markov chain where $x_0 \sim q(x)$, $q(x_t | x_{t-1}) \sim \mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}; \beta_t I)$.

In word: we begin by sampling the initial image through the true distribution, then at each step, we add a zero mean homoscadic gaussian white noise, but also shrink the vector towards zero by a scalar factor.

(Diffusion being a misnomer)

The reason for this modification is that without it, it would be just like a Brownian motion and the variance will explode to infinity, which we don't want. We can show that this converge to a white noise in the time limit $t \rightarrow \infinity$ with a bounded variance. But more powerful than that, we will demostrate the **reparametrization trick**, which gives us a way to derive various analytically tractable/closed form expression later on.

Let $\alpha_t = 1 - \beta_t$, $\bar \alpha_t = \prod_{i=1}^t \alpha_i$. Let $\epsilon_t \sim \mathcal{N}(0,1)$ be iid sequence of Gaussian. We can rewrite the definition above as (obviously) $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}$. Moreover, by repeatedly doing substitution of this formula, we have:

$$
x_t = \sqrt{\bar \alpha_t} x_0 + \sqrt{1 - \bar \alpha_t} \epsilon
$$
where $\epsilon$ here is a merge of the $\epsilon_t$'s.

Our mathematical problem is to perform the reverse diffusion process. The theoretically exact one is analytically untractable, but we can derive a closed form expression if we cheat a bit *and additionally condition on the original image*. Indeed it is Gaussian distributed and we can again give closed form formula for the mean and variance.

Using Bayes formula (with the conditioning on $x_0$ kept as part of the background),

$$
q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)} \\
\sim \exp \left( -\frac{1}{2} \left( \frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar \alpha_{t-1}} x_0)^2}{1-\bar \alpha_{t-1}} + \frac{(x_t - \sqrt{\bar \alpha_t} x_0)^2}{1-\bar \alpha_t} \right) \right) \\
\vdots \\
= \exp \left( -\frac{1}{2} \left( (\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar \alpha_{t-1}}) x_{t-1}^2 - (\frac{2\sqrt{\alpha_t}}{\beta_t} x_t + \frac{2\sqrt{\bar \alpha_{t-1}}}{1 - \bar \alpha_{t-1}} x_0) + C(x_t, x_0) \right) \right)
$$

(more mundane algebra skipped)

Now how should we train the network? In our setup $x_0$ is the data, $x_{1\ :\ T}$ is the latent, and $p_\theta(x_{t-1} | x_t)$ is our network. Our training objective is to minimize the cross entropy:

$$
- \mathbb E_{x_0 \sim q(x_0)} \left[ \log p_\theta(x_0) \right] \leq - \mathbb E_{x_{0\ :\ T} \sim q(x_{0\ :\ T})} \log \frac{p_\theta(x_{0\ :\ T})}{q(x_{1\ :\ T} | x_0)}
$$

using the variational lower bound. Now the key results here is that by making suitable adjustment, we can express this surrogate objective in terms of comparing $p_\theta(x_{t-1} | x_t)$ against our cheating theoretical distribution $q(x_{t-1} | x_t, x_0)$ instead of the honest/actual one $q(x_{t-1} | x_t)$

## Markov Chain Monte Carlo, Langevin Dynamics, and Score based generative model


## Sampling methods in Stable diffusion




## Appendix

### Some interpretation of entropy inequalities in information theory

