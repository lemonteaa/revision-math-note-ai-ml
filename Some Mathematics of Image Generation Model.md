# Some Mathematics of Image Generation Model

## Overview

## Math Prerequisites and Reviews

> TODO: Currently I wrote about it inline in the content below, can consider moving it up here.

Unfortunately (and I really mean to say sorry about this), this is not for the faint of heart. A reasonably strong background at the upper undergraduate level is expected in advanced probability theory, theoretical statistics, information theory, and things like dynamical system/analysis of differential equations. It might be overkill, but without these preparation you may find yourself overwhemled.



## Latent Diffusion Model - Variational Autoencoder

### Background - Variational Bayesian Inference and Evidence Lower Bound

**References for this subsection**

- [ELBO — What &amp; Why | Yunfan’s Blog](https://yunfanj.com/blog/2021/01/11/ELBO.html)

- [The evidence lower bound (ELBO) - Matthew N. Bernstein](https://mbernste.github.io/posts/elbo/)

- [Eric Jang: A Beginner's Guide to Variational Methods: Mean-Field Approximation](https://blog.evjang.com/2016/08/variational-bayes.html)

- https://www.cs.jhu.edu/~jason/tutorials/variational.html



**Foreword**

This topic is deceptively tricky to get right mainly due to difficulty with conceptual modelling. The math formula themselves are manageable, but perhaps due to the diversity of application and it being somewhat inter-disciplinary, there's lots of catcha and details to be careful of.

> TODO: This is only a draft and an attempt to weave a reasonably coherent but also comprehensive narrative. Might get overhauled completely in the future.

So let me list the potential traps first:

- *Ground truth is relative* - We will be doing "model of a model" like a basic model $M$ that imperfectly capture something in the real world, and then another model $M'$ that is an approximation of $M$. While $M$ itself is only an approximation of reality (i.e. the ground truth), when we consider it relative to $M'$, it will become the "ground truth" to compare against in that context. In a similar spirit, although we will assign "roles" to entities, these roles are often relative and contextual, and may flip over completely as we change perspective (which we will do quite a lot of).

- *Anything can become a latent variable* - Continuing from the theme above, a variation (no pun intended) is that we will switch from a setup where some variables are given and we want to infer some other latent variables, into one that consider all variables on an equal footing, then flip over and try to infer the originally given variable from the originally latent one. More complex setup includes one that may instead insert new variables in the middle of a derivation. A specific version of this is the *Frequentist vs Bayesian debate* - as we're in the Bayesian framework, we will consider even parameters that are normally modelled non-probabilistically and just forcefully subsume it as "yet another joint random variable".

- *The distribution itself is uncertain* - A meta and (possibly) non-parametric version of the same issue above. While in a standard setup we will have a single probability distribution, even though it may be "flexible" with the underlying measure space expanding in an ad hoc way as we change the model and add new variables etc, here we will often consider more than one (candidate) probability distribution.

Hopefully, the fuzzy and messy list above will get you into the right "headspace" to properly understand it. My personal experience is that the majority of the difficulty in groking this topic lies in the messy semantics of the model, and not the math itself.



**Some Archetypal Examples**

Given the diversity of origins/usage of this technique, it is perhaps best to give a selection of representative examples.



*Example 1 - Classical Bayesian Inference on a Graphical Model*

Let $(X, Z) \sim P$ be jointly distributed random variable with probability distribution $P$. Suppose we are given an observed value of $X$ - perhaps representing some real world data. Then we would like to infer possible value of the latent (hidden) variable $Z$. In a classical setup, $P$ represent our model of how things come to be - we have a piror over $Z$, as well as an explanatory model in terms of a likelihood distribution $X | Z$. Our goal is to find the posterior distribution $Z | X$.

Experienced reader (which I expect you to be as otherwise this topic might be too hard for you, sorry) should recognize that this is just the standard Bayesian model setup, and answer is given by using the Bayes Theorem. Namely, we have

$p(Z = z | X = x) = \frac{p(X = x | Z = z) p(Z = z)}{p(X = x)} $.

Technically the above is clearest way to write the formula as it is clear about using the density function $p$ and that we should give specific values of $x, z$ to it. But for convinience, we often write instead

$p(Z | X) = \frac{p(X | Z) p(Z)}{p(X)}$

with the implicit understanding that you should insert the concrete values yourself.

A number of remarks are in order:

- Recall that once we are given the joint probability distribution, we can derive the other quantities. Marginal distribution of individual variables is done by marginalization: $p(X) = \int_Z p(X, Z) dZ$. (Ditto for $p(Z)$). Conditional distribution can then be defined as $p(X|Z) = \frac{p(X, Z)}{p(Z)}$.

- In complex setups we may opt to define a joint distribution by specifying some of the individual components instead of directly. For example, we may specify the piror $p(Z)$ and the likelihood $p(X|Z)$, then the joint distribution is recovered as $p(X, Z) := p(X|Z) p(Z)$.

- One potential difficulty in the solution above is that the marginal distribution $p(X) = \int_Z p(X, Z) dZ$ is often difficult to evaluate analytically, or outright impossible to evaluate (eg if only part of the distribution is defined analytically at all while the rest contain an empirical component that we only have oracle access to). This is one major motivation to develop the technique of Variational Bayesian in the first place.

- One may sometimes consider additional tasks to do. An example would be to do prediction - given a value of the latent $Z$, try to predict what value $X$ would we observe. This is one instance of the "flipped role" scenario above.



*Example 2 - (Non-parametric) Density Estimation*

Suppose we have an unknown true probability distribution $p*$ over $X$. We can sample from $p*$ empirically to get a dataset of $X$ values. We develop a non-parametric probability model $p$ and want to fit it to match $p*$ closely. Reusing the example above, our model may, for instance, specify the piror $p(Z)$ and the likelihood $p(X|Z)$ . Then the estimated distribution over $X$ would be $p(X) = \int_Z p(X, Z) dZ = \int_Z p(X|Z) p(Z) dZ$.

One concrete example is the mixture of gaussian model. Let $X = Y + Z$, where $(Y, Z)$ are jointly distributed independent random variables. Moreover,

- $Y$ is a raw estimated empirical distribution and its (marginal) density function is just weighted sum of delta functions.

- $Z$ is a standard Gaussian.

We then have

$p(X = x | Z = z) = p(Y = x - z) = \sum_i w_i \delta(x - z - y_i)$

And so

$p(X = x) = \sum_i w_i \text{Gaussian}(x - y_i)$.

This is often visualized as smoothing out the deltas/datasets using the Gussian as a mollifier.

Notice that the way you specify the model may constraint the possible shape of the final marginal distribution for $X$, so that a perfect fit may become impossible.



*Example 3 - Bayesian Maximum Likelihood*

This can be considered to be a parametric counterpart of the above.

Recall that we can estimate a model parameter by doing optimization:

$\theta^* = \argmax_\theta \ln p_\theta(x)$

This is the Maximum Likelihood Estimator (MLE). The theoretical justification comes again from the example 1 above, but where we turn the non-probabilistic parameter $\theta$ into a latent random variable with probability representing "degree of belief" (so the joint distribution is written as $p(X; \theta)$). Of course, once this is justified and we no longer need to touch it, we can put it in the subscript.

Here, the same difficulty in example 1 applies: the marginal distribution comes from the integral $p_\theta(x) = \int p_\theta(x, z) dz$.

Additional note on example 3:

Similar to how least square regression is equivalent to maximum likelihood given the set of assumptions (one of which is independent homoscenditic Gaussian noise), it pays to derive mathematically equivalent but diverse perspectives. The term $\ln p(x)$ is also called the evidence, and it turns out in the context of fitting to an empirical distribution, MLE is equivalent to minimizing the KL-divergence between the actual distribution $p^*$ versus the model $p$.

To see this, first note that KL-divergence is same as cross-entropy after shifting by a term that is constant with respect to optimization:

$D_\text{KL}(p^* || p_\theta) = \mathbb E_{x \sim p^*} [ \ln \frac{p^*}{p_\theta}] = - \mathbb E_{x \sim p^*} [ \ln p_\theta ] + H(p^*)$

Thus, minimizing this with respect to $\theta$ is same as maximizing $\mathbb E_{x \sim p^*} [ \ln p_\theta ]$. From Stochastic Gradient Descent, we then see that $\ln p_\theta$ is a stochastic proxy for the objective when we are sampling from the empirical/true distribution $p^*$.



**Formulation**

The main idea is to proxy the posterior distribution by a function $q_\phi(z|x)$ that approximate the theoretical true posterior $p(z|x)$.

Note:

- As $p$ or $p_\theta$ is already a model, this can be considered a "model of model".

- (TODO) This is also called an amortization or mean field estimate.

- (TODO) While it is possible to do so, the proxy does not have to specify the full joint distribution. This flexibility/provision to only "partially specify a distribution" makes it powerful.

- It is called Variational because we will try to choose $q$ from within some family of functions to optimize some objective, which is basically calculus of variation in principle. However, in many setup as the function space we'll optimize over is parametrized by finitely many parameters ($\phi$), it is effectively a finite dimensional space and in that case ordinary calculus will also suffice.



Now, let's derive the Evidence Lower Bound (ELB).

$\ln p(x) = \ln \int p(x, z) dz = \ln \int \frac{p(x, z)}{q_\phi(z|x)} q_\phi(z|x) dz = \ln \mathbb E_{Z \sim q_\phi} \left[ \frac{p(x, z)}{q_\phi(z|x)} \right] \geq \mathbb E_{Z \sim q_\phi} \left[ \ln \frac{p(x, z)}{q_\phi(z|x)} \right] = \mathcal L$

(Where we used Jensen's inequality)

Then, in many relevant problems, we will optimize the ELBO as a proxy of doing maximum likelihood (i.e. maximizing $\ln p(x)$).

The main value comes from investigating various property of the quantity $\mathcal L$ and its interpretations.



First, we can expand it into two terms:

$\mathcal L = \mathbb E_{Z \sim q_\phi} [ \ln p(x, z) ] + H(q_\phi(\cdot | x) ) $

Where the dot is to indicate the variable that the outer entropy function is active over - in other words we held $x$ fixed and consider it as a distribution over $z$.

This can be interpreted with respect to the example 2 above of kernel estimation. In that context, the first term represent optimizing a weighted sum where the data is fixed ($\ln p(x, z)$), but where the weights (probabilities assigned to $Z$) is variable. In that case we should concentrate our weights to the max value of $\ln p(x, z)$. The second term can be considered as a regularization term to prevent a delta function solution. For instance, if $q_\phi$ is a parametrized family of Gaussians and the parameters $\phi$ are the means and variance, then maximizing entropy also in balance with the first term will result in the Gaussian spreading out somewhat around the peak value of $\ln p(x, z)$. This can also be interpreted as invoking the principle of maximum entropy as a philosophical rule to apply on the piror of $q_\phi$. (i.e. absent any information, we should generally choose our piror to maximize entropy. Hence for example in a finite categorical/discrete random variable, with no information we would choose a uniform probability as piror).



A second propety is that it turns out to be possible to get an exact formula for the gap between the evidence and the ELBO (!). So if you don't like the seemingly arbitrary inequality technique, you may think of it as an algebraic identity instead. The catch though is that the expression may, depending on applications, involves quantities that we don't have an analytical expression for.

Result: $\ln p(x) - \mathcal L = D_\text{KL}( q_\phi( \cdot | x) || p( \cdot | x) )$

(Both are considered as $z$ distribution)

Notice that this is a reverse-KL as here we regard $q_\phi$ as the "ground truth" when the actual situation is normally the opposite. We can think of this as the information loss necessary to distort from one distribution to the other. One way to explain why reverse-KL appears is the notion that when minimizing the KL, it will results in either a zero-forcing or opposite behavior depending on the order of the distributions.

Also, as we will show later, the ELBO can be written as sum of term one of which is a KL distance. So in the full identity, there will actually be two KL terms. Something to keep in mind of.

Finally, in the context of applying this theory to variational autoencoder (which is the one of the main goal of this article), the ELBO can also be expressed, alternatively, as the sum of an end-to-end reconstruction loss plus a latent-shape matching term.



Let's prove the claims above. For the exact identity:

$\ln p(x) - \mathcal L = \mathbb E_{Z \sim q_\phi} [ \ln p(x) - \ln \frac{p(x,z)}{q_\phi(z|x)} ] = \mathbb E_{Z \sim q_\phi} [ \ln \frac{p(x) q_\phi(z|x)}{p(x, z)} ]$

The fraction inside can be seen to be

$\frac{p(x) q_\phi(z|x)}{p(x) p(z|x)} = \frac{q_\phi(z|x)}{p(z|x)}$

From which the result follows.

This derivation then suggest an alternative expansion: suppose we do the exact same derivation but begin from just $\mathcal L$ itself. What if, instead of expanding $p(x, z) = p(z|x) p(x)$, we instead go $p(x, z) = p(x|z) p(z)$?

$\mathcal L = \mathbb E_{Z \sim q_\phi} [ \ln \frac{p(x, z)}{q_\phi(z|x)} ] = \mathbb E_{Z \sim q_\phi} [ \ln p(x|z) + \ln \frac{p(z)}{q_\phi(z|x)} ] = \mathbb E_{Z \sim q_\phi} [ \ln p(x|z) ] + D_\text{KL}( q_\phi(\cdot | x) || p( \cdot ) )$

Where the last KL terms compare against the marginal $p$ distribution of $Z$ instead of the conditional distribution.



### Setup for Variational Autoencoder

**Some preparations**

Recall that an autoencoder based on a neural network, is a pair of neural networks that learn two functions (recalling that neural networks are universal function approximator), the encoder, and the decoder. They should have the property that $\text{Dec}( \text{Enc}(x) ) \approx x$, and that $\text{Enc}(x)$ is some vector with smaller dimension than $x$. They can be considered to be a form of compression.



A Generative Model is some probability model $X \sim P$ that approximates some unknown empirical distribution $p^*$. If we have a model, we can generates synthetic samples emulating the data by doing a random sampling over $P$. A naive example algorithm to do so in general is to use the inverse-CDF method on $P$. Although the technical form of this problem seems like non-parametric density estimation, or if the model is parametric, it's just standard parametric inference, in Generative Modelling, the actual/empirical distribution is often too complicated to reasonably fit using classical methods above. One example (the one we care about in this article) is when $X$ is a tensor representing an image with dimension $(w,h,c)$ where $w$ and $h$ are the width and heights, and $c$ is the channels (such as RGBA). Then the "empirical distribution" could be something like "a random image from the internet" or "a random image produced by a human artist" (with all the copyright controversies that entails), or "a random image from an art sharing website with at least B+ aesthetic rating", etc.



There is a conditional variant of generative modelling also. In that extension, we instead have $(X, Y) \sim P$, where $X$ is the data, and $Y$ is a conditioning label. If we successfully learn a model that is a close fit to the ground truth of real world distribution, then we can perform conditional sampling from the function $p(X|Y)$, where the value of $Y$ is given/controlled by the user of the model. In the main example, the conditioning label would be a text caption of the image (during training phase), and user can supply a text prompt for $Y$ to control the image generation process.



**Math modelling and objective function**

Given the challenge of having a highly complex empirical distribution with humanly meaningful patterns, a latent model is one way to produce a more flexible family of possible distribution (Refer to example 1 and 2 above). Instead of modelling $X$ directly, we design a model over the pair $(X, Z) \sim P$, where $Z$ is some latent variables, and we are responsible to design the full $P$ ourselves.



For image generation, $Z$ will be a perceptual compression of $X$ - it will have shrinked dimension, with the color channels replaced by semantic channels that represent "the extent to which some abstract semantic feature is present at a rough spatial location". The reason this may work is because for what we actually care about which is whether the images look realistic, minor perceptual noise that doesn't change the image's semantic can usually be filtered away without affecting the perceived quality or degree of realism.



Similar to example 2, we will design $P$ by specifying the piror of $Z$ as well as the likelihood $X|Z$. One powerful way to increase flexibility is to set $Z$ piror to be standard Gaussians, and then let $X := f_\theta(Z) + E$, where $E \sim \mathcal N(0, I)$ is a standard Gaussian noise, and $f_\theta$ is a deterministic but complicated function, modelled by a neural network.



The $(X, Z) \sim P_\theta$ we have just defined is a latent (unconditional) generative model, and $f_\theta$ is, in this context, the decoder network. Sampling from this model can be done by first sampling $Z$, and then sampling $X|Z$ by the formula above. At the same time, as it is a parametric density estimation model, one way to "train" it, or to fit it to the empirical distribution, is, by example 3, to perform a maximum likelihood estimates of the parameter $\theta$. Copying the solution in example 3, we can do Stochastic Gradient Descent to maximize $\ln p_\theta(x)$, where $x$ is sampled empirically/from the dataset.



As directly evaluating either the theoretical/model marginal $p_\theta(x)$ itself, or the theoretical model for encoding/doing exact inferencing of the latent from the data, i.e. $p_\theta(z|x)$, would be analytically intractable, we apply the Variational Bayesian Inference Idea above and introduce a proxy function $q_\phi(z|x)$, which in this context is modelled as the encoder neural network with parameter $\phi$. Its role is to perform approximate inferencing of the latent. Before precisely defining the form of $q_\phi$, we first rewrite the formula above:

$Z \sim \mathcal N(0, I), X|Z \sim \mathcal N(D_\theta(Z), I)$

Where $D_\theta$ is a deterministic neural network that output the model mean.

Then, $Z|X \sim_{q} \mathcal N(E_\phi(X), \sigma_\phi(X)^2 I )$



With this setup, we will modify the training objective to the ELBO:

$\mathcal L_{\theta, \phi} = \mathbb E_{Z \sim q_\phi} [ \ln \frac{p_\theta(x, z)}{q_\phi(z|x)} ]$

To see why this make sense, we apply the main results derived in the section above.

First, we have that $\mathcal L_{\theta, \phi} = \ln p_\theta(x) - D_\text{KL}( q_\phi( \cdot | x) || p_\theta( \cdot | x) ).$

This is easy to interpret as a joint optimization with two objectives:

1. Maximize the likelihood of our density model/decoder part just like before

2. Minimize the reverse KL distance of the encoder from the theoretical encoder from "inverting" the decoder.

With the other main result, we can also write

$\mathcal L_{\theta, \phi} = \mathbb E_{Z \sim q_\phi} [ \ln p_\theta(x|z)] - D_\text{KL}( q_\phi( \cdot | x) || p_\theta( \cdot) )$

This can be interpreted as follows.

1. The first term is the end-to-end reconstruction loss. Given an input $x$, it first approximately infer the latent using the encoder, then generate a sample output $x'$ using the decoder from the inferred latent. The term inside is then the log likelihood density of reproducing the same input.

2. The second term is latent-shape matching. It is a regularizing term to try to make the encoder infer latent whose distribution stay close to the piror (a standard Gaussian) on expectation. Although we can still expect the individual instances of an inferred latent to deviate from 0, our aim would still be achieved if this term ensure that the inferred latent value doesn't become larger and larger and escape. This is also the reason why we allow modulating the variance only for the encoder part - a mean substantially different from 0 can be compensated for by decreasing the variance.

Finally, as there are known formula for the information theoretic quantities related to Gaussians, we can substitute them into the ELBO to get a more concrete formula:

$\mathcal L_{\theta, \phi} = -\frac{1}{2} \mathbb E_{Z \sim q_\phi} [ || x - D_\theta(z) ||^2 ] - \frac{1}{2} (N \sigma_\phi(x)^2 + || E_\phi(x) ||^2 - 2N \ln \sigma_\phi(x) ) + \text{ Const.} $

Thus ultimately, we want to minimze the mean square error (MSE) of the reconstructed image compared to the original, in the pixel by pixel sense and using the Euclidean distance. We also want to minimize the encoded latent's mean around 0 as well as make the variance stay close to 1.





## Denoising Diffusion Probabilistic Model

The core technology powering modern image generation AI is the diffusion model. Here we follow the formulation of an early proposal in the DDPM paper (Denoising Diffusion Probabilistic Model).

(we will see later that after training, sampling from it to actually generate image will be very slow if using a naive approach. Hence the need for special sampling algorithm which we'll cover in the next major section. One variant of DDPM, called DDIM, address this in a different way though)



The basic idea is the following: Starting from a random image $x_0 \sim q(x)$, where $q$ is an empirical distribution, we perform the **forward diffusion process** by successively adding noise to it in a controlled manner, to result in a sequence $x_1, \cdots, x_T$. We will design it so that in the limit, it converges to white noise. Then, we will design and train a neural network that can predict the noise in each step to perform a **backward diffusion process**. When done, we can then generate image by starting from a random $x_T$ and doing the backward diffusion process.



This may seem counter-intuitive...



Now let's do the math. First we define a quality of life notation: we will use $x_{i\ :\ j}$ (note the Colon) to denote a slice of the sequence $\{ x_k \}_{k=0}^T$ , i.e. $x_i, x_{i+1}, \cdots, x_j$ (inclusive).

The formal definition of the forward diffusion process is as follows:

Let $\beta_t \in (0, 1)$ be a sequence of real numbers, called the **noise schedule**. Let $x_0 \sim q(x)$ be sampled empirically, then let $\{ x_k \}_{k=0}^T$ be a Markov chain whose transition probability is defined as $q(x_t | x_{t-1}) \sim \mathcal N (\sqrt{1 - \beta_t} x_{t-1}; \beta_t I).$



Readers with a sharp eye may have noticed that we are actually also shrinking the data at each step before adding the noise. The reason is that without such shrinking, it will basically be just the standard Brownian motion, which will have overall variance exploding to infinity as time goes on.



To work with diffusion processes, derive various properties or prove things etc, it is important that we get a handle on a nice expression for what we defined above. It turns out that the **reparametrization trick** will give us something even better - a direct, closed form formula for sampling at any time step without recurrence relation.

Let $\alpha_t = 1 - \beta_t, \bar \alpha_t = \prod_{i=1}^t \alpha_i$. Let $\epsilon_t \sim \mathcal{N}(0,1)$ be iid sequence of Gaussian. We can rewrite the definition above as $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}$. The $\sqrt{\alpha_t}$ factor corresponds to the shrinking factor of $\sqrt{1-\beta_t}$ which appeared in the mean of the normal distribution parameter in the definition. The $\sqrt{1-\alpha_t} = \sqrt{\beta_t}$ factor we attach to the standard Gaussian is because scaling a random variable by a scalar will scale its variance by the *square* of that scaling factor. Moreover, doing iteration of the recurrence relation by repeated substitution, we have:

$$
x_t = \sqrt{\bar \alpha_t} x_0 + \sqrt{1 - \bar \alpha_t} \epsilon
$$

where $\epsilon$ here is a merge of the $\epsilon_t$'s. This expression also tell us that provided $\bar \alpha_t$ converge to zero, our sequence will converge to standard white noise.

Our main mathematical problem is to perform the reverse diffusion process. The theoretically exact one is analytically untractable, but we can derive a closed form expression if we cheat a bit *and additionally condition on the original image*. Indeed it is Gaussian distributed and we can again give closed form formula for the mean and variance.

Let's show the result first:

$$
q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)} \\
\sim \exp \left( -\frac{1}{2} \left( \frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar \alpha_{t-1}} x_0)^2}{1-\bar \alpha_{t-1}} - \frac{(x_t - \sqrt{\bar \alpha_t} x_0)^2}{1-\bar \alpha_t} \right) \right) \\
\vdots \\
= \exp \left( -\frac{1}{2} \left( (\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar \alpha_{t-1}}) x_{t-1}^2 - (\frac{2\sqrt{\alpha_t}}{\beta_t} x_t + \frac{2\sqrt{\bar \alpha_{t-1}}}{1 - \bar \alpha_{t-1}} x_0)x_{t-1} + C(x_t, x_0) \right) \right)
$$

The algebra can be tedious so here's a summary of what happened:

- In the first line, we apply a variant of the standard Bayes theorem. Originally we have $q(x_{t-1}|x_t) = \frac{q(x_t|x_{t-1}) q(x_{t-1})}{q(x_t)}$. Turns out we can just slip in $x_0$ as an additional condition into all terms in the formula and it will still be valid (why?)

- Then we expand each factor on right hand side. In general, for $p(y|x)$ where $y = ax + z, z \sim \mathcal N(0, \sigma^2)$, the probability density is simply the probability that the gaussian lands on the value of $y-ax$, which we substitute into the pdf expression and scale by the variance. In the expression above, the first term uses the original recurrence, while the other two uses the direct sampling formula we derived. Notice the minus sign for the third term as it's in denominator.

- We then have to do some brute force elementary algebra that basically expand everything and recollect coefficient by viewing it as a multivariate polynomial of $x_0, x_{t-1}, x_t$. However, we really only cares about monomial terms that contain non-zero powers of $x_{t-1}$, so instead of actually expanding it all out, we can more strategically find all possible combinations in the expansion that may involves $x_{t-1}$ and leaves out the rest. The third line grouped it by the power of $x_{t-1}$.

Now let's interpret the expression. Given the value of $x_t, x_0$ (so that we may treat them as constant in this context), we find that the pdf of the reverse process exhibit a form that can be recast to fit the formula for a Gaussian pdf. To make the whole thing fit, we should complete the square of the polynomial expression above.

In particular, because $\frac{(X-\mu)^2}{\sigma^2} = \frac{X^2}{\sigma^2} - \frac{2\mu}{\sigma^2}X + \frac{\mu^2}{\sigma^2}$, we can match coefficients to solve for $\tilde \beta_t$ (variance) and $\tilde \mu_t(x_t, x_0)$ (the mean). Skipping the algebra, we list the result:

$\tilde \beta_t = \frac{1 - \bar \alpha_{t-1}}{1 - \bar \alpha_t} \cdot \beta_t$

$\tilde \alpha_t (x_t, x_0) = \frac{\sqrt{\alpha_t} (1 - \bar \alpha_{t-1}) }{1 - \bar \alpha_t} x_t + \frac{\sqrt{\bar \alpha_{t-1}} \beta_t }{1 - \bar \alpha_t} x_0 = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar \alpha_t} } \epsilon \right)$

(For the variance, a hint is to clear the denominator, and that $\beta_t + \alpha_t (1 - \bar \alpha_{t-1}) = (\alpha_t + \beta_t) - \alpha_t \bar \alpha_{t-1} = 1 - \bar \alpha_t$ )

(For the mean, just multiply $\tilde \beta_t$ onto the coefficient of $x_{t-1}$ term by term.)

(For the alternative expression of the mean, derive $x_0 = \frac{1}{\sqrt{\bar \alpha_t}} ( x_t - \sqrt{1 - \bar \alpha_t} \epsilon)$ from the direct sampling formula, then substitute it into $x_0$ and do some algebra simplifications. Notice that $\epsilon$ is the merged noise)

(You may find it helpful to notice that $\frac{\sqrt{\bar \alpha_{t-1}}}{\sqrt{\bar \alpha_t}} = \frac{1}{\sqrt{\alpha_t}}$. Also, the hint when computing the variance while clearing denominator above will be useful again.)
















