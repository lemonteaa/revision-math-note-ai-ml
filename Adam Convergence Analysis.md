# Adam Convergence Analysis

> Source: [[2104.14840] A Novel Convergence Analysis for Algorithms of the Adam Family and Beyond](https://arxiv.org/abs/2104.14840) 
> 
> See Also: [[1411.3803] Stochastic Compositional Gradient Descent: Algorithms for Minimizing Compositions of Expected-Value Functions](https://arxiv.org/abs/1411.3803) Lemma 2 in this paper is used above



## Some terms

- SEMA: Stochastic estimator based on moving average



## Introduction



## Setups

The author noted that a wide family of Adam related algorithms can be unified under the following description. One key observation (made by other also) is that although variants of momentum based algorithms often has complicated interlocking, intermediate steps, we can usually unroll these intermediate steps and the result is  usually that we can formulate it in terms of a standard gradient update like in vanilla gradient descent, but the gradient is estimated through an auxiliary sequence that represents an exponential average of historical gradients. The exponential average is because it is defined through recursion with weighted sum.

Then, the idea of RMSProp etc is incorporated through an coordinate-wise adaptive step size.



**Adam Style Algorithm:**

$v_{t+1} = \beta_t v_t + (1 - \beta_t) \mathcal O_{\nabla F} (x_t)$

$x_{t+1} = x_t - \eta s_t \circ v_{t+1}$

Notations:

- $0 \leq \beta_t \leq 1$ is the momentum parameter, usually chosen close to 1.

- $\mathcal O_{f}$ for any function $f$ is a stochastic proxy of that function, with oracle access model (that is we can only ask for evaluation on a finite number of specific values and access is metered for algorithmic cost). We assume the proxy is unbiased in that $\mathbb E [ \mathcal O_{f} (x) ] = f(x)$.

- $v_t$ is then the SEMA sequence. In this application it is used for estimating the gradient.

- $\eta$ is the global learning rate, and $s_t$ is a coordinate wise adaptive step size. The inner circle represent the vector coordinate wise scaling, that is, $x \circ y := (x_1 y_1, \cdots, x_i y_i, \cdots, x_n y_n)$. It is a good notational shortform and has many property that come from carefully reasoning from component-wise to global. As an example, $\lambda (x \circ y) = (\lambda x) \circ y = x \circ (\lambda y)$ for scalar $\lambda$ and in particular $\lambda (x \circ y) = (\sqrt{\lambda} x) \circ (\sqrt{\lambda} y) .$

- Finally, $x_t$ is the actual iterates updated by the gradient with both SEMA and adaptive scaling deployed.

We will also assume that $\nabla F$ is $L_F$-Lipschitz continuous: $\| \nabla F(x) - \nabla F(y) \| \leq L_F \| x - y \|$ . Moreover, we need to assume the adaptive scaling is globally  $L_\infty$ bounded: $c_l \leq | s_{t, i} | \leq c_u$.



Let $\tilde \eta_t := \eta s_t$, and then $\tilde \eta_{t, i} = \eta s_{t, i}$ satisfy $\eta c_l \leq \tilde \eta_{t, i} \leq \eta c_u$. This notation will be used in lemma 5. A general consequence of the $L_\infty$ bound is that the coordinate wise scaling can be bounded: we have that $ | s \circ x | $ (the simple absolute value applied to a vector means apply it coordinate-wise) is upper/lower bounded by $c_u | x |$ or $c_l | x |$ coordinate-wise, and hence their corresponding vector norm will also have the corresponding bound.



Finally, in actual instantiation of the algorithm we will need to specify a definition for $s_t$. One family of choice is to set $s_t := \frac{1}{\sqrt{u_t} + G}$, where $u_t$ is some sequence that forms a second order moment estimate of the gradients, and $G$ is a number to give a strict upper bound if the moment goes to zero.



**Convergence Type:**

As the paper didn't assume that $F$ itself is convex, it uses the notion of convergence where $\mathbb E [ \| \nabla F (x_t) \|^2 ] \leq \epsilon$. Notice that if we add an assumption that $F$ is $\mu$-strongly convex, then the PL inequality for strongly convex function would allow us to bound $\| x_t - x^* \|$ as a consequence.



The Theorem 6 in the paper shows that it is not the expected value of gradient norm-square at individual iterate, but an ergodic type average over the sequence, that satisfies the bound. This is why the algorithm in the paper randomly select $I = \{ 0, 1, \cdots, T \}$ and return $x_I$ instead. The law of total expectation would gives $\mathbb E [ \| \nabla F (x_I) \|^2 ] = \mathbb E_{i \sim I} [ \mathbb E [ \| \nabla F (x_i) \|^2 | I = i] ] = \sum_i \frac{1}{T+1} \mathbb E [ \| \nabla F(x_i) \|^2 ] = \mathbb E [ \sum_i \frac{1}{T+1} \| \nabla F(x_i) \|^2 ] $



## Proof of Variance Recursion Property

Let's first prove a general inequality:

$\| a + b \|^2 \leq (1 + \epsilon) \| a \|^2 + (1 + \frac{1}{\epsilon} ) \| b \|^2$ for all $\epsilon > 0$.

We first move term to left hand side:

$2 \langle a, b \rangle = \| a + b \|^2 - \| a \|^2 - \| b \|^2 \leq \epsilon \| a \|^2 + \frac{1}{\epsilon} \| b \|^2$

Then apply a normalization trick in elementary inequality:

$\langle a, b \rangle = \langle \sqrt{\epsilon} a, \frac{1}{\sqrt{\epsilon}} b \rangle $

So letting $a' = \sqrt{\epsilon} a, b' = \frac{1}{\sqrt{\epsilon}} b$, the inequality reduces to $2 \langle a', b' \rangle \leq \| a' \|^2 + \| b' \|^2$ which is true by expanding $\| a' - b' \|^2 \geq 0$.



For the actual proof, we adapt part of the Wang paper Lemma 2 into the notation of this paper.

A key first step is to relate the tracking error terms $z_{t+1} - h(x_t)$ recursively. To this end let $e_t = (1 - \gamma_t) (h(x_t) - h(x_{t-1}))$. By using this extra term as well as definition of $z_{t+1} = (1 - \gamma_t) z_t + \gamma_t \mathcal O_h (x_t)$ and substituting and simplifying, we get

$(z_{t+1} - h(x_t)) + e_t = (1 - \gamma_t) (z_t - h(x_{t-1})) + \gamma_t ( \mathcal O_h (x_t) - h(x_t))$

We then expand the expected norm square of this quantity's right hand side. A key point is that the expectation is taken only to things in step $t$ and so only $\mathcal O_h(x_t)$ and $h(x_t)$ come under the expectation. As a result, because we assume the proxy is unbiased, the cross term will involves $\mathbb E [ \mathcal O_h (x_t) - h(x_t) ] = 0$, so it is eliminated.



Apply the elementary inequality above with $\epsilon = \gamma_t$,

$\| z_{t+1} - h(x_t) \|^2 = \| (z_{t+1} - h(x_t) + e_t ) - e_t \|^2 \leq (1 + \gamma_t) \| \cdots \|^2 + (1 + \frac{1}{\gamma_t}) \| e_t \|^2 $

Taking expectation, substutiting and expanding, we end up with these terms:

*Recursion term:*

$(1 + \gamma_t) (1 - \gamma_t)^2 \| z_t - h(x_{t-1}) \|^2 $. We upper bound the coefficient by: $\cdots = (1 - \gamma_t^2)(1 - \gamma_t) \leq (1 - \gamma_t)$.

*Proxy Variance term:*

$(1 + \gamma_t) \gamma_t^2 \mathbb E [ \| \mathcal O_h (x_t) - h(x_t) \|^2 ]$. The coefficient can be upper bounded as $(1 + 1) \gamma_t^2 = 2 \gamma_t^2$ because we constrained $\gamma_t \leq 1$.

*Lipschitz term (from the last $e_t$ related term):*

$(1 + \frac{1}{\gamma_t} ) (1 - \gamma_t)^2 \| h(x_t) - h(x_{t-1}) \|^2$, by substituting the definition of $e_t$. The norm square can be upper bounded by invoking the Lipschitz condition as $L^2 \| x_t - x_{t-1} \|^2$, while the coefficient is upper bounded: $\cdots = \frac{1}{\gamma_t} (1 + \gamma_t) (1 - \gamma_t)^2 = \frac{1}{\gamma_t} (1 - \gamma_t^2) (1 - \gamma_t) \leq \frac{1}{\gamma_t}$





## Proof of Lemma 5

We first do the standard second order Taylor upper bound (need regularity assumption on $F$). The main content here is how to proceed next.



First, note that in our algorithm, $x_{t+1} - x_t = - \tilde \eta_t \circ v_{t+1}$. After substituting, the second order term becomes $\frac{L_F}{2} \| \tilde \eta_t \circ v_{t+1} \|^2$. To expand the first order term, we use the easy form of the polarization identity $ -2 \langle x, y \rangle = \| x - y \|^2 - \| x \|^2 - \| y \|^2$ to expand

 $ - \langle \nabla F (x_t), (\tilde \eta_t \circ v_{t+1}) \rangle = -\langle (\sqrt{ \tilde \eta_t} \circ \nabla F (x_t)), (\sqrt{ \tilde \eta_t } \circ v_{t+1}) \rangle$

(This identity is because component-wise scaling is associative and distribute over inner products: $(x \circ y) \circ z = x \circ ( y \circ z)$ and $\langle (x \circ y), z \rangle = \langle x, (y \circ z) \rangle $, giving us

$ \langle a, (b \circ c) \rangle = \langle a, ( (\sqrt{b} \circ \sqrt{b} ) \circ c ) \rangle = \langle a, ( \sqrt{b} \circ ( \sqrt{b} \circ c)) \rangle = \langle ( \sqrt{b} \circ a), (\sqrt{b} \circ c) \rangle $

)



Then we notice that for any vector, $\eta_t c_l \| x \|^2 \leq \| \sqrt{\tilde \eta_t} \circ x \|^2 \leq \eta_t c_u \| x \|^2$ and apply this to each of the expanded terms.


