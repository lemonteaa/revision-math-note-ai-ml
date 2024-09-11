# Neural Tangent Kernel Theory and Related Topics



> TODO: This is a DRAFT.



## Prerequisite



## Setup

Let's revisit the basic setup of supervised learning in ML. For simplicity, we consider models with scalar output (such as a binary classification model). Let the model be a parametrized class of functions $f(x; \theta)$. $x$ is the input and $\theta$ is the parameter. Both inputs are vector. For an intro to NTK theory we will restrict ourself to square loss: $l(x, y) := \frac{1}{2} | x - y |^2$. Then recall that the supervised learning problem is to minimize the empirical loss:

$\min_\theta \sum_i l(f(x_i ; \theta), y_i)$

Where $\{ (x_i, y_i) \}_i$ is the training dataset with sample inputs and corresponding labelled/ground truth outputs.

One general algorithm to solve this is the gradient descent. If we take the continuum limit of that algorithm, it can also be modeled as a continuous time dynamical system through an ODE:

$\frac{d\theta}{dt} = - \nabla_\theta l(\theta)$

Where we abuse notation a bit and let $l$ here denote the empirical loss above after substituting the dataset and model in. This is also called the gradient flow as the parameter moves along the vector field produced by the gradient of the loss function.



## Transition to function space, and re-interpretation

The first idea of NTK is to change our view to the function space. Recall that model learning can also be interpreted as function approximation/fitting where we try to have the function pass through the training data points. So, instead of looking at hidden parameters, we look at the observable fitted value of the function and how that evolves over time.



Let $u(t)$ be a column vector whose i-th entry, $(u(t))_i := f(x_i; \theta(t))$. It is the model evaluated at the training datasets. We also define the residual $r(t) := u(t) - y$. ($y$ is the vector of training data labels, so this is the error from a perfect fit and will approach 0 as the fit improves)



We can derive an ODE for $r(t)$ also. To do this, we expand the gradient in the first ODE above under square loss:

$\nabla_\theta l(\theta) = \nabla_\theta \sum_i \frac{1}{2} | f(x_i ; \theta) - y_i |^2 = \sum_i (f(x_i ; \theta) - y_i) \nabla_\theta f(x_i ; \theta)$



Let the matrix have entry $G_{ij} := \nabla_\theta f(x_i ; \theta_j)$, then the expression above is $G^T r(t)$.

Then apply chain rule again:

$\left( \frac{dr}{dt} \right)_i = \frac{d}{dt} f(x_i ; \theta(t)) = \langle \nabla_\theta f(x_i ; \theta), \frac{d\theta}{dt} \rangle$

Substitute the original ODE and simplifying, we get:

$\frac{dr}{dt} = - H_t(x) r(t)$

Where $H_t(x) = GG^T$ is a time-varying matrix whose $i,j$ entry is $\langle \nabla_\theta f(x_i ; \theta), \nabla_\theta f(x_j ; \theta) \rangle$

This matrix is known as the NTK kernel. Notice that in the exact case, this is a realization of a kernel function at the training dataset, where the kernel function is $K(x, x') := \langle \nabla_\theta f(x ; \theta), \nabla_\theta f(x' ; \theta) \rangle$. It should be obvious that it is symmetric and positive semidefinite (and so is the corresponding matrix - why?). It can hence be interpreted in the RKHS theory. One obvious feature map from reading the formula above is the map $x \mapsto \nabla_\theta f(x; \theta)$, although other feature map derived from RKHS theory is also possible.



The only issue is that the matrix is time-varying. NTK theory is fundamentally about various conditions upon which the matrix will become effectively constant over time (it may or may not be actually constant throughout the whole parameter space - NTK theory only cares that it effectively is, such as when the relevant region in the parameter space that we move about is such that the matrix doesn't change). As we will see, there is more than one situation where this happen. In this section, we will solely perform a thought experiment on what the implication would be if $H_t$ is effectively constant.



A first consequence is that in that case, we will be able to show a global convergence result for the training algorithm in continuous time limit, without needing to impose convexity assumption. This is considered a major breakthrough in academia and is one reason why NTK theory attracted so much attention (but not without controversy). To see this, compute the time derivative of the loss function:

$\frac{dl}{dt} = \langle \nabla_\theta l(\theta), \frac{d\theta}{dt} \rangle = -(r^T G) (G^T r) = - r^T H r \leq -\lambda_m r^T r = -2\lambda_m l(t)$

Where $\lambda_m$ is the smallest eigenvalue of $H$. Provided it is a strictly positive number, then some analysis will show that the loss will converge to 0, which is indeed a global minima (if attainable). Moreover, doing a similar spectral analysis on the ODE for $r(t)$ should also shows the residual going to zero. Thus it generally suffice to analyze the spectrum of the matrix $H$. Now do note that the details of the NTK theory lies in making these intuitive argument precise as the actual situation is only an approximately constant matrix. One particularly tricky point is the "apparently circular argument" - in one situation we need the property "parameter only have small perturbation over time" to show "H is effectively constant", but we need H approximately constant to show that the parameter doesn't deviate much over time (!). This seeming contradiction can be resolved through using "induction on time".



A second consequence is about re-interpreting the model, and using it to potentially explain things like the double descent theory. Consider the linearized version of the model, so that the gradient is constant throughout parameter space:

$f(x ; \theta) = f(x; \theta_0) + \langle \nabla_\theta f( x; \theta_0), \theta - \theta_0 \rangle$

With some algebra we can see that this is basically a linear model (in the sense of linear in the parameter, but it is still nonlinear with respect to the data input). So let's change notation a bit and cleanup. Let $\phi(x) := \nabla_\theta f( x, \theta)$ be the feature map (the exact value of $\theta$ doesn't matter), and shifting the origin values a bit we may consider the model $R(x; \theta) := \langle \phi(x), \theta \rangle$, which is manifestedly a linear model with the kernel trick. We have $\nabla_\theta R = \phi(x)$ independently of the parameter value, and so the resulting H matrix would have entry $\langle \nabla_\theta R( x_i; \theta), \nabla_\theta R( x_j; \theta) \rangle = \langle \phi(x_i), \phi(x_j) \rangle = K(x_i, x_j)$

so it is exactly the same.

Now the obvious thing to do is to ask if we can show that the model is effectively the same as linear regression with kernel trick, why can't we just use the known explicit formula for the fitted model? Well, one stumbling block is that in some scenarios, we may have the parameter space being basically infinite dimensional. To overcome this, we again can use the RKHS theory, this time using the representer theorem.

The result of using that theorem is that we can still get to the same optima by reparametrizing and restricting ourself to a finite dimensional subspace of the parameter space. Now the model is $R'(x; a) = \sum_i a_i K(x_i, x)$. To see how is this a re-parametrization, expand using feature map: $\cdots = \sum_i a_i \langle \phi(x_i), \phi(x) \rangle = \langle \sum_i a_i \phi(x_i), \phi(x) \rangle$. Moreover, with this re-parametrization, the linear regression model can be written as $Y = Ka + e$, where $Y$ is a column vector of the labels, $K_{ij} = K(x_i, x_j)$ is the same as the $H$ matrix, $a$ is a vector of the new parameters, and $e$ is the error/residual. A final trick to note is that as $K = H$ is both symmetric and positive definite (in the nice case), it is actually invertible (using the spectral theorem), so the residual is zero. So the formula does simplify compared to the general case.

There is something more we can interpret out of this. In the case of the infinite width limit of neural network, this result tell us it is effectively the same as the kernel method, where the model perform similarity search across all training data $K(x_i, x)$ and then return a weighted sum of the scores (hence the interpolation). This corresponds to the interpolation regime in the double descent theory. Moreover, we may explain this as the neural network having sufficient capacity (being extremely over-parametrized in the limit) to basically remember the full training dataset (this is also why it can get to zero loss and perfect fit on the training dataset?). Finally, a very interesting thing is that although a finite computer cannot compute anything meaningful on a literally infinitely large model directly, by the NTK theory above, we can still perform training and inference indirectly, provided we know the $H$ matrix. For example, we can evaluate the fitted model at a point outside of the training dataset:

$R'(x') = \left[ K(x', x_1), \cdots, K(x', x_n) \right] H^{-1} y$

(This formula is from rewriting the definition of model $R'$ in linear algebra and substitute the solved parameter values $H^{-1} y$ per discussion above)

The formula explicitly depends only on access to the kernel function $K(x, x')$, and many research and software library have been done to allow either deriving it analytically, or computing it using various approximation/simulation methods, for various neural network architecture (Such as CNN and Transformer, among others).



## Situations where NTK arise

### Infinite Width Limit of Neural Network



### Linearizing Models




