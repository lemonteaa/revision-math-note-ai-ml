# Reproducing Kernel Hilbert Space, the Kernel trick, and SVM

This is a revision notes based on materials from wikipedia.

I assume strong level of familarity with functional analysis and linear algebra, as well as some background on statistical learning theory.

## Shorthands

- PSD = Positive Semidefinite
- RKHS = Reproducing Kernel Hilbert Space
- SVM = Support Vector Machine


## Background: Mercer theorem

It is the infinite dimensional generalization of the proposition that a psd matrix is always a gram matrix. (We skip all the analytic details on continuity etc)

Let $ K: X^2 \rightarrow \mathbb{R} $ be a symmetric psd function. Define a linear operator

$ T_K(f) := \int K(x, y) f(y) dy $

Then apply the spectral theorem for compact operator - we get a countable sequence of eigenvalues (and a corresponding orthonormal basis for the eigenvectors). Now notice that

$ \langle f, T_K f \rangle = \int \int f(x) K(x, y) f(y) dx dy \geq 0$

By a continuity argument to pass from the discrete/finite definition of psd.

Then for a nonzero eigenvector $ \langle e , T_K e \rangle = \lambda \langle e, e \rangle \geq 0 \implies \lambda \geq 0$.

Finally, as $ K \mapsto T_K$ is injective, we can by direct verification shows that $ K(x, y) = \sum \lambda_i e_i(x) e_j(y) $

(Personal, selfish note: Aside from RKHS and machine learning, the theory of psd symmetric kernel function is also useful in areas like stochastic process, fourier analysis, etc, which I have a personal interest in due to working with "White Noise Theory")

## The word game of reproducing kernel hilbert space

Learning RKHS can be tricky because of the many equivalent ways to formulate it, and also because of the subtle difference of the meaning of the same symbols - they ultimately represent exactly the same "stuff" - but you must be very careful not to assume those properties/equivalence when your task is precisely to prove the equivalence (otherwise it would be circular reasoning!).

### Basic Definition and Properties

Let X be any set, H be any subset of the (set-theoretic/no analytic restriction) function space over X with vector space structure given by the usual pointwise addition/multiplication. Further suppose H is a Hilbert space with an inner product that is **not necessarily** the standard $ L^2 $ one. (because of this point, and we will have multiple, competing inner products in the future, it is extremely important to be very clear about exactly which inner product we are using, hence we will always label it)

Then we say that H is a RKHS if the evaluation linear functional (over H)

$ L_x: f \mapsto f(x) $

is bounded/continuous for all points x in X.

This already imply a whole host of thing, the most important of which is the existence of **reproducing kernels** (hence the name) as well as a kernel function.

By Riesz Representation Theorem, for each x we have an element $K_x \in H$ (the fact that it is inside H is an important detail and cannot be taken for granted in future situation as we are dealing with the full function space sometimes) representing the linear functional $L_x$, so that

$ f(x) = L_x(f) = \langle f, K_x \rangle_H $ ... (1)

The $ K_x $ are called the reproducing kernels as it "reproduces" the evaluation functional.

Now here's a small trick. What if we apply (1) to $ K_x $ itself? (We can do this only because it is inside of H) (Also makes me think of the poisson bracket etc in Hamiltonian Mechanics) Well,

$ K_x (y) = L_y (K_x) = \langle K_x, K_y \rangle_H := K(x, y)$ ... (2)

We *define* this entity to be the **kernel function** associated with the RKHS. (It is of type $ X^2 \rightarrow \mathbb{R} $)

There are immediately a number of proposition we should prove:

- $K$ is symmetric and psd. This follows by working with the inner product expression of $K$ via linearity and nonnegativity of the norm.
- The partial function derived from $K$ (by partial application of one argument) is the reproducing kernels $K_x$, which is in $H$. To see this, check that such a function would be like $ y \mapsto K(x, y) = K_x(y) $ (by (2)), and hence is in fact just $K_x$.
- As a consequence, in the expression $K(x, y) = \langle K_x, K_y \rangle_H$ we can take $K_x$ etc as the partial function from the 2-argument $K$ instead.

### Moore-Aronszajn theorem

We have seen that given any RKHS is an associated kernel function. It turns out this function already contain all information of the RKHS itself! That is, given a kernel function $K$ that is symmetric and psd, there exists, up to isomorphism, an unique RKHS $H_K$ whose associated kernel function is $K$. Hence we can just specify $K$ itself.

The proof is basically a standard exercise in doing free algebra/universal construction type of proof (which you should be familiar with if you worked in any area of pure math at undergrad level). But let's work it out - it is illuminating.

To avoid confusion, we denote the partial function of $K$ by $\tilde{K_x}$, then we have $\tilde{K_x} (y) = K(x, y)$.

What is the space $H_K$? Well, from last section, we see that at minimal we need to include the reproducing kernels (i.e. the partial functions of the associated kernel). Then, as we are dealing with Hilbert space, this imply we need at least the completion of the linear span of them. It turns out this is enough, and is also the only solution up to isomorphism.

Now let's define an inner product by requiring

$\langle \tilde{K_x}, \tilde{K_y} \rangle_{H_K} := K(x, y)$

Then extend by linearity and continuity to the whole of $H_K$. (We took this definition in order to satisfy (2))

Then $ \langle \tilde{K_y}, \tilde{K_x} \rangle_{H_K} = K(x, y) = \tilde{K_y}(x) = L_x(\tilde{K_y})$. By the same linearity and continuity argument, we then have

$ \langle f, \tilde{K_x} \rangle_{H_K} = L_x(f) = f(x)$ for all $ f \in H_K$, hence the $\tilde{K_x}$ are the reproducing kernels, and $H_K$ is in fact a RKHS whose associated kernel is $K$.

(Skipped the uniqueness part)

### Feature maps

A feature map is any function (not necessarily linear) $\phi : X \rightarrow F$ where F is some hilbert space. Note that any feature map give rise to a symmetric psd kernel (which then implicitly defines a RKHS from the sections above) by defining 

$K(x, y) := \langle \phi(x), \phi(y) \rangle_F$...(3) 

(Notice the inner product now take place in $F$ and not $H$, hence it is not exactly the same as (2)). The reason it is symmetric and psd is the same argument used above, but applied to $F$ instead of $H$.

We can go the opposite way too: given any symmetric psd kernel, we can find some feature map so that (3) holds. And there is in fact more than one way to do so.

Example 1: Take $F = H$ and the feature map is $x \mapsto K_x$.

Example 2 (spectral representation): By Mercer's theorem, we have that

$ K(x, y) = \sum \lambda_i e_i(x) e_i(y) = \sum \left( \sqrt{\lambda_i} e_i(x) \right)  \left( \sqrt{\lambda_i} e_i(y) \right)$. (it is crucial that the eigenvalues are nonnegative)

Hence we can take $F = l^2$ with the standard inner product, and the feature map sends x to the countably infinite sequence $ ( \sqrt{\lambda_1} e_1(x), \sqrt{\lambda_2} e_2(x), \cdots )$

### Constructing RKHS from Feature Maps

Let $H_\phi$ be the space of functions $X \rightarrow \mathbb{R}$ that are pullback of linear functional on $F$. As $F$ is a Hilbert space, by Riesz representation we are considering functions of the form $x \mapsto \langle w, \phi(x) \rangle_F$. We can see that the function's data is captured by the vector $w \in F$ alone. Actually there's a bit more - we need to consider uniqueness. By linearity, two vectors $w$ and $w'$ represent the same pullbacked function if and only if their differences satisfy $\langle w - w', \phi(X) \rangle_F = 0$, or in other words $w - w' \in (\text{im} \phi)^\prep$. One nice thing about prep space is that it is always a linear closed subspace, even if the input is an arbitrary subset. Hence $H_\phi$ as a Hilbert space should be isomorphic to $F/Y$, $Y = (\text{im} \phi)^\prep$.

We recall how an inner product can be imposed on the quotient space. For any equivalence class pick a cannonical representative to be the one with shortest distance to the origin/smallest norm. This can be done because the equivalent class is a closed affine set, so we can apply the Hilbert Projection Theorem. Then just apply the original inner product in the space. Linearity works out because projection is also linear.

The resulting space is indeed a RKHS: the result of evaluation at a point $x$ satisfies $| \langle w, \phi(x) \rangle | \leq || w || || \phi(x) ||$ and so $|| \phi(x) ||$ is the norm of $L_x$.

To see formula (3), notice that as our inner product now take place in $F$, the evaluation functional is in effect given by taking inner product of the representing vector against $\phi(x)$. That is, $\phi(x)$ is the representing vector of $L_x$ (!).

## Representer Theorem

Consider the empirical risk minimization problem on a RKHS with norm regularization. That is, given a RKHS $H$, a finite collection of tuple $(x_i, y_i)$ where $x_i \in X$ and $y_i \in \mathbb{R}$, we want to find/learn a function $f \in H$ that best approximate the given data - $f(x_i)$ should be close to $y_i$. The regularization means that we want $\| f \|_H$ to be small.

Concretely, we want to find

$f^* = \text{argmin}_{f \in H} E(x_1, y_1, f(x_1), \cdots) + g( \| f \|_H )$

The theorem then states that we can always express such a solution as finite linear combination of the reproducing kernels at the training points $K_{x_i}$.

To see why this is the case, first note that we can always write $f^*$ as such a sum, plus a component that lies in the prep space of the span of those kernels:

$ f^* = \sum \alpha_i K_{x_i} + v$

Then, by the reproducing kernel property,

$ f^*(x_j) = \langle f^*, K_{x_j} \rangle_H = \cdots = \sum \alpha_i \langle K_{x_i}, K_{x_j} \rangle_H$ 

Note that $v$ vanished as it is prependicular to $K_{x_j}$, hence the expression is independent of $v$.

Now, because $v$ is prependicular to the remaining part of the expression $\sum \alpha_i K_{x_i}$, we can remove $v$ - it will cause the regularization term to decrease (by Pythagorean theorem) while $f^*(x_j)$ remains unchanged for all $x_j$, hence the empirical risk is also unchanged.

## Relation to SVM

