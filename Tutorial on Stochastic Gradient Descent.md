# Tutorial on Stochastic Gradient Descent

## Reminder of basic math

**Symmetric positive (semi-)definite matrix**

A matrix $A$ is **symmetric positive (semi-)definite**, often written in shorthand **spd**, if $A = A^T$ and for all $x \in \mathbb{R}^n$, $x^T A x = \langle x, Ax \rangle \geq 0$.

By spectral theorem, a symmetric $A$ is diagonal after a change of basis that is unitary, and then we have $A = U^T D U \implies x^T A x = x^T (U^T D U) x = (Ux)^T D (Ux)$. Thus a symmetric $A$ is positive (semi-)definite iff all eigenvalues are nonnegative.

Then, define an equivalence relation: $A \succeq B \iff A - B \text{ is pd}$.

**Multivariate Taylor expansion up to second order**

We have that $f(y) = f(x) + \langle \nabla f(x), y - x \rangle + \frac{1}{2} (y - x)^T \nabla^2 f(z) (y - x)$, where $z$ lies on the line segment joining $x$ and $y$.

## Gradient Descent

In the simpliest formulation, we have a recurrence relation:

$x_{n+1} := x_n - \eta \nabla f(x_n)$.

There are many kinds of convergence criteria. Since this is a tutorial, we will use the nicest one possible even if somewhat unrealistic and/or make the proof "too easy".

Consider the following assumptions:

1. $\nabla f$ is Lipshitz Continuous. This implies that $\nabla^2 f \preceq \beta I$ almost everywhere.

2. Strong convexity: $\nabla^2 f \succeq \mu I$. (Note that vanilla convexity is just $\nabla^2 f \succeq 0$.)

(*) Notice that $A \succeq B \implies A - B \text{ pd} \implies x^T(A-B)x \geq 0 \implies x^T A x \geq x^T B x \ \forall x$



### Descent Lemma

The basic idea here is that since a function is locally approximately linear, if we only move slightly in the right direction, we can expect the function to decrease. However, the second order correction terms may cancel out this effect especially if we move too far. Therefore, we will be using the Taylor expansion with a quantified error term to be able to control/bound this correction such that it will not overturn the main effect.

So, with assumption 1 and substitute into the Taylor expansion:

$f(x_{n+1}) = f(x_n) + \langle \nabla f(x_n) , -\eta \nabla f(x_n) \rangle + \cdots$

And the error term is, by (*), upper bounded by $\frac{1}{2} (-\eta \nabla f(x_n))^T (\beta I) (-\eta \nabla f(x_n))$, so the expression simplifies to:

$f(x_n) - \eta \| \nabla f(x_n) \|^2 + \frac{\eta^2 \beta}{2} \| \nabla f(x_n) \|^2$. Hence, grouping coefficients, we consider the quadratic function $\eta \mapsto -\eta + \frac{1}{2} \eta^2 \beta = \eta \left( \frac{\eta \beta}{2} - 1 \right)$. You may plot the graph, but basically, it is a quadratic that opens upward (because the square term has positive coefficient), has root at $0$ and $\frac{2}{\beta}$, has the central vertical axis at $\eta = \frac{1}{\beta}$, where it attains a minimum value of $-\frac{1}{2\beta}$. Let the roots be A ($\eta = 0$) and B ($\eta = \frac{2}{\beta}$), and the vertex/minimum point be C.

Then by convexity, the line segment AB lies above the curve in that interval of $\eta$, and the slope of the line is $\frac{ -\frac{1}{2\beta} }{ \frac{1}{\beta} } = -\frac{1}{2}$. Hence, for $\eta \in (0, \frac{1}{\beta} ], -\eta + \frac{\eta^2 \beta}{2} \leq -\frac{\eta}{2}$, and then we have that

$f(x_{n+1}) \leq f(x_n) - \frac{\eta}{2} \| \nabla f(x_n) \|^2$.

### PL Inequality

The strict convexity assumption is stronger, but one important thing it buys us is that roughly speaking, the further along we move away from the global minima, the larger the gradient vector norm would be. The PL inequality will quantify this intuition in a really nice way.

Again, taking assumption 2 and substituting to Taylor, with similar algebra, we get:

$f(y) \geq f(x) + \langle \nabla f(x), y - x \rangle + \frac{\mu}{2} \| y - x \|^2$.

Now here's a tricky but powerful idea: Consider $x$ fixed and then optimize with respect to $y$ on both side, the left hand side's minima is $f^*$. (Recall that convex function has unique global minima) What about the right hand side? Well, we can complete the square:

$\frac{\mu}{2} \| (y - x) + \frac{\nabla f(x)}{\mu} \|^2$

$= \frac{\mu}{2} \| y - x \|^2 + 2 \cdot \frac{\mu}{2} \langle y - x, \frac{\nabla f(x)}{\mu} \rangle + \frac{\mu}{2} \| \frac{\nabla f(x)}{\mu} \|^2$

$= \frac{\mu}{2} \| y - x \|^2 + \langle y - x, \nabla f(x) \rangle + \frac{1}{2\mu} \| \nabla f(x) \|^2$

Hence,

$f^* \geq \min_y \left( f(x) + \frac{\mu}{2} \| (y - x) + \frac{\nabla f(x)}{\mu} \|^2 - \frac{1}{2\mu} \| \nabla f(x) \|^2 \right) = f(x) - \frac{1}{2\mu} \| \nabla f(x) \|^2$We may then rearrange to get

$\| \nabla f(x) \|^2 \geq 2\mu (f(x) - f^*)$

(The difference $f(x) - f^*$ is the **optimality gap**)

### Proof of linear convergence

Finally, we can prove that gradient descent converge, under assumption 1 + 2 above and if we choose the **step size** $\eta$ to be a uniform small number.

Proof: By descent lemma and PL inequality,

$f(x_{n+1}) \leq f(x_n) - \frac{\eta}{2} \cdot 2\mu ( f(x_n) - f^* ) = (1 - \eta \mu) f(x_n) + (\eta\mu) f^*$

Subtracting $f^*$ on both sides give us

$f(x_{n+1}) - f^* \leq (1 - \eta\mu) (f(x_n) - f^*)$, hence

$f(x_n) - f^* \leq \gamma^n ( f(x_0) - f^* )$

That is, the optimality gap is $O ( \gamma^n )$, where $\gamma := 1 - \eta\mu \geq 1- \frac{\mu}{\beta}$.

(Note: $\beta \geq \mu \implies \frac{\mu}{\beta} \leq 1$)

## Stochastic Gradient Descent

### Setups

TODO
















