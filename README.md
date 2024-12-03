# sparse-bfgs
Sparse Hessian + BFGS = ?

Quasi-Newton methods aim to build $B^k \approx \nabla^2 f(x^k)$  or $H^k \approx \left(\nabla^2 f(x^k)\right)^{-1}$ according to some assumptions.

In BFGS method, the update rule is:

```math
B^{k+1} = B^{k} + a u u^{T} + b v v^{T}
```

Suppose that the Hessian matrix is sparse. Denote the sparsified matrix: $\tilde{B^k} = \text{sparse}(B^k)$. Then the update rule becomes:

```math
B^{k+1} = \tilde{B^{k}} + a u u^{T} + b v v^{T}
```

We determine $a, b, u, v$ by the secant equation

```math
\begin{align*}
\nabla f(x^{k+1}) - \nabla f(x^{k}) &= B^{k+1} (x^{k+1} - x^{k}) \\
y^k &= B^{k+1} s^k \\
y^k &= (\tilde{B^{k}} + a u u^{T} + b v v^{T}) s^k \\
(a u^{T} s^k) u + (b v^{T} s^k) v &= y^k - \tilde{B^{k}} s^k \\
\end{align*}
```

We can choose $a, b, u, v$ arbitrarily as long as the equation holds. For simplicity, we can choose:

```math
\left\{
\begin{align*}
a &= \frac{1}{u^T s^k} \\
u &= y^k \\
b &= -\frac{1}{v^T s^k} \\
v &= \tilde{B^{k}} s^k \\
\end{align*}
\right.
```


