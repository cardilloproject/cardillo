Introduction
++++++++++++

.. todo::

   Write an introduction.

Mechanics
---------

The used nomenclature inside this documentation is based on :cite:t:`Capobianco2021`.

**Kinematic variables**: Time $t$, gen. coordinates $\vq$, gen. velocitites $\vu$, gen. accelerations $\va$.

**Kinetic equation**

$$
\begin{aligned}
\mathbf{M}(t, \vq) \ \diff[\vu] &= \vh(t, \vq, \vu) \ \diff[t] + \vW_{\vg}(t, \vq) \ \diff[\vP_{\vg}] + \vW_{\vga}(t, \vq) \ \diff[\vP_{\vga}] \\
&\phantom{==} + \vW_{\mathrm{N}}(t, \vq) \ \diff[\vP_{\mathrm{N}}] + \vW_{\mathrm{F}}(t, \vq) \ \diff[\vP_{\mathrm{F}}]
\end{aligned}
$$

**Kinematic equation**

$$
\begin{aligned}
\diff[\vu] &= \mathbf{a} \ \diff[t] + \sum_i(\vu^+ - \vu^-) \ \diff[\delta_{t_i}] \\
\dot{\vq}(t, \vq, \vu) &= \mathbf{B}(t, \vq) \vu + \vbe(t, \vq)
\end{aligned}
$$

**Bilateral constraints**

$$
\begin{aligned}
\vg(t, \vq) &= \mathbf{0} \\
\vga(t, \vq, \vu) &= \mathbf{0}
\end{aligned}
$$

**Contact laws**: Normal cone inclusions linking gaps $\vg_{\mathrm{N}}(t, \vq)$ and friction velcities $\vga_{\mathrm{F}}(t, \vq, \vu)$ to percussion measures $\diff[\vP_{\mathrm{N}}]$ and $\diff[\vP_{\mathrm{F}}]$, respectively.
