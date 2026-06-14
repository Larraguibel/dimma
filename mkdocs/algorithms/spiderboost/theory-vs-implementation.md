# Differences between theory and implementation

What is not fully congruent between Theorem 4.2 / Algorithm 2 of Arora et al. and the dimma implementation.

---

**1. `b1 = n` in the theory, but not in the implementation.**

Theorem B.3 sets `b1` (the anchor-step batch size) `= n` (the full dataset). This is what gives the paper its convergence rate. In the implementation `b1 = 4096`, which is a small fraction of `n = 800,000`. This changes the noise scale `σ₁` and weakens the privacy-utility guarantee relative to what Theorem 4.2 claims.

**2. `F0` is unknown.**

The optimal settings of `b2` (variation-step batch size), `T` (total number of iterations), and `q` (anchor frequency — an anchor step fires every `q` steps) all depend on `F0 = F(w0; S) − min_w F(w; S)`, the initial suboptimality. This is never observable in practice, so the theoretically derived parameter settings cannot be used directly. The implementation substitutes heuristic values, which is reasonable but means the implemented algorithm does not formally achieve the $O\left(\left(\frac{\sqrt{d}}{n \epsilon}\right)^{\frac{2}{3}}\right)$ rate of Theorem 4.2.

**3. `q` is preset, not derived.**

As discussed, the paper derives `q` from `n`, `ε`, `T`, `d`, `L1` (the gradient-Lipschitz constant) and `δ`. In the implementation it is a fixed hyperparameter. Whether the chosen value of 50 is in the right ballpark depends on the specific problem dimensions, and there is no guarantee it is.

**4. The model is not Lipschitz or smooth by assumption: it is enforced by clipping.**

The paper assumes the loss function is genuinely `L0`-Lipschitz (i.e., per-sample gradient norms are bounded by `L0`) and `L1`-smooth. In the implementation, gradient clipping is used to *enforce* these constants, which is the standard practical approach. However, clipping introduces bias — the clipped gradient is not an unbiased estimate of the true gradient — and the paper's analysis does not account for this bias. This is a known gap between DP-SGD theory and practice, not specific to this implementation.
