# Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

**Authors:** Raman Arora, Raef Bassily, Tomás González, Cristóbal Guzmán, Michael Menart, Enayat Ullah
**Venue:** Proceedings of the 40th International Conference on Machine Learning (ICML), PMLR 202, 2023
**Source:** Uploaded PDF: `Faster Rates of Convergence to Stationary Points in Differentially Private Optimization(1).pdf`

> **Conversion note:** This revision normalizes the mathematical content into LaTeX syntax inside Markdown. Tables are retained as Markdown tables, and the PDF contains no raster plots requiring replacement.

## Abstract

We study the problem of approximating stationary points of Lipschitz and smooth functions under $(\varepsilon, \delta)$-differential privacy (DP) in both the finite-sum and stochastic settings. A point $\widehat{w}$ is called an $\alpha$-stationary point of a function $F: \mathbb{R}^d \to \mathbb{R}$ if $\|\nabla F(\widehat{w})\| \leq \alpha.$

We give a new construction that improves over the existing rates in the stochastic optimization setting, where the goal is to find approximate stationary points of the population risk given $n$ samples. Our construction finds a $\widetilde{O}\!\left( \frac{1}{n^{1/3}} + \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2} \right)$ stationary point of the population risk in time linear in $n$.

We also provide an efficient algorithm that finds an $\widetilde{O}\!\left( \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{2/3} \right)$ stationary point in the finite-sum setting. This improves on the previous best rate of $\widetilde{O}\!\left( \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2} \right).$

Furthermore, under the additional assumption of convexity, we completely characterize the sample complexity of finding stationary points of the population risk, up to polylogarithmic factors, and show that the optimal rate on population stationarity is $\widetilde{\Theta}\!\left( \frac{1}{\sqrt{n}} + \frac{\sqrt{d}}{n\varepsilon} \right).$

Finally, we show that our methods can be used to provide dimension-independent rates of

$$
O\!\left( \frac{1}{\sqrt{n}} + \min\!\left\{ \left(\frac{\sqrt{\operatorname{rank}}}{n\varepsilon}\right)^{2/3}, \frac{1}{(n\varepsilon)^{2/5}} \right\} \right)
$$

on population stationarity for Generalized Linear Models (GLM), where $\operatorname{rank}$ is the rank of the design matrix, which improves upon the previous best known rate.

*Equal contribution.*

## 1. Introduction
Protecting users' data in machine learning models has become a central concern in multiple contexts, e.g. those involving financial or health data. In this respect, differential privacy (DP) is the gold standard for rigorous privacy protection (Dwork & Roth, 2014). Therefore, recent research has focused on the limits and possibilities of solving some of the most well-established machine learning problems under the constraint of DP. Despite intensive research, some fundamental problems remain not completely understood. One example is nonconvex optimization; namely, the task of approximating stationary points, which has been heavily studied in recent years in the non-private setting (Fang et al., 2018; Ma et al., 2018; Carmon et al., 2017; Nesterov & Polyak, 2006; Ghadimi & Lan, 2013; Arjevani et al., 2019; Foster et al., 2019). This problem is motivated by the intractability of nonconvex (global) optimization, as well as by a number of settings where stationary points have been shown to be global minima (Ge et al., 2016; Sun et al., 2016).
### 1.1. Contributions

In this work, we make progress toward resolving the complexity of approximating stationary points in optimization under the constraint of differential privacy, for both empirical and population risks. In what follows, $d$ is the problem dimension, $n$ is the dataset size, and $(\varepsilon,\delta)$ are the approximate-DP parameters.

Our first set of results pertains to the task of approximating stationary points of the population risk. We provide the fastest rate up to date for this problem under DP, $\widetilde{O}\!\left( \frac{1}{n^{1/3}} + \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2} \right),$ with an algorithm that moreover has oracle complexity $n$, i.e. is single-pass. This algorithm is a noisy version of SPIDER whose gradient estimators are built using a tree-aggregation data structure for prefix sums.

For empirical nonconvex optimization, also called the finite-sum case, we provide algorithms with rate $O\!\left(\left(\frac{\sqrt{d}}{n\varepsilon}\right)^{2/3}\right),$ and oracle complexity

$$
\widetilde{O}\!\left( \max\left\{ \left(\frac{n^5\varepsilon^2}{d}\right)^{1/3}, \left(\frac{\sqrt{n}\varepsilon}{d}\right)^2 \right\} \right).
$$

This rate is sharper than the best known for this problem. For convex losses, we give an algorithm based on recursive regularization that achieves the optimal rate $\widetilde{\Theta}\!\left( \frac{1}{\sqrt{n}} + \frac{\sqrt{d}}{n\varepsilon} \right)$ on population stationarity. To establish optimality, we give a lower bound of $\Omega\!\left(\frac{\sqrt{d}}{n\varepsilon}\right)$ on empirical stationarity under DP, and a non-private lower bound of $\Omega\!\left(\frac{1}{\sqrt{n}}\right)$ on population stationarity.

Finally, for generalized linear models (GLMs), using Private SpiderBoost as the base algorithm yields a rate of

$$
\widetilde{O}\!\left( \frac{1}{\sqrt{n}} + \min\left\{ \left(\frac{\sqrt{\operatorname{rank}}}{n\varepsilon}\right)^{2/3}, \frac{1}{(n\varepsilon)^{2/5}} \right\} \right)
$$

on population stationarity.

| Setting | Convergence target | Our rate | Previous best-known rate |
|---|---:|---:|---:|
| Non-convex | Empirical | $\left(\frac{\sqrt{d}}{n\varepsilon}\right)^{2/3}$ (Thm. 4.2) | $\left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2}$ (Wang et al., 2017) |
| Non-convex | Population | $\frac{1}{n^{1/3}} + \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2}$ (Thm. 3.2) | $\sqrt{d\varepsilon} + \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2}$ (Zhou et al., 2020) |
| Convex | Population | $\frac{1}{\sqrt{n}} + \frac{\sqrt{d}}{n\varepsilon}$ (Thm. 5.1) | None |
| Non-convex GLM | Empirical | $\left(\frac{\sqrt{\operatorname{rank}}}{n\varepsilon}\right)^{2/3} \wedge \frac{1}{(n\varepsilon)^{2/5}}$ (Cor. 6.2) | $\left(\frac{\sqrt{\operatorname{rank}}}{n\varepsilon}\right)^{1/2}$ (Song et al., 2021) |
| Non-convex GLM | Population | $\frac{1}{\sqrt{n}} + \left(\frac{\sqrt{\operatorname{rank}}}{n\varepsilon}\right)^{2/3} \wedge \frac{1}{(n\varepsilon)^{2/5}}$ (Cor. 6.2) | None |
| Convex GLM | Population | $\frac{1}{\sqrt{n}} + \frac{\sqrt{\operatorname{rank}}}{n\varepsilon} \wedge \frac{1}{\sqrt{n\varepsilon}}$ (Cor. 6.2) | None |

*Table 1. Results summary. Logarithmic factors and function-class parameters are omitted. The symbol $\wedge$ denotes the minimum of the quantities.*

### 1.2. Our Techniques
Our methods combine multiple techniques from optimization and differential privacy in novel ways. The lower bound for the empirical norm of the gradient uses fingerprinting codes to a loss similar to that used for Differentially Private- Empirical Risk Minimization (DP-ERM) (Bassily et al., 2014), crafted to work in the unconstrained case. This lower bound can be extended to the population gradient norm by a known re-sampling argument (Bassily et al., 2019). We also give a non-private lower bound of \Omega (1/ \sqrt{n}) on population stationarity with n samples which holds even in dimension 1, as opposed to previous results (Foster et al., 2019). Efficient algorithms for (both empirical and population) norm of the gradient are derived using noisy versions of variance-reduced stochastic first order methods, which have proved remarkably useful in DP stochastic optimization (Asi et al., 2021; Bassily et al., 2021b;a). In the case of the empirical risk, we use a noisy version of SpiderBoost (Wang et al., 2019c). We remark that our methods can achieve comparable rates when applied to similar algorithms such as Spider (Fang et al., 2018) and Storm (Cutkosky & Orabona, 2019), but SpiderBoost allows for a larger learning rate which is We consider for complexity the first-order oracle model, standard for continuous optimization (Nemirovsky & Yudin, 1983). This is the rate obtained after fixing a mistake in the proof of **Theorem 4.1 in (Song et al., 2021). Specifically, in their proof, the** last term in Eq. (14) is missing a factor of T. considered better in practice. For the population risk, it is worth noting that the empirical norm of the gradient does not translate directly into population gradient guarantees, even if the algorithm in use is uniformly stable (Bousquet & Elisseeff, 2002), since this type of guarantee does not enjoy a stability-implies-generalization property. Therefore, we opt for single pass methods that combine variance-reduction with tree-aggregation; these techniques are particularly suitable for the classical Spider algorithm (Fang et al., 2018), which is the one we base our method on. For the convex setting, we use recursive regularization (Allen-Zhu, 2018) which was used to achieve the optimal non-private rate by (Foster et al., 2019). Finally, our method for (non-convex) GLMs uses the Johnson-Lindenstrauss based dimensionality reduction technique similar to (Arora et al., 2022), which focused on the convex setting. Moreover, for population stationarity of GLMs, we give a new uniform convergence result of gradients of Lipschitz functions. This guarantee, unlike the prior work of (Foster et al., 2018), has only \operatorname{poly}-logarithmic dependence on the radius of the constraint set, which is crucial for our analysis.
### 1.3. Related Work
The current work fits within the literature of differentially private optimization, which has primarily focused on the convex case (Chaudhuri et al., 2011; Jain et al., 2012; Kifer et al., 2012; Bassily et al., 2014; Talwar et al., 2014; Jain & Thakurta, 2014; Talwar et al., 2015; Bassily et al., 2019; Feldman et al., 2020; Asi et al., 2021; Bassily et al., 2021b). The culmination of this line of work for the convex smooth case showed that optimal rates are achievable in linear time (Feldman et al., 2020; Asi et al., 2021; Bassily et al., 2021b). Our work shows that in the convex case similar rates are achievable for the norm of the gradient: this result is useful, e.g., for dual formulations of linearly constrained convex programs (Nesterov, 2012), and moreover it has become a problem of independent interest (Allen-Zhu, 2018; Foster et al., 2019).3 Regarding stationary points for nonconvex losses, work in DP is far more recent, and primarily focused on the empirical stationarity (Wang et al., 2017; Zhang et al., 2017; To provide a specific example, consider the dual of the regularized discrete optimal transport problem, as discussed in (Diakonikolas & Guzmán, 2023), Section 5.6. If the marginals \mu, \nu in that model are accessed through i.i.d. samples, then this becomes an SCO problem. Moreover, it is argued in that reference that approximate stationary points provide approximately feasible and optimal transports through duality arguments. Hence, the result is an SCO problem where we require approximate stationary points.

| Setting | Convergence target | Our rate | Previous best-known rate |
|---|---:|---:|---:|
| Non-convex | Empirical | $\left(\frac{\sqrt{d}}{n\varepsilon}\right)^{2/3}$ (Thm. 4.2) | $\left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2}$ (Wang et al., 2017) |
| Non-convex | Population | $\frac{1}{n^{1/3}} + \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2}$ (Thm. 3.2) | $\sqrt{d\varepsilon} + \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2}$ (Zhou et al., 2020) |
| Convex | Population | $\frac{1}{\sqrt{n}} + \frac{\sqrt{d}}{n\varepsilon}$ (Thm. 5.1) | None |
| Non-convex GLM | Empirical | $\left(\frac{\sqrt{\operatorname{rank}}}{n\varepsilon}\right)^{2/3} \wedge \frac{1}{(n\varepsilon)^{2/5}}$ (Cor. 6.2) | $\left(\frac{\sqrt{\operatorname{rank}}}{n\varepsilon}\right)^{1/2}$ (Song et al., 2021) |
| Non-convex GLM | Population | $\frac{1}{\sqrt{n}} + \left(\frac{\sqrt{\operatorname{rank}}}{n\varepsilon}\right)^{2/3} \wedge \frac{1}{(n\varepsilon)^{2/5}}$ (Cor. 6.2) | None |
| Convex GLM | Population | $\frac{1}{\sqrt{n}} + \frac{\sqrt{\operatorname{rank}}}{n\varepsilon} \wedge \frac{1}{\sqrt{n\varepsilon}}$ (Cor. 6.2) | None |

*Table 1. Results summary: log factors and function-class parameters are omitted. The symbol $\wedge$ stands for the minimum of the quantities.* Wang & Xu, 2019; Wang et al., 2019a)4. Under similar assumptions to ours these works approximate stationary points with rate \widetilde{O} \sqrt{d} n\varepsilon 1/2, which is slower than ours. Works addressing population guarantees for the norm of the gradient under DP are scarce. (Zhou et al., 2020) proposed a noisy gradient method, whose population guarantee is obtained by generalization properties of DP. However, the best guarantee obtainable with their analysis is O \sqrt{d} n\varepsilon 1/2 + \sqrt{d}$\varepsilon$. Note that for any $\varepsilon$ this rate is \Omega [d/n]1/3

. Under additional assumptions (on the Hessian), (Wang & Xu, 2019) obtains a rate of \widetilde{O}( p d/(n\varepsilon)) by uniform convergence of gradients, which is sharper when $\varepsilon$ is constant. By contrast, our rate is much faster than both for $\varepsilon$ = \Theta(1). In particular, in this range, our rates are faster than those obtained by uniform convergence, O( p d/n) (Foster et al., 2018). Moreover, our method runs in time linear in n. On the other hand, in the much more restrictive setting where the loss satisfies the Polyak-Łojasiewicz (PL) inequality, (Zhang et al., 2021) provide population risk $bounds of \widetilde{O}(d/[n\varepsilon]2$ ) under DP. The work of (Bassily et al., 2021a) studies population guarantees for stationarity in constrained settings, obtaining rates O 1 n^{1/3} + \sqrt{d} n\varepsilon 2/5 in linear time. Notice first that these guarantees are based on the Frank-Wolfe gap, making those results incomparable to ours. Despite this fact, Another work, (Wang et al., 2019b), claims to achieve this with improved oracle complexity. However, the analysis therein contains an error which is not easily fixed. Specifically, (Wang et al., 2019b, proof of Theorem 4.1) uses $\sigma$^2 0b2 0 > 0.7 to employ privacy amplification via subsampling. This is not true as they set $\sigma0 = 1/[d^{1/4}\sqrt$ n] and b0 = \sqrt{n}/d^{1/4}. (Zhou et al., 2020) omits the term \sqrt{d}$\varepsilon$, but this omission is only valid when $\varepsilon$ < 1/[n \sqrt{d}]1/3. their rates are slower than ours.6 On the other hand, they provide results for (close to nearly) stationary points in constrained/unconstrained settings, for a broader class of weakly convex losses (possibly nonsmooth). This result is then more general, but the rate of O 1 n^{1/4} + \sqrt{d} n\varepsilon 1/3 is substantially slower than ours, and their algorithm has oracle complexity which is superlinear in n. The problem of stationary points in (nonprivate) stochastic optimization has drawn major attention recently (Ghadimi & Lan, 2013; 2016; Fang et al., 2018; Allen-Zhu, 2018; Foster et al., 2018; 2019; Arjevani et al., 2019). To the best of our knowledge, no lower bounds for the sample complexity7 of this problem are known (beyond those known for the convex case (Foster et al., 2019)). On the other hand, oracle complexity is by now understood: in high dimensions, for (on average) smooth losses the optimal stochastic oracle complexity rate is O(1/n^{1/3} ) (Arjevani et al., 2019). Although this provides some evidence of the sharpness of our results (see Appendix B.2), note that these lower bounds require very high dimensional constructions (namely, d = \Omega(1/\alpha4 ), where \alpha is the rate), which limits their applicability in the private setting. In an independent and concurrent work, (Tran & Cutkosky, 2022) achieve a rate of O( \sqrt{d} n\varepsilon 2/3
+ 1
\sqrt{n} $) on the empirical gradient with gradient complexity O(n^{7/3}$ $\varepsilon$3/4 /d^{2/3} ) using a DP tree aggregation method. Note that our result removes the 1/ \sqrt{n} term and improves the oracle complexity to \widetilde{O} max n^5 \varepsilon^2 d 1/3, n\varepsilon \sqrt{d}, which is better whenever We believe our methods can be extended to constrained settings using gradient mapping, a guarantee for which is stronger than for Frank-Wolfe gap (Lan, 2020, Section 7.5.1). We defer this extension to future work. Sample complexity is the fundamental limit on the sample size needed, as a function of \alpha, to achieve \alpha stationarity. This is different from the oracle complexity as one is not limited to first-order methods.

d \leq n^2 $\varepsilon$1/4 (i.e. essentially whenever the error is nontrivial). Further, we accomplish this with a much simpler analysis.
## 2. Preliminaries

Let $f: \mathbb{R}^d \times \mathcal{X} \to \mathbb{R}$ denote a loss function taking as input the model parameter $w$ and data point $x \in \mathcal{X}$. We assume that $w \mapsto f(w;x)$ is $L_0$-Lipschitz and $L_1$-smooth. That is, for all $x \in \mathcal{X}$ and $w_1,w_2 \in \mathbb{R}^d$,

|f(w_1;x)-f(w_2;x)| \leq L_0\|w_1-w_2\|,

and $\|\nabla f(w_1;x)-\nabla f(w_2;x)\| \leq L_1\|w_1-w_2\|.$

Given a dataset $S=(x_1,\ldots,x_n) \in \mathcal{X}^n$, the empirical risk is

$$
F(w;S)=\frac{1}{n}\sum_{i=1}^n f(w;x_i).
$$

Assuming that the data points are sampled i.i.d. from an unknown distribution $\mathcal{D}$, the population risk is $F(w;\mathcal{D})=\mathbb{E}_{x\sim\mathcal{D}} f(w;x).$

For the empirical case, define

$$
F_0 = F(0;S)-\min_{w\in\mathbb{R}^d}\{F(w;S)\},
$$

and similarly for the population loss. We use $w^\ast$ to denote the population-risk minimizer. We write $I_d$ for the $d\times d$ identity matrix and $[a]=\{1,2,\ldots,a\}$ for $a\geq 1$.

**Stationary points.** Given a dataset $S$, the goal is to find an $\alpha$-stationary point $\bar{w}$ of either the empirical or population risk:

$$
\|\nabla F(\bar{w};S)\|\leq \alpha \qquad\text{or}\qquad \|\nabla F(\bar{w};\mathcal{D})\|\leq \alpha.
$$

**Differential privacy.** An algorithm $\mathcal{A}$ is $(\varepsilon,\delta)$-differentially private if, for all datasets $S$ and $S'$ differing in one data point and all events $E$ in the range of $\mathcal{A}$, $\mathbb{P}(\mathcal{A}(S)\in E) \leq e^\varepsilon \mathbb{P}(\mathcal{A}(S')\in E)+\delta.$

**Generalized linear models.** For data domains $\mathcal{X}\subseteq\mathbb{R}^d$ and $\mathcal{Y}\subseteq\mathbb{R}$, a loss function $f:\mathbb{R}^d\times\mathcal{X}\times\mathcal{Y}\to\mathbb{R}$ is a GLM if $f(w;(x,y))=\phi_y(\langle w,x\rangle)$ for some function $\phi_y$.

**Definition 2.1 (($\gamma,\beta$)-JL property).** A random matrix $\Phi\in\mathbb{R}^{k\times d}$ satisfies the $(\gamma,\beta)$-JL property if, for any $u,v\in\mathbb{R}^d$, $\mathbb{P}\!\left[ \left|\langle \Phi u,\Phi v\rangle-\langle u,v\rangle\right| >\gamma\|u\|\|v\| \right]\leq \beta.$

## 3. Stationary Points of Population Risk
For the population gradient, we provide a linear time algorithm; see Algorithm 1 for pseudocode. It is a noisy variant of SPIDER (Fang et al., 2018), and utilizes a variance reduction technique tailored to an underlying binary tree structure. Namely, we run T rounds, where at the beginning of round t we build a binary tree of depth D, $whose nodes are denoted by u_t,s, where s \in {0, 1}D$. Every node u_t,s is associated with a parameter vector w_t,s and a gradient estimate \nabla t,s. Next, we perform a Depth-First- Search traversal of the tree. We denote by DFS[D] the set of nodes in the visiting order excluding the root, for example: DFS[2] = {u0, u00, u01, u1, u10, u11}. When a left child node is visited, it receives the same parameter vector and gradient estimator of the parent node.
### Algorithm 1. Tree-based Private SPIDER

**Input:** Private dataset $S=(x_1,\ldots,x_n)\in\mathcal{X}^n$; privacy parameters $(\varepsilon,\delta)$; number of rounds $T$; batch size $b$ at the beginning of each round; tree depth $D$; step-size parameter $\beta$; accuracy parameter $\widetilde{\alpha}$.

1. Set $w_{0,\ell(2^D-1)}=0$.
2. For $t=1,\ldots,T$:
   1. Set $w_{t,\emptyset}=w_{t-1,\ell(2^D-1)}$.
   2. Draw a batch $S_{t,\emptyset}$ of $b$ data points, and set $S\leftarrow S\setminus S_{t,\emptyset}$.
   3. Set

      $$
      \sigma_{t,\emptyset}^2= \frac{8L_0^2\log(1.25/\delta)}{b^2\varepsilon^2}.
      $$

   4. Set

      $$
      \nabla_{t,\emptyset} =\frac{1}{b}\sum_{x\in S_{t,\emptyset}}\nabla f(w_{t,\emptyset};x)+g_{t,\emptyset}, \qquad g_{t,\emptyset}\sim\mathcal{N}\!\left(0,I_d\sigma_{t,\emptyset}^2\right).
      $$

   5. For $u_{t,s}\in DFS[D]$:
      1. Let $s=\widehat{s}c$, where $c\in\{0,1\}$.
      2. If $c=0$, set

         $$
         \nabla_{t,s}=\nabla_{t,\widehat{s}}, \qquad w_{t,s}=w_{t,\widehat{s}}.
         $$

      3. Otherwise, draw a batch $S_{t,s}$ of $b/2^{|s|}$ data points and set $S\leftarrow S\setminus S_{t,s}$; then set

         $$
         \sigma_{t,s}^2= \frac{8\cdot 2^D\beta^2\log(1.25/\delta)}{b^2\varepsilon^2},
         
         \Delta_{t,s} =\frac{2^{|s|}}{b}\sum_{x\in S_{t,s}} \left(\nabla f(w_{t,s};x)-\nabla f(w_{t,\widehat{s}};x)\right) +g_{t,s}, \qquad g_{t,s}\sim\mathcal{N}\!\left(0,I_d\sigma_{t,s}^2\right),
         $$

and

         $$
         \nabla_{t,s}=\nabla_{t,\widehat{s}}+\Delta_{t,s}.
         $$

      4. If $|s|=D$, i.e. $u_{t,s}$ is a leaf:
         1. If $\|\nabla_{t,s}\|\leq 2\widetilde{\alpha}$, return $w_{t,s}$.
         2. Let $u_{t,s^+}$ be the next vertex in $DFS[D]$.
         3. Set

            $$
            \eta_{t,s}= \frac{\beta}{2^{D/2}L_1\|\nabla_{t,s}\|}.
            $$

         4. Set

            $$
            w_{t,s^+}=w_{t,s}-\eta_{t,s}\nabla_{t,s}.
            $$

3. Return $w$ chosen uniformly at random from $\{w_{t,s}:t\in[T],\ u_{t,s}\text{ is a leaf}\}$.

On the other hand, when a right child node is visited, it receives a fresh set of samples and uses it to update the gradient estimator coming from the parent node. Every time a leaf node is reached, a gradient step is performed using the gradient estimator associated to the leaf. Finally, the parameter vector of a right child node comes from the gradient step performed at the right-most leaf in the left sub-

tree of it. The use of the binary tree structure is beneficial because every gradient estimator is updated at most D times within a round of 2D optimization steps, as opposed to the original SPIDER algorithm where the gradient estimators are updated at every optimization step. This way, we are able to perform the same number of optimization steps but adding substantially smaller amounts of noise, leading to a faster rate than the one we would get without using the tree. In the following, we denote by \ell(k) the binary representation of any number k \in [0, 2D
- 1] and by |s| the depth of u_t,s for
any t \in [T]. The proposed algorithm is similar to the one in Section 5 of (Bassily et al., 2021b) for constrained Differentially Private-Stochastic Convex Optimization (DP-SCO), with the key difference that Algorithm 1 executes each round with fixed depth trees, which is key for our convergence analysis, whereas the prior work leverages convexity to construct trees that increase depth by one at each round. In addition, to choose the step-size in (Bassily et al., 2021b) the authors leverage the bounded diameter of the domain, while our step-size is chosen as that of (Fang et al., 2018), i.e. normalized by the norm of the gradient estimator and proportional to the target accuracy. This choice is crucial for controlling the sensitivity of the gradient variation estimator in the unconstrained setting, and consequently for the privacy analysis as well. Our results are presented below and the proofs are deferred to Appendix C. **Theorem 3.1 (Privacy guarantee).** For any $\varepsilon,\delta\in[0,1]$, Algorithm 1 is $(\varepsilon,\delta)$-DP.

**Theorem 3.2 (Accuracy guarantee).** Let $p\in(0,1)$ and $\varepsilon,\delta>0$. Set

$$
b=\max\left\{n^{2/3},\frac{\sqrt{n}\,d^{1/4}}{\sqrt{\varepsilon}}\right\}, \qquad D2^{D+1}=b, \qquad T=\frac{n}{b(D/2+1)},

\alpha=\sqrt{2}L_0\max\left\{ \frac{1}{n^{1/3}}, \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2} \right\},

\beta=\alpha\min\left\{1,\frac{\sqrt{b\varepsilon}}{\sqrt{d}}\right\}, \qquad \widetilde{\alpha}=\widetilde{C}\alpha,
$$

where

$$
\widetilde{C} =256\log\!\left(\frac{1.25}{\delta}\right) \log\!\left(\frac{2T2^{D+1}}{p}\right) + \frac{8L_1F_0\sqrt{2D(D/2+1)}}{2L_0^2}.
$$

Then, for any

$$
n\geq \max\left\{ \frac{\sqrt{d}(D/2+1)^2}{\varepsilon}, (D/2+1)^3 \right\},
$$

with probability at least $1-p$, Algorithm 1 ends in line 20 and returns an iterate $w_{t,s}$ satisfying

$$
\|\nabla F(w_{t,s};\mathcal{D})\| \leq 3\sqrt{2}L_0\widetilde{C} \max\left\{ \frac{1}{n^{1/3}}, \left(\frac{\sqrt{d}}{n\varepsilon}\right)^{1/2} \right\}.
$$

Furthermore, Algorithm 1 has oracle complexity $n$.

## 4. Stationary Points of Empirical Risk
### 4.1. Efficient Algorithm with Faster Rate
The algorithm for our upper bound is a noisy version of the SpiderBoost algorithm (Wang et al., 2019c). The algorithm SpiderBoost itself is essentially the Spider algorithm (Fang et al., 2018) with a different learning rate and analysis. It works by running a series of phases of length q. Each phase starts with a minibatch estimate of the gradient, and subsequent gradient estimates within the phase are then computed by adding an estimate of the gradient variation. The key to the analysis is to bound the error in the gradient estimate at each iteration. Towards this end, we have the following generalization of the (Wang et al., 2019c) Lemma 1, which follows directly from (Fang et al., 2018) Proposition 1. **Lemma 4.1. Consider Algorithm 2, and for any t \in** {0,.., T} let st = j t q k q. If each \nabla t computed in line 9 is an unbiased estimate of \nabla F(w_t; S) satisfying E h $\|\nabla st - \nabla F(wst; S)\|$ i \leq $\tau$^2 1 and each \Delta_t computed in line 13 is an unbiased estimate of the gradient variation satisfying E h $\|\Delta_t - [\nabla F(w_t; S) - \nabla F(w_t-1; S)]\|$ i \leq $\tau$^2 $2 \|w_t - w_t-1\|$. Then for any t \geq st + 1, the iterates of Algorithm 2 satisfy E

$\|\nabla t - \nabla F(w_t)\|2$ \leq $\tau$^2 t X k=st+1 E

$\|w_k - w_k-1\|2$
+ $\tau$^2
1. For privacy, using smoothness we observe the sensitivity of the gradient variation estimate at iteration t is proportional to \beta \|w_t - w_t-1\|. Thus we can apply the above lemma with $\tau$^2 1 = L^2 b_1
+ L^2
0$\sigma$^2 1 and $\tau$^2 2 = L^2 b_2
+ L^2
1$\sigma$^2 2 (note the Gaussian noise in line 13 is drawn with variance scale at most $\sigma$^2 $2 \|w_t - w_t-1\|$ ). By carefully balancing the algorithm parameters, we are then able to obtain the following result. The full proof is deferred to Appendix B.1. **Theorem 4.2 (Private Spiderboost ERM). Let \varepsilon, \delta \in [0, 1].** Let n \geq max

(L_0$\varepsilon$)2 F0L1d log(1/$\delta$), \sqrt{d} max{1, \sqrt L1F0/L_0} $\varepsilon$

$. Algorithm 2 is (\varepsilon, \delta)-DP. Further, there exist settings of$ T, \eta, q, b_1, b_2 such that Algorithm 2 has E [\|\nabla F(\bar{w}; S)\|] bounded as O \left(

\sqrt F0L1L0 p d log (1/$\delta$) n\varepsilon !2/3 + L_0 p d log (1/$\delta$) n\varepsilon \right)

and oracle complexity \widetilde{O}

max

n^{5/3} \varepsilon^2/3 d^{1/3}

,

n\varepsilon \sqrt{d} 2

. Note that the restriction on n in the theorem statement is essentially trivial when the upper bound is nontrivial. We remark that the case where the dominant error term is \alpha = \widetilde{O} \sqrt{d} n\varepsilon 2/3

, then we approximately have oracle complexity \widetilde{O} max 1 \alpha3, n \alpha

.
### 4.2. Lower Bound
We now show a lower bound for the sample complexity of finding a stationary point under differential privacy in the unconstrained setting, which shows that the O L_0 \sqrt{d} log(1/$\delta$) n\varepsilon

### Algorithm 2. Private SpiderBoost
Input: Dataset: S \in $\mathcal{X}^n$
, Function: f: \mathbb{R}^d $\times X \to \mathbb{R},$
Learning Rate: \eta, Phase Size: q, Batch Sizes b_1, b_2,
$Privacy Parameters: (\varepsilon, \delta), Iterations: T$
1: w_0 = 0
2: $\sigma$1 =
cL0 \sqrt log(1/$\delta$) $\varepsilon$ max n b_1, \sqrt T \sqrt qn o, where c is a universal constant.
3: $\sigma$^2 =
cL1 \sqrt log(1/$\delta$) $\varepsilon$ max n b_2, \sqrt T n o
4: b
$\sigma$^2 = 2cL0 \sqrt log(1/$\delta$) $\varepsilon$ max n b_2, \sqrt T n o
5: for t = 0, . . . , T do
6: if mod (t, q) = 0 then
7: Sample batch S_t of size b_1
$8: Sample g_t \sim \mathcal{N}(0, I_d\sigma^2$ 1)
9: \nabla t = 1
b_1 P x\inSt $\nabla f(w_t; x) + g_t$
10: else
11: Sample batch S_t of size b_2
12: g_t \sim N

0, I_d min n $\sigma$^2 $2 \|w_t - w_t-1\|$, b $\sigma$^2 o
13: \Delta_t = 1
b_2 P x\inSt $[\nabla f(w_t; x) - \nabla f(w_t-1; x)]+g_t$
14: \nabla t = \nabla t-1 + \Delta_t
15: end if
$16: w_t+1 = w_t - \eta\nabla t$
17: end for
18: return \bar{w} uniformly at random from {w_1, . . . , wT }
term in the rate given in Theorem 4.2 is necessary. Furthermore, as our lower bound holds for all levels of smoothness, it also shows that our rate in Theorem 4.2 is optimal in the (admittedly uncommon) regime where L_1 \leq \sqrt{d}L2 F0n\varepsilon. Our lower bound in fact holds even for convex functions. Furthermore, this result implies the same lower bound (up to log factors) for the population gradient using the technique in (Bassily et al., 2019), Appendix C. **Theorem 4.3. Given L_0, L_1, n, \varepsilon = O(1), 2-\Omega(n)** $\leq \delta \leq 1/n1+\Omega(1)$, there exists an L_0-Lipschitz, L_1-smooth (convex) loss f: \mathbb{R}^d \times X \to \mathbb{R} and a dataset S of n points such that any $($\varepsilon$, $\delta$)$-DP algorithm run on S with output \bar{w} satisfies, \|\nabla F(\bar{w}; S)\| = \Omega L_0 min 1, p d log (1/$\delta$) n\varepsilon !!. The proof is based on a reduction to DP mean estimation. Specifically, we consider a instance of the Huber loss function for which the minimizer is the empirical mean of the dataset. We then argue that close to the minimizer, the empirical stationarity is lower bounded by DP mean estimation bound (Steinke & Ullman, 2015), and far away, by construction, the empirical stationarity is L_0. Proof of Theorem 4.3. For any r > 0, let Wr denote the ball of radius r centered at the origin. Let B = L_0 L_1. Consider the loss function: f(w; x) = ( L_1 2 \|w - x\| $if \|w - x\| \leq B$ L_0 \|w - x\| - L^2 2L1 otherwise The function f(w; x) is convex, L_1-smooth and L_0-
$$
Lipschitz in \mathbb{R}^d. We restrict to datasets S = {x_i} n i=1 where x_i \in WB/4 for all i, and let F(w; S) = 1 n Pn i=1 f(w; x_i) be the empirical risk on S. The unconstrained minimizer of F(w; S) is w^\ast = 1 n Pn i=1 x_i which lies in WB/4. For any w \in W3B/4, w lies in the quadratic region around all data points. Hence, from L_1-strong convexity of w 7\to F(w; S) on W3B/4, we have that whenever \bar{w} \in W3B/4,
$$
\|\nabla F(\bar{w}; S)\| \|\bar{w} - w^\ast $\| \geq \langle \nabla F(\bar{w}; S), w^\ast$
- \bar{w} \rangle
$\geq F(\bar{w}; S) - F(w^\ast$; S) \geq L_1 $\|\bar{w} - w^\ast$ \|. Let E be the event that \bar{w} \in W3B/4 and let \mathbb{E}_E denote the conditional expectation (conditioned on event E) operator. Then, $\mathbb{E}_E\|\nabla F(\bar{w}; S)\| \geq$ L_1 $E \|\bar{w} - w^\ast$ \| \geq L_1 \Omega

L_0 4L1

min 1, p d log (1/$\delta$) n\varepsilon !!. where the last inequality follows from known lower bounds for DP mean estimation (Steinke & Ullman, 2015; Kamath & Ullman, 2020). We remark that the lower bound in the referenced work is for algorithms which produce outputs in the ball of the same radius as the dataset, i.e. WB/4. However, a simple post-processing argument shows that the same lower bound applies to algorithms which produce output in W3B/4. Specifically, assuming the contrary, we simply project the output in W3B/4 to WB/4: privacy is preserved by post-processing and the distance to the mean cannot increase by the non-expansiveness property of projection to convex sets, hence a contradiction. This gives us, $\mathbb{E}_E [\|\nabla F(\bar{w}; S)\|] \geq \Omega L_0 min 1,$ p d log (1/$\delta$) n\varepsilon !! Let W̃ = {w: \|w - w^\ast $\| \leq B/2}. Since W̃ \subseteq W3B/4,$ we have that the above conditional lower bound applies for \bar{w} \in W̃ as well. We now consider \bar{w} ̸\in W̃. Let w' be any point on the boundary of W̃, denoted as \partialW. Note that w' lies in the region where, for any data point, the

corresponding loss is a quadratic function. Hence, by direct computation, \nabla F(w'; S) = L_1 (w'
- w^\ast
). Therefore, \langle \nabla F(w' ), w'
- w^\ast
\rangle = L_1 \|w'
- w^\ast
\| = L1B2. We now apply gradient monotonicity to obtain the following (see Lemma A.1, Appendix A), $EEc \|\nabla F(\bar{w}; S)\| \geq$ L1B2 \cdot B = L_0, where E^c denotes the complement set of E. We combine the above bounds using the law of total expectation as follows, $\mathbb{E}[\|\nabla F(\bar{w}; S)\|]$ = \mathbb{E}_E[\|\nabla F(\bar{w}; S)\|]\mathbb{P}\{\bar{w} \in E} + EEc [\|\nabla F(\bar{w}; S)\|]\mathbb{P}\{\bar{w} \in E^c } = \Omega

L_0 min n 1, p d log (1/$\delta$) n\varepsilon o $P(\bar{w} \in E) + \Omega(L_0)P(\bar{w} \in E^c$ ) = \Omega

L_0 min n 1, p d log (1/$\delta$) n\varepsilon o. This completes the proof. Challenges for Further Rate Improvements: Given the above lower bound, the question arises as to whether the \widetilde{O} \sqrt{d} n\varepsilon ]2/3

term can be improved. An informal argument using the oracle complexity lower bound of (Arjevani et al., 2019) suggests several major challenges in obtaining further rate improvements. A more detailed version of the following discussion can be found in Appendix B.2. Consider methods which ensure privacy by directly privatizing the gradient/gradient variation queries. The aim of such methods is to design some private stochastic first order oracle, O$\varepsilon$',$\delta$', such that a set of G queries to O$\varepsilon$',$\delta$' satisfies $($\varepsilon$, $\delta$)$-DP, and use this oracle in some optimization algorithm A(O$\varepsilon$',$\delta$' ). Such a setup encapsulates numerous results in the convex setting (Bassily et al., 2019; Kulkarni et al., 2021), and is even more dominant in nonconvex settings (Wang et al., 2017; Zhou et al., 2020; Abadi et al., 2016). Under advanced composition based arguments, to make G calls to such a private oracle one needs $\varepsilon$' \leq $\varepsilon$/ \sqrt Now, standard fingerprinting code arguments suggest lower bounds on the level of accuracy of any such private oracle (Steinke & Ullman, 2015). Specifically, without leveraging further problem structure beyond Lipschitzness, one needs the gradient estimation error to be at least $\tau$1 = \Omega

L_0 \sqrt Gd log(1/$\delta$) n\varepsilon

. A similar argument suggests the error in the gradient variation between iterates w, w' must at least $\tau$^2 \|w - w' \| = \Omega L_1\|w-w' \| \sqrt Gd log(1/$\delta$) n\varepsilon

. Now consider some optimization algorithm, A, which takes as input a stochastic oracle O for some smooth function L. The lower bound of (Arjevani et al., 2019) suggests that if A makes at most G queries to O (as a black box) the algorithm $satisfies E [\|\nabla L(A(O))\|] = \Omega$

F_0$\tau$^2$\tau$1 G 1/3
+ $\tau$1
\sqrt G

. If O is a private oracle satisfying the previously mentioned conditions, we would then have under the setting of $\tau$1 and $\tau$^2 suggested by privacy that the convergence guarantee for $E [\|\nabla L(A(O))\|] is lower bounded as$ \Omega \left(

\sqrt F0L1L0 p d log (1/$\delta$) n\varepsilon !2/3 + L_0 p d log (1/$\delta$) n\varepsilon \right). This indicates a substantial challenge for future rate improvements, as alternative methods which avoid private gradients (see e.g. (Feldman et al., 2020)) rely crucially on stability guarantees arising from convexity.
## 5. Stationary Points in the Convex Setting
### Algorithm 3. Recursive Regularization
Input: Dataset S, loss function f, steps T, {\lambda_t}t, {R_t}t,
PrivateSubRoutine, number of steps of sub-routine
${K_t}, selector functions {S_t(\cdot)}t, step size {\etat}t, noise$ variances {\sigma_t}t
1: w_0 = 0, n_0 = 1
2: Define function (w, x) \to f(0)
(w; x) = f(w; x) + $\lambda$0 $2 \|w - w_0\|$
3: for t = 1 to T - 1 do
4: n_t = n_t-1 +
j
|S|
T k $5: \bar{w}t = PrivateSubRoutine(S_{n_t}-1:n_t$, f(t-1), R_t, $K_t, \etat, S_t(\cdot), \sigma_t)$
6: Define function (w, x) \to f(t)
(w; x) = f(t-1) (w; x) + \lambda_t $2 \|w - \bar{w}t\|$
7: end for
Output: \bar{w} = \bar{w}T
In this section, we additionally assume that the loss function is convex. The motivation for this is two-fold: firstly, this setting has recently gained attention in a non-private setting (Nesterov, 2012; Allen-Zhu, 2018; Foster et al., 2019). Secondly, in this setting we are able to establish tightly the sample complexity of approximate stationary points. Our method is based on the recursive regularization technique proposed in (Allen-Zhu, 2018), and further improved by (Foster et al., 2019). The main idea, as the name suggests, is to recursively regularize the objective and optimize it via some solver. For the DP setting, the key idea is to use a private sub-routine as the inner solver. Furthermore, while a solver for the unconstrained problem suffices non-privately, we need to carefully increase the radius of the constrained set over which the solver operates. **Theorem 5.1. Let L_0, L_1, $\varepsilon$, $\delta$ > 0, d, n \in N. Let w 7\to** $f(w; x) be an L_0-Lipschitz L_1-smooth convex function for$

all x. Let R_t = \sqrt t \|w^\ast \|, \lambda_t = 2t $\lambda, \etat = log(K_t)$ \lambdatKt, T =

log2 L_1 $\lambda$

, $\sigma$^2 t = 64L2 0K2 t log(1/$\delta$) $n^2\varepsilon^2, and S_t({w_k}k) =$ PKt $k=1(1-\etat\lambda_t)-k$ PKt k=1 (1 - \etat\lambda_t) -k w_k.
1. (Optimal rate) Algorithm 3 run with NoisyGD
(Algorithm 7 in Appendix D) as the PrivateSubRoutine with above parameter settings and $\lambda$ = L^2 $L_1\|w^\ast\| min 1$ n, d n^2\varepsilon^2

and K_t = max

L_1+\lambda_t \lambda_t log

L_1+\lambda_t \lambda_t

, n^2 \varepsilon^2

L^2 0$\lambda$+L 3/2

T 2\lambdadL2 0 log(1/$\delta$)

$satisfies (\varepsilon, \delta)-DP, and given a dataset S of n i.i.d.$ samples from D, outputs \bar{w} such that $E \|\nabla F(\bar{w}; D)\| = \widetilde{O}$ L_0 \sqrt{n} + L_0 \sqrt{d} n\varepsilon !. Furthermore, the above rate is tight up to polylogarithmic factors.
2. (Linear time rate) Algorithm 3 run with
PhasedSGD (Algorithm 5) as the PrivateSubRoutine with with above parameter settings and $\lambda$ = max

L^2 $L_1\|w^\ast\|2 min 1$ n, d n^2\varepsilon^2

, L_1 log(n) n

and K_t = \lfloor n T \rfloor satisfies $($\varepsilon$, $\delta$)$-DP and given a dataset S of n i.i.d. samples from D, in linear time, outputs \bar{w} with $E \|\nabla F(\bar{w}; D)\| = \widetilde{O}$ L_0 \sqrt{n} + L_0 \sqrt{d} n\varepsilon + L_1 \|w^\ast \| \sqrt{n} !. The proof of the above result is deferred to Appendix D. For the tightness of the rate, the necessity of the second term L_0 \sqrt{d} n\varepsilon is due to our DP empirical stationarity lower bound, Theorem 4.3. For the first “non-private” term L_0 \sqrt{n}, even though (Foster et al., 2019) proved a sample complexity lower bound, their instance is not Lipschitz and has $d = \Omega (n log (n)), hence not applicable. To remedy this, we$ give a new lower bound construction with a Lipschitz function in d = 1, Theorem A.2 in Appendix A. The polylog dependence on L_1 and \|w^\ast \| in the upper bounds, is consistent with the non-private sample complexity in (Foster et al., 2019). The second result is a linear time method which has an additional L_1 \|w^\ast \| / \sqrt{n} term. Firstly, if the smoothness parameter is small enough, then there is no overhead; this small-enough smoothness is precisely the regime in which we have linear time methods with optimal rates for smooth DP-SCO (Feldman et al., 2020). More importantly, (Foster et al., 2019) showed that even in the non-private setting, a polynomial dependence on L_1 \|w^\ast \| is necessary in the stochastic oracle model. However, the optimal nonprivate term, shown in (Foster et al., 2019), is L_1 \|w^\ast \| /n^2, achieved by accelerated methods. Improving this dependency, if possible, is an interesting direction for future work.
## 6. Generalized Linear Models
In this section, we assume that the loss function is a generalized linear model (GLM), f(w; (x, y)) = \phi_y (\langle w, x \rangle). Also, assume the norm of data points x are bounded by \|X\| and $the function \phi_y: \mathbb{R} \to \mathbb{R} is L_0-Lipschitz and L_1-smooth$ for all y. Furthermore, let \operatorname{rank} denote the \operatorname{rank} of design $matrix X \in \mathbb{R}^n\timesd$.
### Algorithm 4. JL method
Input: Dataset S, function (z, y) \to \phi_y(z), Algorithm A,
$JL matrix \Phi \in \mathbb{R}^k\timesd$, L_0, L_1, \|X\| $1: w̃ = A((z, y) \to \phi_y(z), {(\Phi x_i, y_i)}$ n i=1, $2L0 \|X\|, 2L1 \|X\|$, \varepsilon, \delta/2)
Output: \bar{w} = \Phi^\top
w̃ Algorithm 4 is a generic method which converts any for smooth Lipschitz losses with an empirical stationarity guarantee to get dimension-independent rates on population stationarity for smooth Lipschitz GLMs. This algorithm is the JL method from (Arora et al., 2022) used therein to give excess risk bounds for convex GLM. We note that while the JL method there is limited to the Noisy GD method, ours is a black-box reduction. Furthermore, unlike (Arora et al., 2022), we show that the JL method gives finer \operatorname{rank} based guarantees by leveraging the fact it acts as an oblivious approximate subspace embedding (see Definition E.1 in Appendix E). **Theorem 6.1. Let A be an $($\varepsilon$, $\delta$)$-DP algorithm which when** run on a L_1-smooth L_0-Lipschitz function on a dataset $S = {(x_i, y_i)}$ n $i=1 where x_i \in X \subseteq \mathbb{R}^d$, guarantees E [\|\nabla F(A(S); S)\|] \leq g(d, n, L_1, L_0, $\varepsilon$, $\delta$) and \|A(S)\| \leq $\operatorname{poly}(n, d, L_0, L_1) with probability at least 1 - 1$ \sqrt{n}. Then,
### Algorithm 4. run with
k =

min

arg min j\inN

$g(j, n, 2L0 \|X\|, 2L1 \|X\|$, \varepsilon, \delta/2) + $L_0 \|X\| log (n)$ \sqrt j

, \operatorname{rank} log

2n $\delta$

on a L_0-Lipschitz, L_1-smooth GLM loss, is $($\varepsilon$, $\delta$)$-DP. Furthermore, given a dataset of n i.i.d samples from D, its $output \bar{w} has E [\|\nabla F(\bar{w}; D)\|] bounded as$ \widetilde{O}

L_0 \|X\| \sqrt{n} $+ g(k, n, 2L0 \|X\|, 2L1 \|X\|$, \varepsilon, \delta/2)

The expression for k above comes from the subspace embedding property of JL, and from balancing the dimension of the embedding with respect to the error of A and the approximation error of the JL embedding. The proof is based on the properties of JL matrices: oblivious subspace embedding and preservation of norms, together with a new

uniform convergence result for gradients of Lipschitz GLMs. The full proof is deferred to Appendix E. Below, we instantiate the above with our proposed algorithms. **Corollary 6.2. Under the assumptions of Theorem 6.1, Algorithm 4 run with A as** $1. Private Spiderboost (Alg. 2) yields \|\nabla F(\bar{w}; D)\| =$ \widetilde{O}

\sqrt{n}
+ min
\sqrt \operatorname{rank} n\varepsilon 2/3, 1 (n\varepsilon)2/5

.
2. Algorithm 3 with NoisyGD as PrivateSubRoutine, under the additional assumption that w \to f(w; (x, y))
$is convex for all x, y, yields \|\nabla F(\bar{w}; D)\| =$ \widetilde{O}

\sqrt{n}
+ min
\sqrt \operatorname{rank} n\varepsilon, 1 \sqrt{n}\varepsilon

. We remark that the above technique also gives bounds on empirical stationarity. In particular, the first term 1 \sqrt{n}, in the above guarantees, is the uniform convergence bound and the second term is the bound on empirical stationarity.
## Acknowledgements
RA and EU are supported, in part, by NSF BIGDATA award IIS-1838139 and NSF CAREER award IIS-1943251. RB's and MM's research is supported by NSF CAREER Award 2144532 and NSF Award AF-1908281. CG and TG's research was partially supported by INRIA Associate Teams project, FONDECYT 1210362 grant, ANID Anillo ACT210005 grant, and National Center for Artificial Intelligence CENIA FB210017, Basal ANID.
## References
Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. In 23rd ACM Conference on Computer and Communications Security, CCS '16, pp. 308-318, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450341394. doi: 10.1145/2976749.2978318. URL https://doi. org/10.1145/2976749.2978318. Allen-Zhu, Z. How to make the gradients small stochastically: Even faster convex and nonconvex sgd. Advances in Neural Information Processing Systems, 31, 2018. Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro, N., and Woodworth, B. Lower bounds for non-convex stochastic optimization, 2019. Arora, R., Bassily, R., Guzmán, C., Menart, M., and Ullah, E. Differentially private generalized linear models revisited. arXiv preprint arXiv:2205.03014, 2022. Asi, H., Feldman, V., Koren, T., and Talwar, K. Private stochastic convex optimization: Optimal rates in l1 geometry. In International Conference on Machine Learning, pp. 393-403. PMLR, 2021. $Bassily, R., Smith, A., and Thakurta, A. Private empirical$ risk minimization: Efficient algorithms and tight error bounds. In 2014 IEEE 55th Annual Symposium on Foundations of Computer Science, pp. 464-473. IEEE, 2014.
$$
Bassily, R., Feldman, V., Talwar, K., and Guha Thakurta,
$$
## A. Private stochastic convex optimization with optimal
rates. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., and Garnett,
## R. (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates,
Inc., 2019. URL https://proceedings. neurips.cc/paper/2019/file/ 3bd8fdb090f1f5eb66a00c84dbc5ad51-Paper. pdf. $Bassily, R., Guzmán, C., and Menart, M. Differentially$ private stochastic optimization: New results in convex and non-convex settings. Advances in Neural Information Processing Systems, 34, 2021a. Bassily, R., Guzman, C., and Nandi, A. Noneuclidean differentially private stochastic convex optimization. In Belkin, M. and Kpotufe, S. (eds.), Proceedings of Thirty Fourth Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pp. 474-499. PMLR, 15-19 Aug 2021b. URL https://proceedings.mlr. press/v134/bassily21a.html. Bousquet, O. and Elisseeff, A. Stability and generalization. The Journal of Machine Learning Research, 2:499-526,
2002.
Bun, M., Dwork, C., Rothblum, G. N., and Steinke, T. Composable and versatile privacy via truncated cdp. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing, STOC 2018, pp. 74-86, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450355599. doi: 10.1145/
### 3188745.3188946. URL https://doi.org/10.
1145/3188745.3188946. Carmon, Y., Duchi, J. C., Hinder, O., and Sidford, A. ”convex until proven guilty”: Dimension-free acceleration of gradient descent on non-convex functions. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, pp. 654-663. JMLR.org,
2017.
Chaudhuri, K., Monteleoni, C., and Sarwate, A. D. Differentially private empirical risk minimization. Journal of Machine Learning Research, 12(Mar):1069-1109, 2011.

Cohen, M. B. Nearly tight oblivious subspace embeddings by trace inequalities. In Proceedings of the twenty-seventh annual ACM-SIAM symposium on Discrete algorithms, pp. 278-287. SIAM, 2016. Cutkosky, A. and Orabona, F. Momentum-based variance reduction in non-convex sgd. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., $Fox, E., and Garnett, R. (eds.), Advances in Neural$ Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings. neurips.cc/paper/2019/file/ b8002139cdde66b87638f7f91d169d96-Paper. pdf. Diakonikolas, J. and Guzmán, C. Complementary composite minimization, small gradients in general norms, and applications, 2023. Duchi, J. Lecture notes for statistics 311/electrical engineering 377. URL: https://stanford. edu/class/stats311/Lectures/full notes. pdf. Last visited on, 2:23, 2016. Dwork, C. and Roth, A. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science, 9(3-4):211-407, 2014. Dwork, C., McSherry, F., Nissim, K., and Smith, A. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, pp. 265-284. Springer, 2006. Fang, C., Li, C. J., Lin, Z., and Zhang, T. Spider: Near-optimal non-convex optimization via stochastic path-integrated differential estimator. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa- $Bianchi, N., and Garnett, R. (eds.), Advances in Neural$ Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings. neurips.cc/paper/2018/file/ 1543843a4723ed2ab08e18053ae6dc5b-Paper. pdf. Feldman, V., Koren, T., and Talwar, K. Private stochastic convex optimization: optimal rates in linear time. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing, pp. 439-449, 2020. Foster, D. J., Sekhari, A., and Sridharan, K. Uniform convergence of gradients for non-convex learning and optimization. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett,
## R. (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates,
Inc., 2018. URL https://proceedings. neurips.cc/paper/2018/file/ 59ab3ba90ae4b4ab84fe69de7b8e3f5f-Paper. pdf. Foster, D. J., Sekhari, A., Shamir, O., Srebro, N., Sridharan, K., and Woodworth, B. The complexity of making the gradient small in stochastic convex optimization. In Conference on Learning Theory, pp. 1319-1345. PMLR,
2019.
Ge, R., Lee, J. D., and Ma, T. Matrix completion has no spurious local minimum. In Lee, D., Sugiyama, M., Luxburg, $U., Guyon, I., and Garnett, R. (eds.), Advances in Neural$ Information Processing Systems, volume 29. Curran Associates, Inc., 2016. URL https://proceedings. neurips.cc/paper/2016/file/ 7fb8ceb3bd59c7956b1df66729296a4c-Paper. pdf. Ghadimi, S. and Lan, G. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341-2368, 2013. Ghadimi, S. and Lan, G. Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming, 156(1):59-99, 2016. Jain, P. and Thakurta, A. (near) dimension independent risk bounds for differentially private learning. In ICML, 2014. Jain, P., Kothari, P., and Thakurta, A. Differentially private online learning. In 25th Annual Conference on Learning Theory (COLT), pp. 24.1-24.34, 2012.
$$
Jin, C., Netrapalli, P., Ge, R., Kakade, S. M., and Jordan,
$$
## M. I. A short note on concentration inequalities for
random vectors with subgaussian norm. arXiv preprint arXiv:1902.03736, 2019. Kamath, G. and Ullman, J. A primer on private statistics. arXiv preprint arXiv:2005.00010, 2020. Kifer, D., Smith, A., and Thakurta, A. Private convex empirical risk minimization and high-dimensional regression. In Conference on Learning Theory, pp. 25-1, 2012. Kulkarni, J., Lee, Y. T., and Liu, D. Private non-smooth erm and sco in subquadratic steps. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan,
## J. W. (eds.), Advances in Neural Information Processing
Systems, volume 34, pp. 4053-4064. Curran Associates, Inc., 2021. URL https://proceedings. neurips.cc/paper/2021/file/ 211c1e0b83b9c69fa9c4bdede203c1e3-Paper. pdf. Lan, G. First-order and stochastic optimization methods for machine learning. Springer, 2020. Ma, C., Wang, K., Chi, Y., and Chen, Y. Implicit regularization in nonconvex statistical estimation: Gradient descent converges linearly for phase retrieval and matrix

completion. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 3345-3354. PMLR, 10-15 Jul 2018. URL https://proceedings.mlr.press/v80/ ma18c.html. Nemirovsky, A. S. and Yudin, D. B. Problem complexity and method efficiency in optimization. Wiley-Interscience,
1983.
Nesterov, Y. How to make the gradients small. Optima. Mathematical Optimization Society Newsletter, (88):10- 11, 2012. Nesterov, Y. and Polyak, B. Cubic regularization of newton method and its global performance. Mathematical Programming, 108:177-205, 2006. $Rudelson, M. and Vershynin, R. Non-asymptotic theory of$ random matrices: extreme singular values. In Proceedings of the International Congress of Mathematicians 2010 (ICM 2010) (In 4 Volumes) Vol. I: Plenary Lectures and Ceremonies Vols. II-IV: Invited Lectures, pp. 1576-1602. World Scientific, 2010. Song, S., Steinke, T., Thakkar, O., and Thakurta, A. Evading the curse of dimensionality in unconstrained private glms. In International Conference on Artificial Intelligence and Statistics, pp. 2638-2646. PMLR, 2021. Steinke, T. and Ullman, J. Between pure and approximate differential privacy. Journal of Privacy and Confidentiality, 7, 01 2015. doi: 10.29012/jpc.v7i2.648. Sun, J., Qu, Q., and Wright, J. A geometric analysis of phase retrieval. In 2016 IEEE International Symposium on Information Theory (ISIT), pp. 2379-2383, 2016. doi:
10.1109/ISIT.2016.7541725.
Talwar, K., Thakurta, A., and Zhang, L. Private empirical risk minimization beyond the worst case: The effect of the constraint set geometry. arXiv preprint arXiv:1411.5417,
2014.
Talwar, K., Thakurta, A., and Zhang, L. Nearly optimal private lasso. In NIPS, 2015. Tran, H. and Cutkosky, A. Momentum aggregation for private non-convex erm. In Advances in Neural Information Processing Systems, volume 35. Curran Associates, Inc., 2022. URL https://openreview. net/pdf?id=x56v-UN7BjD. Wang, D. and Xu, J. Differentially private empirical risk minimization with smooth non-convex loss functions: A non-stationary view. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 1182-1189,
2019.
Wang, D., Ye, M., and Xu, J. Differentially private empirical risk minimization revisited: Faster and more general. Advances in Neural Information Processing Systems, 30,
2017.
Wang, D., Chen, C., and Xu, J. Differentially private empirical risk minimization with non-convex loss functions. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 6526-6535. PMLR, 09- 15 Jun 2019a. URL https://proceedings.mlr. press/v97/wang19c.html. Wang, L., Jayaraman, B., Evans, D., and Gu, Q. Efficient privacy-preserving nonconvex optimization. CoRR, abs/1910.13659, 2019b. URL http://arxiv.org/ abs/1910.13659. Wang, Z., Ji, K., Zhou, Y., Liang, Y., and Tarokh,
## V. Spiderboost and momentum: Faster variance
reduction algorithms. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., and $Garnett, R. (eds.), Advances in Neural Information$ Processing Systems, volume 32. Curran Associates, Inc., 2019c. URL https://proceedings. neurips.cc/paper/2019/file/ 512c5cad6c37edb98ae91c8a76c3a291-Paper. pdf. Zhang, J., Zheng, K., Mou, W., and Wang, L. Efficient private erm for smooth objectives. In Proceedings of the 26th International Joint Conference on Artificial Intelligence, IJCAI'17, pp. 3922-3928. AAAI Press, 2017. ISBN 9780999241103. Zhang, Q., Ma, J., Lou, J., and Xiong, L. Private stochastic non-convex optimization with improved utility rates. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pp. 3370-3376,
2021.
Zhou, Y., Chen, X., Hong, M., Wu, Z. S., and Banerjee, A. Private stochastic non-convex optimization: Adaptive algorithms and tighter generalization bounds. CoRR, abs/2006.13501, 2020. URL https://arxiv.org/ abs/2006.13501.

## A. Lower bounds
### A.1. Missing details from DP Empirical Stationarity Lower Bound
Proof of Theorem 4.3. For any r > 0, let Wr denote the ball of radius r centered at the origin. Let B = L_0 L_1. Consider the loss function: f(w; x) = ( L_1 2 \|w - x\| $if \|w - x\| \leq B$ L_0 \|w - x\| - L^2 2L1 otherwise The function f(w; x) is convex, L_1-smooth and L_0-Lipschitz in \mathbb{R}^d
$$
. We restrict to datasets S = {x_i} n i=1 where x_i \in WB/4 for all i, and let F(w; S) = 1 n Pn i=1 f(w; x_i) be the empirical risk on S. The unconstrained minimizer of F(w; S) is w^\ast = 1 n Pn i=1 x_i which lies in WB/4. For any w \in W3B/4, w lies in the quadratic region around all data points. Hence, from L_1-strong convexity of w \to F(w; S) on W3B/4, we have that whenever \bar{w} \in W3B/4,
$$
\|\nabla F(\bar{w}; S)\| \|\bar{w} - w^\ast $\| \geq \langle \nabla F(\bar{w}; S), w^\ast$
- \bar{w} \rangle \geq F(\bar{w}; S) - F(w^\ast
; S) \geq L_1 $\|\bar{w} - w^\ast$ \|. Let E be the event that \bar{w} \in W3B/4 and let \mathbb{E}_E denote the conditional expectation (conditioned on event E) operator. Then, $\mathbb{E}_E \|\nabla F(\bar{w}; S)\| \geq$ L_1 $E \|\bar{w} - w^\ast$ \| \geq L_1 \Omega

L_0 4L1

min 1, p d log (1/$\delta$) n\varepsilon !!. where the last inequality follows from known lower bounds for DP mean estimation (Steinke & Ullman, 2015; Kamath & Ullman, 2020). We remark that the lower bound in the referenced work is for algorithms which produce outputs in the ball of the same radius as the dataset, i.e. WB/4. However, a simple post-processing argument shows that the same lower bound applies to algorithms which produce output in W3B/4. Specifically, assuming the contrary, we simply project the output in W3B/4 to WB/4: privacy is preserved by post-processing and the distance to the mean cannot increase by the non-expansiveness property of projection to convex sets, hence a contradiction. This gives us, $\mathbb{E}_E [\|\nabla F(\bar{w}; S)\|] \geq \Omega L_0 min 1,$ p d log (1/$\delta$) n\varepsilon !! Let W̃ = {w: \|w - w^\ast \| \leq B/2}. Since W̃ \subseteq W3B/4, we have that the above conditional lower bound applies for \bar{w} \in W̃ as well. We now consider \bar{w} ̸\in W̃. Let w' be any point on the boundary of W̃, denoted as \partialW. Note that w' lies in the region where, for any data point, the corresponding loss is a quadratic function. Hence, by direct computation, \nabla F(w'; S) = L_1 (w'
- w^\ast
). Therefore, \langle \nabla F(w' ), w'
- w^\ast
\rangle = L_1 \|w'
- w^\ast
\| = L1B2. We now apply Lemma A.1 which gives us, $EEc \|\nabla F(\bar{w}; S)\| \geq$ L1B2 \cdot B = L_0, where E^c denotes the complement set of E. We combine the above bounds using the law of total expectation as follows, \mathbb{E}[\|\nabla F(\bar{w}; S)\|] = \mathbb{E}_E[\|\nabla F(\bar{w}; S)\|]\mathbb{P}\{\bar{w} \in E} + EEc [\|\nabla F(\bar{w}; S)\|]\mathbb{P}\{\bar{w} \in E^c } = \Omega

L_0 min n 1, p d log (1/$\delta$) n\varepsilon o $P(\bar{w} \in E) + \Omega(L_0)P(\bar{w} \in E^c$ ) = \Omega

L_0 min n 1, p d log (1/$\delta$) n\varepsilon o. This completes the proof.

**Lemma A.1. Let G, \mathbb{R} \geq 0, d \in N. Let WR(w_0) denote the Euclidean ball around w_0 of radius \mathbb{R} and let \partialWR(w_0) denote** its boundary. Let f: \mathbb{R}^d $\to \mathbb{R} be a differentiable convex function. Suppose w_0 \in \mathbb{R}^d$ is such that for every v \in \partialWR(w_0),
$$
\langle \nabla f(v), v - w_0 \rangle \geq G, then for any w ̸\in WR(w_0), we have \|\nabla f(w)\| \geq G R. Proof. For a unit vector u \in \mathbb{R}^d, define directional directive f' u(w) = \langle \nabla f(w), u \rangle. We first show that for any u \in \mathbb{R}^d: \|u\| = 1 and any w' \in \mathbb{R}^d, the function f' u(w' + ru) is non-decreasing in r \in R+. This simply follows from monotonicity of gradients since f is convex. In particular, for any r' > r > 0, we have f' u(w' + r' u) - f' u(w'
$$
+ ru) = \langle \nabla f(w'
+ r'
u) - \nabla f(w'
+ ru), u \rangle
= r' - r \langle \nabla f(w'
+ r'
u) - \nabla f(w'
+ ru), w'
+ ru - (w'
+ ru) \rangle
> 0
We now prove the claim in the lemma statement. Let w ̸\in \partialWR and define u = w-w_0 $\|w-w_0\|. Then from Cauchy-Schwarz$ inequality and the above monotonicity property, we have, $\|\nabla f(w)\| \geq \langle \nabla f(w), u \rangle = f'$ u(w) \geq f' $u(w_0 + Ru) = \langle \nabla f(w_0 + Ru), u \rangle$ = \mathbb{R} $\langle \nabla f(v), v - w_0 \rangle \geq$ G \mathbb{R} which finishes the proof.
### A.2. Non-private Sample Complexity Lower Bound
**Theorem A.2. For any L_0, L_1, n, d \in N, there exists a distribution D over some set X and a L_0-Lipschitz, L_1-smooth** (convex) loss function w \to f(w; x) such that given n i.i.d samples from D, the output \bar{w} of any algorithm satisfies, $E \|\nabla F(\bar{w}; D)\| = \Omega$

L_0 \sqrt{n}

Proof. We construct a hard instance in d = 1 dimension. Let p \in [0, 1] be a parameter to be set later and let v \in {-1, 1} be chosen by an adversary. Let the data domain X = {-1, 1} and consider the distribution D on X as follows: x = ( 1 with probability 1+vp -1 with probability 1-vp Note that \mathbb{E}[x] = vp. Consider the loss function f(w; x) as f(w; x) = L_0 wx + L_1 \Delta(w) where \Delta is the Huber regularization function, defined as, \Delta(w) = (
|w|
if |w| \leq L_0 2L1 L_0|w| L_1 - L^2 4L2 otherwise Note that the loss function w \to f(w; x) is convex, L_0-Lipschitz and L_1-smooth in \mathbb{R}^d, for all x. The population risk function is, F(w; D) = L_0 wpv + L_1 \Delta(w) Let \bar{w} be output some algorithm given n i.i.d. samples from D. Consider two cases:

$$
Case 1: |\bar{w}| > L_0
$$
2L1: The gradient norm in this case is
|\nabla F(\bar{w}; D)|
= L_0 vp + L_0\bar{w} 2 |\bar{w}| = L^2 0p2 + L^2 + L^2 2 |\bar{w}| vp\bar{w} \geq L^2 - L^2 p = L^2 - L^2 \sqrt{n} \geq L^2 where the first inequality follows since v \bar{w}
|\bar{w}| \geq -1, the third equality follows by setting p = 1
\sqrt 16n and the second inequality follows since n \geq 1. We therefore have that E |\nabla F(\bar{w}; D)| \geq L_0 \sqrt.
$$
Case 2: |\bar{w}| \leq L_0
$$
2L1: In this case, the gradient norm is,
|\nabla F(\bar{w}; D)|
= L_0 vp + L_1\bar{w} Suppose there exists an algorithm with output \bar{w}, which, with n samples guarantees that E |\nabla F(\bar{w}; D)| < o

L_0 \sqrt{n}

. Then from Markov's inequality, with probability at least 0.9, we have that |\nabla F(\bar{w}; D)| < o

L^2 n

. Let w̃ = -2L1\bar{w} L_0, then we have that with probability at least 0.9,
|\nabla F(\bar{w}; D)|
\leq o

L^2 n

⇐\Rightarrow |vp - w̃| < o

n

This contradicts the well-known bias estimation lower bounds, with p = 1 \sqrt 16n, using Le Cam's method ((Duchi, 2016), $Example 7.7), hence E |\nabla F(\bar{w}; D)| \geq \Omega$

L_0 \sqrt{n}

. Combining the two cases finishes the proof.
## B. Missing Results for Empirical Stationary Points
### B.1. Private Spiderboost
The following lemma largely follows from the analysis in (Wang et al., 2019c). We present a full proof below for completeness. **Lemma B.1. Let the conditions of Lemma 4.1 be satisfied. Let \eta \leq 1** 2L1 and q \leq O

$\tau$^2 2 \eta2

. Then the output of Private SpiderBoost, \bar{w} satisfies $E [\|\nabla F(\bar{w}; S)\|] = O$ s F_0 \etaT
+ $\tau$1
!. (1) Proof. In the following, for any t \in [T], let st = j t q k q (i.e. the index corresponding to the start of the phase containing iteration t). By a standard analysis for smooth functions we have (recalling that \nabla t is an unbiased estimate of \nabla F(w_t; S) for any t \in [T]) $F(w_t+1; S) \leq F(w_t; S) +$ \eta $\|\nabla F(w_t; S) - \nabla t\|$ -

\eta - L1\eta2

\|\nabla t\|.

Taking expectation we have the following manipulation using the update rule of Algorithm 2 $E [F(w_t+1; S) - F(w_t; S)] \leq$ \eta E h $\|\nabla F(w_t; S) - \nabla t\|$ i -

\eta - L1\eta2

E h \|\nabla t\| i \leq \eta$\tau$^2 t X k=st+1 E h $\|w_k+1 - w_k\|$ i + \eta E h $\|\nabla st - F(wst; S)\|$ i -

\eta - L1\eta2

E h \|\nabla t\| i \leq \eta3 $\tau$^2 t X k=st+1 E h \|\nabla k\| i + \eta$\tau$^2 -

\eta - L1\eta2

E h \|\nabla t\| i, where the second inequality follows from Lemma 4.1 and the last inequality follows from the update rule. Note that if t = st the sum is empty. Summing over a given phase we have $E [F(w_t+1; S) - F(wst; S)] \leq$ \eta3 $\tau$^2 t X k=st k X j=st+1 E h \|\nabla j\| i + t X k=st h \eta$\tau$^2 2 -

\eta 2 - L1\eta2

E h \|\nabla k\| ii \leq \eta3 $\tau$^2 2 q t X k=st E h \|\nabla k\| i + t X k=st h \eta$\tau$^2 2 -

\eta 2 - L1\eta2

E h \|\nabla k\| ii = - t X k=st " \eta - L1\eta2 - \eta3 $\tau$^2 2 q

| {z }
A E h \|\nabla k\| i - \eta$\tau$^2
#
, (2) where the second inequality comes from the fact that each gradient appears at most q times in the sum. We now sum over all phases. Let P = {p0, p1,...,} = n 0, q, 2q,..., j T -1 q k q, T o. We have $E [F(wT; S) - F(w_0; S)] \leq$
|P |
X i=1 E

F(wpi; S) - F(wpi-1; S)

\leq - T X t=0 A E h \|\nabla k\| i + T\eta$\tau$^2. Rearranging the above yields T T X t=0 E h \|\nabla k\| i \leq F_0 TA + \eta$\tau$^2 2A. (3) Now let i∗ denote the index of \bar{w} selected by the algorithm. Note that E h $\|\nabla F(wi∗; S)\|$ i \leq 2E h $\|\nabla F(wi∗; S) - \nabla i∗ \|$ i
+ 2E
h \|\nabla i∗ \| i. (4)

The second term above can be bounded via inequality (3). To bound the first term we have by Lemma 4.1 that E h $\|\nabla i∗ - \nabla F(wi∗; S)\|$ i \leq $\tau$^2 t∗ X k=st∗ +1 E h $\|w_k - w_k-1\|$ i
+ $\tau$^2
= \eta2 $\tau$^2 t∗ X k=st∗ +1 E h \|\nabla k\| i
+ $\tau$^2
\leq q\eta2 $\tau$^2 T T X k=0 E h \|\nabla k\| i
+ $\tau$^2
\leq $\tau$^2 2 \eta2 qF0 TA + \eta3 q$\tau$^2 2A $\tau$^2 1 + $\tau$^2 1, where the last inequality comes from inequality (3) and the expectation over i∗. Plugging into inequality (4) one can obtain E h $\|\nabla F(wi∗; S)\|$ i \leq 2F0 TA (1 + $\tau$^2 2 \eta2 q) +

\eta A
+ 2 +
$\tau$^2 2 \eta3 q A

$\tau$^2 1. (5) Now recall A = \eta 2 - L1\eta2 2 - \eta3 $\tau$^2 2 q 2. Since q \leq O

$\tau$^2 2 \eta2

and \eta \leq 1 2L1 we have A = \Theta(\eta). Thus plugging into inequality (5) and again using the fact that q \leq O

$\tau$^2 2 \eta2

we have E h $\|\nabla F(wi∗; S)\|$ i = O

F_0 T\eta (1 + $\tau$^2 2 \eta2 q) +

3 + $\tau$^2 2 \eta3 q A

$\tau$^2

= O

F_0 T\eta
+ $\tau$^2

. The claim then follows from the Jensen inequality. For privacy, we will rely on the moments accountant analysis of (Abadi et al., 2016). This roughly gives the same analysis as using privacy amplification via subsampling and the advanced composition theorem, but allows for improvements in log factors. We provide the following theorem implicit in (Abadi et al., 2016) Theorem 1 below. The same result can be obtained using the analysis for (Kulkarni et al., 2021) Theorem 3.1 which uses the truncated central differential privacy guarantees of the Gaussian mechanism (Bun et al., 2018). **Theorem B.2 ((Abadi et al., 2016; Kulkarni et al., 2021)). Let $\varepsilon$, $\delta$ \in (0, 1] and c be a universal constant. Let D \in Yn** be a $dataset over some domain Y, and let h1,..., hT: Y \to \mathbb{R}^d$ be a series of (possibly adaptive) queries such that for any y \in Y, $t \in [T], \|ht(y)\|2 \leq \lambda_t. Let \sigma_t =$ c\lambda_t \sqrt log(1/$\delta$) $\varepsilon$ max n b, \sqrt T n o. Then the algorithm which samples batches of size B1,.., Bt of size b uniformly at random and outputs 1 n P y\inBt ht(y) + g_t for all t \in [T] where g_t \sim \mathcal{N}(0, I$\sigma$^2 $t ), is (\varepsilon, \delta)-DP.$ We note that the original statement of the Theorem in (Abadi et al., 2016) requires \sigma_t \geq c\lambda_t \sqrt T log(1/$\delta$) n\varepsilon and T \geq n^2 $\varepsilon$ b_2 (or T \geq n^2 $b_2 so long as \varepsilon \leq 1). However, in the case where T \leq n^2$ b_2, one can simply consider the meta algorithm that does run T' = n^2 b_2 steps and only outputs the first T results. This algorithm is at least as private as the algorithm which outputs every result, and under the setting T' the scale of noise is 8\lambda_t \sqrt log(1/$\delta$) b$\varepsilon$. We can now prove the main result for Private Spiderboost, restated below. We note that the setting of b_2 given below will always be less than n under required conditions. More details are provided in the proof below. **Theorem B.3 (Private Spiderboost). Let n \geq max**

(L_0$\varepsilon$)2 F0L1d log(1/$\delta$), \sqrt{d} max{1, \sqrt L1F0/L_0} $\varepsilon$

. Private Spiderboost run with parameter settings \eta = 1 2L1, b_1 = n, b_2 = $ max ( L0n\varepsilon \sqrt F0L1d log(1/$\delta$) 2/3 $, (L0nd log(1/\delta))1/3$ (L1F0)1/6\varepsilon^2/3 )%, T = $ max ( (F0L1)1/4 n\varepsilon \sqrt L0d log(1/$\delta$) 4/3, n\varepsilon \sqrt{d} log(1/$\delta$) )%, and q = j n^2 \varepsilon^2 L^2 1T d log(1/$\delta$) k satisfies $E [\|\nabla F(w̃)\|] = O$ \left(

p F0L1L0d log (1/$\delta$) n\varepsilon !2/3 + p d log (1/$\delta$)L_0 n\varepsilon \right) $is (\varepsilon, \delta)-DP and has oracle complexity \widetilde{O}$ max

n^{5/3} \varepsilon^2/3 d^{1/3}

,

n\varepsilon \sqrt{d} 2

. Proof. For privacy, we rely on the moment accountant analysis of the Gaussian mechanism as per Theorem B.2. Note that each gradient estimate computed in line 9 has elements with \ell2-norm at most L_0, and this estimate is computed at most T q times. Similarly, for a gradient variation at step t in line 13 we have norm bound L_1 \|w_t - w_t-1\|, and have that at most T such estimates are computed. As such, the scale of noise in both cases ensures the overall algorithm is $($\varepsilon$, $\delta$)$-DP by Theorem B.2. We now prove the convergence result. To simplify notation in the following, we define \alphā = \sqrt{d} log(1/$\delta$) n\varepsilon. If b_1 = n (full batch gradient), the conditions of Lemma 4.1 are satisfied with $\tau$^2 1 = O

L^2 0T \alphā2 q

and $\tau$^2 2 = O

L^2 b_2
+ L^2
1T\alphā2

and some setting of q so long as T \geq q n^2 b_2 = q and T \geq n^2 b_2. Further, if b_2 \geq 1 T \alphā2 then $\tau$^2 2 = O L^2 1T\alphā2

. Thus the condition on q in **Lemma B.1 is satisfied with q =** L^2 $\tau$^2 = 1 T \alphā2 since \eta = 1 2L1 Plugging into Eqn. (1) we obtain $E [\|\nabla F(w̃)\|] = O$ r F0L1 T + L_0 \sqrt T\alphā \sqrt q ! = O r F0L1 T
+ L0T\alphā2
!. (6) We now consider the setting of T. Since q = 1 T \alphā2, it suffices to set T \geq 1 \alphā to ensure T \geq q. We now set T = max

(L1F0)1/4 \sqrt L_0\alphā 4/3, 1 \alphā

. Using Eqn. (6) above we have $E [\|\nabla F(w̃)\|] = O$ p F0L1L0\alphā 2/3
+ L_0\alphā

. The claimed rate now follows if there exists a valid setting for b_2 satisfying the previously stated conditions. The restrictions on the batch size implied by T imply we need b_2 \geq n \sqrt T and thus it suffices to have b_2 \geq L 1/3 0 n\alphā2/3 (L1F0)1/6 to satisfy this condition since T \geq

(L1F0)1/4 \sqrt L_0\alphā 4/3. We recall that for the setting of q to be valid we also require b_2 \geq 1 T \alphā2 and because T \geq

(L1F0)1/4 \sqrt L_0\alphā 4/3 it suffices that b_2 \geq

L_0 \sqrt F0L1\alphā 2/3. Thus we need b_2 = max

L_0 \sqrt F0L1\alphā 2/3, L 1/3 0 n\alphā2/3 (L1F0)1/6

. Finally, we need b_2 \leq n whenever q \geq 1. Note that by the setting of q and T we have q \leq

L_0 \sqrt F0L1\alphā 2/3 and thus q \geq 1 =\Rightarrow \sqrt L1F0\alphā L_0

\leq 1. Under this same condition we have L 1/3 0 n\alphā2/3 (L1F0)1/6 \leq n. We further have

L_0 \sqrt F0L1\alphā 2/3 \leq n under the assumption n \geq (L_0$\varepsilon$)2 F0L1d log(1/$\delta$) given in the theorem statement. It can also be verified that under the condition on n given in the theorem statement that q \geq 1. Thus the parameter settings obtain the claimed rate. Note the number of gradient computations is bounded by O

Tb2 + Tb1 q

= \widetilde{O}

n\varepsilon \sqrt{d} 4/3 max ( n\varepsilon \sqrt{d} 2/3, (nd)1/3 \varepsilon^2/3 )
+ n

n\varepsilon \sqrt{d} 2/3 ! = \widetilde{O} max ( n\varepsilon \sqrt{d} 2, n^{5/3} \varepsilon^2/3 d^{1/3} )!.
### B.2. Additional Discussion of Rate Improvement Challenges
We here give a more detailed version of the informal discussion in Section 4.2. We want to emphasize that the goal of the following discussion is not to provide a universal lower bound, but rather to inform future research.

Let L: \mathbb{R}^d 7\to \mathbb{R} be a loss function. We say the randomized mapping O: \mathbb{R}^d $\times (\mathbb{R}^d$ ∪ \perp) \to \mathbb{R}^d $, is a (\tau1, \tau^2)-accurate$ oracle for L if ∀w, w' \in \mathbb{R}^d E O $[O(w, \perp)] = \nabla L(w), E$ O [O(w, w' $)] = \nabla L(w) - \nabla L(w'$ ) E O h $\|O(w, \perp) - \nabla L(w)\|$ i \leq $\tau$^2 1, E O h \|O(w, w' )\| i \leq $\tau$^2 2 \|w - w' \|. In short, O is an unbiased and accurate gradient/gradient variation oracle for L. Define $m(G, L_1, L_0, \tau1, \tau^2) = inf$ A sup O,L inf n $\alpha: E [\|\nabla L(A(O, L_1, L_0, \tau1, \tau^2)\|] \leq \alpha$ o, where the supremum is taken over L_1-smooth functions L satisfying L(0) - arg min w\inRd ${L(w)} \leq L_0, and (\tau1, \tau^2)-accurate$ oracles for L. The infimum is taken over algorithms which make at most G calls to O. We have the following lower bound on m (i.e. a lower bound on the accuracy of optimization algorithms which make at most G queries to the oracle) following from (Arjevani et al., 2019, Theorem 3) and the fact that the oracle model described above is a special case of the multi-query oracles considered by (Arjevani et al., 2019). **Theorem B.4 ((Arjevani et al., 2019)). Let G, L_0, L_1, $\tau$1, $\tau$^2 \geq 0 and define \alpha = L_0$\tau$^2$\tau$1** G 1/3
+ $\tau$1
\sqrt G. If d = \Omegã L0L1 \alpha^2 2

, $then m(G, L_1, L_0, \tau1, \tau^2) = \Omega (\alpha).$ Now consider L such that L(w) = 1 n P x\inS \ell(w; x) for some L_0-Lipschitz and L_1-smooth loss \ell: \mathbb{R}^d \times X \to \mathbb{R} and S \in $\mathcal{X}^n$. We are interested in designing some (b $\tau$1, b $\tau$^2)-accurate and differentially private oracle, b O, which can then be used by an optimization algorithm, A, to obtain an approximate stationary point \bar{w} = A( b O, L_1, L_0, b $\tau$1, b $\tau$^2). Specifically, we want b O to be capable of answering G queries under $($\varepsilon$, $\delta$)$-DP. A common method for achieving this is to ensure each query to O is at least ( $\varepsilon$ \sqrt G, $\delta$)-DP and use advanced composition (or the more refined moment accountant) analysis. Such a setup encapsulates numerous results in the convex setting (Bassily et al., 2019; Kulkarni et al., 2021), and is even more dominant in non-convex settings (Wang et al., 2017; Zhou et al., 2020; Abadi et al., 2016). Our key observation is that under such a setup, any increase in the number of oracle calls to G must be met with a proportional increase in the accuracy parameters (b $\tau$1, b $\tau$^2). Thus, if such an oracle, b O is applied in a black box fashion to a stochastic optimization algorithm A, one can obtain a lower bound on the accuracy of the overall algorithm independent of G. Specifically, since estimating the gradient and gradient variation can be viewed as mean estimation problems on n vectors, we can use fingerprinting code arguments to lower bound b $\tau$1 and b $\tau^2 (Steinke & Ullman, 2015). In Lemma B.5 below, we$ prove that any (b $\tau$1, b $\tau^2)-accurate oracle which ensures that any query is ( \varepsilon$ \sqrt G, $\delta$)-DP must have b $\tau$1 = \Omega

L_0 \sqrt Gd log(1/$\delta$) n\varepsilon

and b $\tau$^2 = \Omega

L_1 \sqrt Gd log(1/$\delta$) n\varepsilon

. Now, observe that by Theorem B.4, we have $m(G, L_1, L_0, b$ $\tau$1, b $\tau$^2) = \Omega \left(

\sqrt F0L1L0 p d log (1/$\delta$) n\varepsilon !2/3 + L_0 p d log (1/$\delta$) n\varepsilon \right), which matches our upper bound. We now remark on several ways the above barrier could be circumvented. The first and most obvious possibility is to employ a different privatization method than private oracles. However, this is particularly difficult in the nonconvex setting as existing methods which avoid private gradients (see e.g. (Feldman et al., 2020) for several such methods) rely crucially on stability guarantees arising from convexity. Other possible ways to beat the above rate is by designing a stochastic optimization algorithm which leverages the structure of the noise used in private implementations of the oracle or makes use of additional assumptions to beat the \Omega

L_0$\tau$^2$\tau$1 G 1/3
+ $\tau$1
\sqrt G

non-private lower bound. Additional Details on Fingerprinting Bound We conclude by giving a concrete construction for the fingerprinting argument mentioned above.

**Lemma B.5. Let L_0, L_1 \geq 0, \varepsilon = O(1), 2-\Omega(n)** $\leq \delta \leq 1 n1+\Omega(1) and p$ d log (1/\delta)/(n\varepsilon) = O(1). Let \ell, L, S satisfy the assumptions above. Then there exists \ell, S such that for any oracle, O, which is ($\tau$1, $\tau$^2)-accurate for L it holds that $\tau$1 = \Omega L_0 p d log (1/$\delta$) n\varepsilon ! and $\tau$^2 = \Omega L_1 p d log (1/$\delta$) n\varepsilon !. Proof. In the following, we use uj to denote the j'th component of some vector u. Let B = L_0 L_1 \sqrt{d} $and define h: \mathbb{R} \to \mathbb{R} as$ h(z) = (L_1 2 w_2 if|w| \leq B L_0 \sqrt{d}
|w| -
L^2 2dL1 otherwise Define d' = d 2 (assume d is even for simplicity) and for any vector u \in \mathbb{R}^d let u(1) $= [u1,..., ud' ]^\top$ and u(2) = $[ud'+1,..., ud]^\top$. Define \ell(w; x) = \ell1(w; x) + \ell2(w; x) where \ell1(w; x) = L_0 \sqrt{d} D w(1), x(1) E, \ell2(w; x) = d X j=d'+1 h(wj)xj. Let W = {w: \|w\|\infty \leq B} and note for any w \in W we have $\nabla \ell(w; x) = [$ x_1 \sqrt{d},..., xd' \sqrt{d} $, wd'+1xd'+1,..., wdxd]^\top$, \nabla 2 $\ell2(w; x) = L_1 \cdot Diag(0,..., 0, xd'+1,..., xd)$ That is, the Hessian of \ell2(w; x) is a diagonal matrix with entries from x. Thus one can observe that for any x \in {±1} d we have that \ell(\cdot; x) is L_0-Lipschitz and L_1-smooth over \mathbb{R}^d. To prove a lower bound on $\tau$1 and $\tau$^2, it suffices to show that for any $($\varepsilon$, $\delta$)$-DP implementation of O there exists w \in \mathbb{R}^d such that E O h $\|O(w; \perp) - \nabla L(w)\|$ i \geq $\tau$^2 1 and there exist w, w' \in \mathbb{R}^d such that E O h \|O(w, w' )\| i \geq $\tau$^2 2 \|w - w' \|. For sake of generality, we will show that these properties hold for a set of w, w'. Note that to lower bound the gradient error, it suffices to lower bound the error with respect to the first d' components. We thus argue using \ell1, and will in fact show a lower bound for any w \in \mathbb{R}^d. Let w \in \mathbb{R}^d $. We have for any (\varepsilon, \delta)-DP oracle O$ there exists a dataset S \subseteq {±1} d, where |S| = n, of fingerprinting codes such that E O $[\|O(w; \perp) - \nabla L(w)\|] \geq E$ O " O(w; \perp)(1) - n X x\inS x(1)
#
= \Omega L_0 p d log (1/$\delta$) n\varepsilon !. The bound follows from standard fingerprinting code arguments. See (Bassily et al., 2014, Lemma 5.1) for a lower bound and (Steinke & Ullman, 2015, Theorem 1.1) for a group privacy reduction that obtains the additional p log (1/$\delta$) factor. This fingerprinting result also induces the parameter constraints in the theorem statement. We thus have $\tau$1 = \Omega

L_0 \sqrt{d} log(1/$\delta$) n\varepsilon

. Similarly, we will argue a bound on the gradient variation using \ell2. Let w, w' \in W and u = (w - w' )(2). In what follows, we only use the second half of the components for each vector, and thus omit the superscript (2) from all vectors for $readability. We have \nabla \ell2(w; x) - \nabla \ell2(w'$; x) = L_1[u1x1,..., ud' xd' ]^\top. Then for any c \in (0, 2L0 L_1 \sqrt{d} ] and u \in {±c} we

have E O h \|O(w, w' $) - (\nabla L(w) - \nabla L(w'$ ))\| i = L^2 1 \cdot E O   d' X j=1 O(w, w' )j - uj n X x\inS xj !2   = L^2 1 \cdot E O   d' X j=1 uj O(w, w' )j uj - n X x\inS xj

!2   = L^2 1 \cdot E O  c^2 d' X j=1 O(w, w' )j uj - n X x\inS xj !2   = \Omega

L^2 1c2 d^2 log (1/$\delta$) n^2\varepsilon^2

, where the last step again comes from fingerprinting results. Note that the extra factor of d as compared to the previous bound comes from the fact that we are considering fingerprinting codes with norm larger by a factor of \sqrt{d}. We also use the fact that the vector O(w, w' ) transformed using u is $($\varepsilon$, $\delta$)$-DP by post processing. Now since c = \|w-w' \| \sqrt{d} we have E O [\|O(w, w' $) - (\nabla L(w) - \nabla L(w'$ ))\|] = L_1 \|w - w' $\| p d log (1/\delta) n\varepsilon !. Finally, noting that E O h \|O(w, w'$ ) - (\nabla L(w) - \nabla L(w' ))\| i \leq E O h \|O(w, w' )\| i we obtain $\tau$^2 = \Omega

L_1 \sqrt{d} log(1/$\delta$) n\varepsilon

. This completes the proof. We remark that the accuracy lower bound for the gradient variation can hold for a much more general set of vectors than that given in the proof. Specifically, the same result can be obtained for any u = w - w' such that u has \Theta(d) components which are \Omega \|u\| \sqrt{d}

(i.e. any sufficiently spread out vector). This uses the fact that it suffices to bound the number of components which disagree in sign with the fingerprinting mean and that fingerprinting codes are sampled using a product distribution, and thus the tracing attack used by fingerprinting constructions holds over any sufficiently large subset of dimensions.
## C. Missing Results for Population Stationary Points
Here we present the proof of privacy and accuracy for Algorithm 1. We start by proving the privacy guarantee. Proof of Theorem 3.1. By parallel composition of differential privacy, and since the used batches are disjoint, it suffices to prove that each step in lines 6 and 15 of the algorithm is $($\varepsilon$, $\delta$)$-DP. Note that the gradient estimator in step 6 has \ell2-sensitivity 2L0/b, so by the Gaussian mechanism this step is (\varepsilon, \delta)-DP. For step 15, suppose S_t,s and S' t,s are neighboring datasets that differ in at most one element: x_i∗ ̸= x' i∗, and let \etat,si and \eta' t,si the respective stepsizes used in step 23. Then $\|\Delta_t,s - \Delta'$ t,s\| = 2|s| b $\|\nabla f (w_t,s; x_i∗ ) - \nabla f (w_t,b$ s; x_i∗ ) - (\nabla f (w_t,s; x' $i∗) - \nabla f (w_t,b$ s; x' i∗ )) \|, and note between the parent node u_t,b s and u_t,s there are 2D-|s| iterates generated by the algorithm, which we denote as

w_t,b s = w_t,s0, w_t,s1 $,..., w_t,s^2|D|-s$ = w_t,s. Then, by smoothness of f and the triangle inequality $\|\Delta_t,s - \Delta'$ t,s\| = 2|s| b $\|\nabla f (w_t,s; zi∗ ) - \nabla f (w_t,b$ s; zi∗ ) - (\nabla f (w_t,s; z' $i∗) - \nabla f (w_t,b$ s; z' i∗ )) \| \leq 2D-|s| X i=1 2|s| b

$\|\nabla f (w_t,si$; zi∗ ) - \nabla f w_t,si-1; zi∗

$\| + \| \nabla f (w_t,si$; z' $i∗) - \nabla f w_t,si-1$; z' i∗

\|

\leq 2D-|s| X i=1 2|s| b L1\etat,si-1 \|\nabla t,si-1 \| + 2D-|s| X i=1 2|s| b L1\eta' t,si-1 \|\nabla ' t,si-1 \| = 2 2D-|s| X i=1 2|s| b \beta 2D/2 = 2\beta^2D/2 b. The Gaussian mechanism combined with our choice of \sigma_t,s certifies privacy of this step. To prove Theorem 3.2 we will need some technical lemmas. Define (T, S) as a random stopping time that indicates when
### Algorithm 1. ends. Also, we say (t1, s1) \preceq2 (t^2, s^2) whenever wt1,s1
comes before wt2,s^2 in the algorithm iterates. **Lemma C.1 (Gradient estimation error, extension of Lemma 6 in (Fang et al., 2018)). Let p \in (0, 1). Then, with probability** 1 - p the event $E = {\|\nabla t,s - \nabla F(w_t,s; D)\|2$ \leq \alpha \cdot \alphã ∀(t, s) \preceq2 (T, S)} holds, under the parameter setting of \sigma_t,\emptyset, \sigma_t,s and \etat,s in Algorithm 1, for \alpha^2 \geq

L^2 b + \beta^2 D2D b

max

1, (d + 1) b\varepsilon^2

and \alphã \geq 256 log

1.25
$\delta$

log

2T2D+1 p

\alpha. Proof. Recall the gradient estimate associated to a left child node is the same as that of the parent node. Hence, the gradient estimate of a non-leaf node is the same as that of the left-most leaf of its left sub-tree. In addition, we only need to control the gradient estimation error when we perform a gradient step, which occurs at the leaves. Then, to prove the claim, it suffices to prove that we can control the gradient estimation error at the leaves. Since, the number of iterations (and leaves) is at most T2D-1, to prove event E happens with probability 1 - p, by the union bound it suffices to prove that $\mathbb{P}[\|\nabla t,s - \nabla F(w_t,s; D)\|2$
> \alpha \cdot \alphã] \leq p
T 2D-1 for every (t, s) \preceq2 (T, S) where u_t,s is a leaf. Denote by Ft the sigma algebra generated by randomness in the algorithm until the end of round t. Fix (t, s) \preceq2 (T, S) $such that u_t,s is leaf, and let u_t,s\emptyset$ = u_t,s0, u_t,s1,..., u_t,sk = u_t,s be the path from the root to s. Next, extract a sub-sequence of it including only the root and the nodes that are right children, obtaining u_t,s\emptyset = u_t,sa0, u_t,sa1,..., u_t,sam = u_t,s. Now we can write $\nabla t,s - \nabla F(w_t,s; D) =$ m X i=0 g_t,sai + X x\inSt,\emptyset b $(\nabla f(w_t,\emptyset; x) - \nabla F(w_t,\emptyset; D))$
| {z }
$\gamma$1,x + m X i=1 X x\inSt,sai 2|sai
|
b h \nabla f(w_t,sai $; x)-\nabla f(w_t,sai-1$; x)

-

\nabla F(w_t,sai $; D)-\nabla F(w_t,sai-1$; D) i
| {z }
$\gamma$2,x,i. To bound the estimation error, we note that $\mathbb{P}[\|\nabla t,s - \nabla F(w_t,s; D)\|2$
> \alpha \cdot \alphã|Ft-1]
\leq P h m X i=0 g_t,sai
>
\alpha \cdot \alphã Ft-1 i
+ P
h X x\inSt,\emptyset $\gamma$1,x + m X i=1 X x\inSt,sai $\gamma$2,x,i
>
\alpha \cdot \alphã Ft-1 i.

and proceed to bound each term on the right hand side separately. By vector subgaussian concentration (see Lemma 1 in (Jin et al., 2019)) and noting that the gaussians are independent of Ft-1, we know that P   m X i=0 g_t,sai
>
\alpha \cdot \alphã   \leq 4d exp - \alpha \cdot \alphã 32($\sigma$^2 t,\emptyset + Pm i=1 $\sigma$^2 t,sai ) !, and in order to bound this probability by p 2T 2D-1, since m \leq D, it suffices that \alpha \cdot \alphã > 32 log

4d T2D p

8L2 0 log (1.25/$\delta$) b_2\varepsilon^2 + 8D2D \beta^2 log (1.25/$\delta$) b_2\varepsilon^2

= 256 log

1.25
$\delta$

d log (4) + log

T2D p

L^2 b_2\varepsilon^2 + D2D \beta^2 b_2\varepsilon^2

. Now, noting that surely $\|\gamma1,x\| \leq$ 2L0 b $and \|\gamma2,x,i\| \leq$ 2\beta^2D/2 b, where the second bound comes from following similar steps as in the privacy analysis in Theorem 3.1, we have that P x\inSt,\emptyset $\gamma$1,x + Pm i=1 P x\inSt,sai $\gamma$2,x,i is a sum of bounded martingale differences when conditioned on Ft-1, thus by concentration of martingale-difference sequences in \ell2 (see Proposition 2 in (Fang et al., 2018)), and using the fact that
|S_t,\emptyset| = b and |S_t,sai
| = b/2|sai
|
it follows that P    X x\inSt,\emptyset $\gamma$1,x + m X i=1 X x\inSt,sai $\gamma$2,x,i
>
\alpha \cdot \alphã
| Ft-1
   \leq 4 exp \left( - \alpha \cdot \alphã h 4L2 b + Pm i=1 4\beta^22D 2|sai
|
b i \right). Repeating a similar argument as before, to bound this term by p 2T 2D-1, it suffices that \alpha \cdot \alphã \geq 64 log

2T2D+1 p

L^2 b + \beta^2 D2D b

. Finally, both conditions hold simultaneously for \alpha^2 \geq

L^2 b + \beta^2 D2D b

max

1, (d + 1) b\varepsilon^2

and \alphã \geq 256 log

1.25
$\delta$

log

2T2D+1 p

\alpha. **Lemma C.2 (Descent lemma; Lemma 7 in (Fang et al., 2018)). Under the assumption that the event E from Lemma C.1** occurs and \beta \leq 2D/2 \alphã, we have that if Algorithm 1 reaches the last line, then $F(wT,\ell(2D); D) - F(0; D) \leq -(T2D-1$ ) \beta \cdot \alphã 4 \cdot 2D/2L1. where wT,\ell(2D) is the last iterate in the T-th tree of Algorithm 1. We provide the proof of Lemma C.2 adapted to our case for completeness.

Proof. By standard analysis for smooth functions we have $F(w_t,s+; D) \leq F(w_t,s; D) -$ \etat,s $(1 - \etat,sL1)\|\nabla t,s\|2$ + \etat,s $\|\nabla t,s - \nabla F(w_t,s; D)\|2$, where \etat,s = \beta $2D/2L1\|\nabla t,s\|$ and u_t,s+ is the node after u_t,s in the tree. Since \beta \leq 2D/2 $\alphã and \|\nabla t,s\| > 2\alphã, we have that$ (1 - \etat,sL1) \geq 1/2. Using this inequality, the definition of \etat,s and the fact that we are assuming E occurs, we obtain $F(w_t,s+; D) - F(w_t,s; D) \leq -$ \beta $4 \cdot 2D/2L1\|\nabla t,s\|$ \|\nabla t,s\|2 + \beta $2 \cdot 2D/2L1\|\nabla t,s\|$ \alpha \cdot \alphã \leq - \beta 4 \cdot 2D/2L1 \cdot \alphã, where the second inequality comes from \|\nabla t,s\| > 2\alphã and \alpha \leq \alphã. Then telescoping over all T2D-1 iterations provides the claimed bound. We are now ready to prove the convergence guarantee of Algorithm 1. Proof of Theorem 3.2. From Lemma C.1, we know that \|\nabla t,s - \nabla F(w_t,s; D)\|2 \leq \alpha \cdot \alphã with probability 1 - p when \alpha = \sqrt 2L0 max

n^{1/3}, \sqrt{d} n\varepsilon 1/2

, \alphã =

256 log 1.25 $\delta$

log

2T 2D+1 p

+ 8L1F0
\sqrt 2D(D/2+1) 2L2

\alpha. Indeed, using our parameter setting, and noting that d > b\varepsilon^2 $if and only if, d > n^{2/3}$ \varepsilon^2, yields \alpha^2 \geq L^2 b max

1, (d + 1) b\varepsilon^2

+ \beta^2 max

1, (d + 1) b\varepsilon^2

= L^2 n^{2/3} $1{d+1\leqn2/3\varepsilon^2} +$ \sqrt{d} n\varepsilon $1{d+1>n^{2/3}\varepsilon^2}$ ! + \alpha^2 min

1, b\varepsilon^2 d

max

1, (d + 1) b\varepsilon^2

\geq L^2 0 max ( n^{2/3}, \sqrt{d} n\varepsilon ) + \alpha^2, which shows our values of \alpha and \alphã are valid for controlling the gradient estimation error with high probability, as claimed in **Lemma C.1.** Now, suppose for the sake of contradiction that Algorithm 1 does not end in line 20 under E. This means it performs T2D-1 gradient updates. We'll show this implies (T2D-1 ) \beta\cdot\alphã 4\cdot2D/2L1
> F_0 and thus contradicts Lemma C.2, which claims that
$F_0 \geq -[F(wT,\ell(2D); D) - F(w_0,\ell(2D); D)] \geq (T2D-1$ ) \beta\cdot\alphã 4\cdot2D/2L1. Indeed, note that by our parameter setting: (T2D-1 ) \beta \cdot \alphã 4 \cdot 2D/2L1
> F_0 ⇐\Rightarrow \beta \cdot \alphã >
8L1F0 T2D/2 ⇐\Rightarrow \alpha min ( 1, \sqrt b$\varepsilon$ \sqrt{d} ) \cdot \alphã > 8L1F0 \sqrt 2D T \sqrt b ⇐\Rightarrow \alpha \cdot \alphã > 8L1F0 \sqrt 2D(D/2 + 1) \sqrt b n max ( 1, \sqrt{d} \sqrt b$\varepsilon$ ) ⇐\Rightarrow \alpha \cdot \alphã > 8L1F0 \sqrt 2D(D/2 + 1) max (\sqrt b n, \sqrt{d} n\varepsilon ),

and noting that by the setting of \widehat{w}e have max n\sqrt b n, \sqrt{d} n\varepsilon o = max n n^{2/3}, \sqrt{d} n\varepsilon o, we conclude the following (T2D-1 ) \beta \cdot \alphã 4 \cdot 2D/2L1
> F_0 ⇐\Rightarrow \alpha \cdot \alphã > 8L1F0
\sqrt 2D(D/2 + 1) max ( n^{2/3}, \sqrt{d} n\varepsilon ) ⇐\Rightarrow \alpha \cdot \alphã > 8L1F0 \sqrt 2D(D/2 + 1) 2L2 \alpha^2. Finally, note \alpha \cdot \alphã =

256 log (1.25/$\delta$) log 2T2D+1 /p

+ 8L1F0
\sqrt 2D(D/2+1) 2L2

\alpha^2 and thus the last inequality holds under our parameter setting. Since this is equivalent to (T2D-1 ) \beta\cdot\alphã 4\cdot2D/2L1
> F_0, we are done with the contradiction. It follows
that with high probability, Algorithm 1 ends in line 20 returning w_t,s such that \|\nabla t,s\| \leq 2\alphã. Also, by Lemma C.1 we have $\|\nabla F(w_t,s; D) - \nabla t,s\| < \alphã, so the returned iterate satisfies by the triangle inequality$ \|\nabla F(w_t,s; D)\| < 3\alphã. In addition, the linear time oracle complexity follows from the fact that at each binary tree we use b samples at the root, and then b/2 in levels 1 to D. This gives a total of b(D/2 + 1) samples used at every round. Since we run the algorithm for T = n $b(D/2+1) rounds, we compute exactly n gradients. To conclude, note the condition n \geq max{$ \sqrt{d}(D/2+1)2 $/\varepsilon, (D/2+1)3$ } implies the number of rounds T is at least 1. Besides, since the definition of D implies 2D < b, the size of the mini-batches are well-defined (meaning Algorithm 1 uses batches with at least 1 sample). This concludes the proof.
## D. Missing Results for Stationary Points in the Convex Setting
We first give pseudo-codes of algorithms used in the section.
### Algorithm 5. Phased SGD(S, (w, x) \to f(w; x)), \mathbb{R}, \eta, S(\cdot), \sigma)
Input: Dataset S, loss function f(\cdot; x)), radius \mathbb{R} of the constraint set W, steps T, \eta, Selection function S, Noise variance
$\sigma$
1: w_1 = 0
2: K = \lceillog (|S|)\rceil and T0 = 1
3: for k = 1 to K - 1 do
4: Tk = 2-k
|S| , \etak = 4-k
\eta, \sigmak = \etak$\sigma$ $5: w_k+1 = OutputPerturbedSGD(w_k, STk-1+1:Tk$, \mathbb{R}, \etak, \sigmak, S(\cdot))
6: end for
Output: \bar{w} = wK
### Algorithm 6. OutputPerturbedSGD(w_1, S, (w, x) \to f(w; x), \Delta(\cdot), \mathbb{R}, \eta, S(\cdot)
Input: Dataset S, loss function f(\cdot; x)), regularizer \Delta(\cdot), radius \mathbb{R} of the constraint set W, steps T, \eta, Selection function
S, Noise variance $\sigma$
1: for t = 1 to |S| - 1 do
$2: w_t+1 = ΠW (w_t - \eta (\nabla f(w_t; xt)))$
3: end for
$4: \xi \sim \mathcal{N}(0, \sigma^2$ I)
5: w̃ = S

{w_t}
|S|
t=1

Output: \bar{w} = w̃ + \xi
Proof of Theorem 5.1. The privacy guarantee, in both cases, follows from the privacy guarantees of Algorithm 7 and Algorithm 5, in Lemmas D.3 and D.6 respectively, together with parallel composition.

### Algorithm 7. Noisy GD(S, (w, x) \to f(w; x)), \mathbb{R}, T, \eta, S(\cdot), \sigma)
Input: Dataset S, loss function (w, x) \to f(w; x), radius \mathbb{R} of the constraint set W, steps T, \eta, Selection function S, Noise
variance $\sigma$
1: w_1 = 0
2: for t = 1 to T - 1 do
$3: \xit \sim \mathcal{N}(0, \sigma^2$ I)
4: w_t+1 = ΠW (w_t - \eta (\nabla F(w_t; S) + \xit))
5: end for
Output: \bar{w} = S

{w_t} T t=1

We now proceed to the utility part. For simplicity of notation, let \mathbb{R} = \|w^\ast \|. Recall the definition of the regularized losses f(t) $(w, x) in Algorithm 3. Let {\alphat}t be such that \mathbb{E}[F(t-1)$ (\bar{w}t; D)] - F(t-1) (w^\ast $t-1; D) \leq \alphat where \bar{w}t are the iterates$ produced in the algorithm and w^\ast t-1 = arg minw\inRd F(t-1) (w; D). Following (Allen-Zhu, 2018; Foster et al., 2019), we first establish a general result which will be useful for both parts of the result.
$$
E \|\nabla F(\bar{w}T; D)\| = E \nabla F(T -1)
$$
(\bar{w}T; D) + \lambda T X t=0 2t (\bar{w}t - \bar{w}T ) \leq E \nabla F(T -1) $(\bar{w}T; D) + \lambda$ T -1 X t=0 2t $E \bar{w}t - w^\ast$ T -1 + \bar{w}T - w^\ast T -1

\leq 2E \nabla F(T -1) $(\bar{w}T; D) + \lambda$ T -1 X t=1 2t $E \bar{w}t - w^\ast$ T -1 + \lambdaE w_0 - w^\ast T -1 \leq 2E \nabla F(T -1) (\bar{w}T; D) + 4 T -1 X t=1 p $\lambda$2t\alphat + \lambdaRT -1 \leq 4 p L_1\alphaT + 4 T -1 X t=1 p $\lambda2t+1\alphat + \lambda2T/2$ \mathbb{R} \leq 4 T X t=1 p $\lambda$2t+1\alphat + p \lambdaL1R where the third and fourth inequality follows from strong convexity of F(T -1) (\cdot; D) and Lemma D.2 respectively. The last inequality follows from the setting of T since we have that F(T -1) is L_1 + PT -1 t=1 2t $\lambda$ \leq L_1 + $\lambda$2T \leq 2L1 smooth. Note that the definition of R_t and Lemma D.1, w^\ast T -1 \leq RT -1, so the unconstrained minimizer lies in the constraint set. Therefore E \nabla F(T -1) $(\bar{w}T; D) = E \nabla F(T -1)$ (\bar{w}T; D) - \nabla F(T -1)
$$
(w^\ast T -1; D) \leq 2 \sqrt L_1\alphaT. Observe that from the setting of T, F(T ) is 4L1 smooth for all t. Furthermore, the radius of the constraint set in the t-th round is R_t = 2T/2 Hence, the Lipschitz constant G_t \leq L_0 + 8L1Rt \leq O L_0 + L12T/2. Now we instantiate \alphat, which is the excess population risk bound of the DP-SCO sub-routine. Optimal rate: The excess population risk guarantee of Algorithm 7 is in Lemma D.3, with (in context of the notation in the Lemma) Lipschitz parameter L_0 being the same and G\Delta = O L12T/2
$$
. Therefore, we have \alphat = \widetilde{O}

G^2 \lambdatn + dG2 \lambdatn2\varepsilon^2

. Plugging in the above estimate, we get, $E \|\nabla F(\bar{w}; D)\| = \widetilde{O}$ G \sqrt{n} + \sqrt{d}G n\varepsilon + r $\lambda$ L_1 \mathbb{R} ! = \widetilde{O} G \sqrt{n} + \sqrt{d}G n\varepsilon ! where the last step follows by setting of $\lambda$. The optimality claim follows by combining the non-private lower bound in Theorem 5.1, and the DP empirical stationarity lower bound in Theorem 4.3 together with a reduction to population stationarity as in (Bassily et al., 2019, Appendix C).

Linear time rate: The excess population risk guarantee of Algorithm 5 is in Lemma D.6, with Lipschitz parameter L_0 being the same and G\Delta = O L12T/2

. This gives us \alphat = \widetilde{O}

L^2 \lambdatn + dL2 \lambdatn2\varepsilon^2

, and thus $E \|\nabla F(\bar{w}; D)\| = \widetilde{O}$ L_0 \sqrt{n} + \sqrt{d}L0 n\varepsilon + p \lambdaL1R ! = \widetilde{O} L_0 \sqrt{n} + \sqrt{d}L0 n\varepsilon + L1R \sqrt{n} ! where the last step follows by setting of $\lambda$. Finally, note that the Lemma D.6 requires that n = \Omegã

L_1+\lambda_t \lambda_t

for all t. This can be checked to be satisfied by substituting the value of \lambda_t.
### D.1. Utility Lemmas
We first present some key results which will be useful in the proofs. **Lemma D.1. Let f: \mathbb{R}^d** $\to \mathbb{R} be an L_1-smooth convex function and let w^\ast$ = arg minw\inRd f(w). Let \mathbb{R} = \|w^\ast $\| and$ w_0 \in \mathbb{R}^d $such that \|w_0\| \leq R. Define ˜$ f(w) = f(w) + \lambda $2 \|w - w_0\|$ and let w̃ = arg min ˜ f(w). Then for any $\lambda$ \geq 0, \|w̃\| \leq \sqrt 2R. Proof. From optimality criterion, 0 = \nabla ˜ $f(w̃) = \nabla f(w̃) + \lambda (w̃ - w_0). Therefore, \nabla f(w̃) = \lambda (w_0 - w̃) and thus$ \langle \nabla f(w̃), w_0 - w̃ \rangle > 0. Furthermore, since f is convex, from monotonicity, \langle \nabla f(w̃), w^\ast $- w̃ \rangle \leq 0. Since both w_0 and w^\ast$ lie in the ball of radius \mathbb{R} (say WR), the above two implies that the hyperplane H = {w: \langle \nabla f(w̃), w - w̃ \rangle = 0} intersects with WR. Furthermore, since \nabla f(w̃) = $\lambda$ (w_0 - w̃), we have that w̃ is the projection of w_0 on H i.e. ΠH(w_0). Let w' = ΠH(0). We have that w' \in WR; this is because the hyperplane cuts the hypersphere WR creating a spherical cap and w' is the center of the cap. From properties of convex projections \|ΠH(w_0) - ΠH(0)\| \leq \|w_0 - 0\| \leq R. Furthermore, $ΠH(0) and ΠH(w_0) - ΠH(0) are orthogonal. Hence \|w̃\|$ = \|ΠH(w_0)\| $= \|ΠH(0)\|$
+ \|ΠH(w_0) - ΠH(0)\|
\leq 2R2. We state the following result from (Allen-Zhu, 2018; Foster et al., 2019). **Lemma D.2. Suppose for every t = 1, 2,... T, \mathbb{E}[F(t-1)** (\bar{w}t; D)] - F(t-1) (w^\ast t-1; D) \leq \alphat where \bar{w}t are the iterates produced in the algorithm, w^\ast t-1 = arg minw\inRd F(t-1) (w; D) and \lambda_t = 2t $\lambda$, we have, $1. For every t \geq 1, \mathbb{E}[ \bar{w}t - w^\ast$ t-1 ] \leq 2\alphat \lambda_t-1 $2. For every t \geq 1, \mathbb{E}[\|\bar{w}t - w^\ast$ t \| ] \leq \alphat \lambda_t
3. \mathbb{E}[
PT $t=1 \lambda_t \|\bar{w}t - w^\ast$ T \|] \leq 4 PT t=1 \sqrt \alphat\lambda_t
### D.2. Lemmas for NoisyGD (Algorithm 7)
**Lemma D.3. Consider a function f(w; x) = \ell(w; x) + \Delta(w), where w \to \ell(w; x) is convex and L_0 Lipschitz for** all x, and \Delta(w) is $\lambda$ strongly convex, G\Delta Lipschitz and H\Delta smooth over a bounded convex set W. Algorithm 6 run with parameters \eta = log(T ) \lambdaT, $\sigma$^2 = 64L2 0T log(1/$\delta$) n^2\varepsilon^2, T = max

L_1+H\Delta $\lambda$ log L_1+H\Delta $\lambda$

, n^2 \varepsilon^2 (L^2 0+G^2 \Delta) dL2 0 log(1/$\delta$)

$and S({w_t}t) =$ PT $t=1(1-\eta\lambda)-t$ PT t=1 (1 - \eta$\lambda$) -t w_t satisfies $($\varepsilon$, $\delta$)$-DP and given a dataset S of n i.i.d. points from D, the excess population risk of its output \bar{w} is bounded by, E

F(\bar{w}; D) - min w\inWR F(w; D)

= O

L^2 \lambdan + dL2 0 log (1/$\delta$) \lambdan2\varepsilon^2

. Proof. For the privacy analysis, as in (Bassily et al., 2014), for fixed w, the sensitivity of the gradient update is bounded by 2L0 n. Applying advanced composition, we have that $\sigma$^2 = 64L2 0T log(1/$\delta$) $n^2\varepsilon^2 suffices for (\varepsilon, \delta)-DP.$ For utility, we first compute a bound on uniform argument stability of the algorithm; let {w_t} and {w' t} be sequence of iterates on neighbouring datasets. Note that the function w \to f(w; x) is L_1 + H\Delta-smooth and $\lambda$-strongly convex for all x. From the setting of T, we have that the step size \eta \leq 1 L_1+H\Delta, hence from the standard stability analysis,

w_t+1 - w' t+1 = w_t - \eta\nabla L(w_t; S) - \eta\nabla \Delta(w_t) - w' t + \eta\nabla L(w' t; S' ) + \eta\nabla \Delta(w' t) = w_t - w' t - \eta (\nabla L(w_t; S) + \nabla \Delta(w_t) - \nabla L(w' $t; S) - \eta\nabla \Delta(w'$ t))
+ \eta (\nabla L(w'
t; S' ) - \nabla L(w' t; S)) = I - \eta \nabla 2 L(w̃t; S) + \nabla 2 \Delta(w̃t)

(w_t - w' t)
+ \eta (\nabla L(w'
t; S' ) - \nabla L(w' t; S)) where the last equality follows from Taylor remainder theorem where w̃t is some intermediate point on the line joining w_t and w' t. Using the fact that \eta \leq 1 L_1+H\Delta, we have w_t+1 - w' $t+1 \leq (1 - \eta\lambda) \|w_t - w'$ t\| + 2\etaL0 n \leq 2L0 \lambdan The above gives the same bound for the iterate using the selector S, $\|S({w_t}) - S({w'$ t})\| \leq 2L0 \lambdan Note that the overall Lipschitz constant for the empirical loss is ˜ $L_0 = L_0 + G\Delta. For the excess empirical risk guarantee, we$ use Lemma 5.2 in (Feldman et al., 2020) to get, $E [L (\bar{w}; S) + \Delta(\bar{w}) - L(w^\ast$; S) - \Delta(w^\ast $)] = E [F (\bar{w}; S) - F(w^\ast$; S)] = \widetilde{O} ˜ L_0 \lambdaT ! = \widetilde{O} ˜ L_0
+ $\sigma$^2
d \lambdaT ! = \widetilde{O} ˜ L_0 \lambdaT + dL2 0 log (1/$\delta$) \lambdan2\varepsilon^2 ! = O

dL2 0 log (1/$\delta$) \lambdan2\varepsilon^2

where the last step follows from the setting of T. For the population risk guarantee, we have, $E [F(\bar{w}; D) - F(w^\ast$; D)] = E [F(\bar{w}; D) - F(\bar{w}; S)] + E [F(\bar{w}; D) - F(w^\ast )] $= \mathbb{E}[L(\bar{w}; D) - L(\bar{w}; S)] + O$

dL2 0 log (1/$\delta$) \lambdan2\varepsilon^2

$\leq L0E \|\bar{w} - \bar{w}'$ \| + O

dL2 0 log (1/$\delta$) \lambdan2\varepsilon^2

= \widetilde{O}

L^2 \lambdan + dL2 0 log (1/$\delta$) \lambdan2\varepsilon^2

where the inequality follows from Lipschitzness and standard generalization gap to stability argument.
### D.3. Lemmas for PhasedSGD (Algorithm 5)
The following lemma gives population risk guarantees for strongly convex functions under privacy, in terms of variance of stochastic gradients, as opposed to standard Lipschitzness bounds. **Lemma D.4 (Variance based bound for constant step-size SGD for strongly-convex functions). Consider a function f(w; x) such that w \to f(w; x) is $\lambda$ strongly convex, L_1 smooth over a convex set W for all x and let**

$Ex \|\nabla f(w; x) - Ex\nabla f(w; x)\|$ \leq V^2 $for all w \in W. Let \gammat = (1 - \eta\lambda)$ -t. Given a dataset S = {x_1, x_2,..., xn} sampled i.i.d from D and \eta \leq 1 2\beta as input, for any w \in W, the iterates of Algorithm 6 satisfy E " Pn t=1 \gammat n X t=1 \gammatF(w_t; D)
#
- F(w) \leq
$\lambda$ e\eta\lambdan - 1 $\|w_0 - w\|$
+ \etaV2
Furthermore, for n = \Omega L_1 $\lambda$ log L_1 $\lambda$

, with \eta = log(n) $\lambdan and S({w_t}t) = 1$ Pn t=1 \gammat Pn t=1 \gammatwt, the excess population risk of $w̃ = S({w_t}t) satisfies$ E

F(w̃; D) - min w\inW F(w; D)

= O

V^2 log (n) \lambdan

Proof. An equivalent way to write the update in Algorithm 6 is w_t+1 = arg min w\inW

$\langle \nabla f(w_t, xt), w \rangle +$ \eta $\|w_t - w\|$
+ ψ(w)

where ψ(w) = 0 if w \in W, otherwise \infty. Following standard arguments in convex optimization, for any w \in W, we have $F(w_t+1; D) - F(w)$ = F(w_t+1; D) + ψ(w_t+1) - F(w; D) - ψ(w) $\leq F(w_t) + \langle \nabla F(w_t), w_t+1 - w_t \rangle +$ L_1 $\|w_t+1 - w_t\|$
+ ψ(w_t+1)
+ F(w; D) - ψ(w)
\leq \langle \nabla F(w_t), w_t+1 - w_t \rangle + \langle \nabla F(w_t), w_t - w \rangle - $\lambda$ $\|w_t - w\|$ + L_1 $\|w_t+1 - w_t\|$
+ ψ(w_t+1) + F(w; D) - ψ(w)
= Ezt

\langle \nabla p(w_t; zt) - \nabla F(w; D), w_t - w_t+1 \rangle + L_1 $\|w_t+1 - w_t\|$
+ \langle \nabla p(w_t; zt), w_t - w \rangle

- $\lambda$ $\|w_t - w\|$
+ ψ(w_t+1) + F(w; D) - ψ(w)
$\leq Ezt h \langle \nabla p(w_t; zt) - \nabla F(w; D), w_t - w_t+1 \rangle - 2\eta - L_1$ \|w_t+1 - w_t\| +

2\eta - $\lambda$

$\|w_t - w\|$ - 2\eta $\|w_t+1 - w\|$ i \leq Ezt h \eta 2 (1 - \etaL1) $\|\nabla p(w_t; zt) - \nabla F(w; D)\|$ +

2\eta - $\lambda$

$\|w_t - w\|$ - 2\eta $\|w_t+1 - w\|$ i \leq \etaV2
+ Ezt

2\eta - $\lambda$

$\|w_t - w\|$ - 2\eta $\|w_t+1 - w\|$

where the first inequality follows from smoothness, the second from strong convexity, the third from Fact D.1 in (Allen-Zhu, 2018), fourth from AM-GM inequality and the last from the assumption about variance bound on the oracle. Now, the above is exactly the bound obtained in the proof of Lemma 5.2 in (Feldman et al., 2020) with the second moment on gradient norm replaced by variance. Repeating the rest of the arguments in that Lemma gives us the claimed result. **Lemma D.5 (Privacy of Algorithm 6). Consider a function f(w; x) = \ell(w; x) + \Delta(w) such that w \to \ell(w; x) is convex,** L_0 Lipschitz, L_1-smooth for all z, and \Delta(\cdot) is $\lambda$ strongly convex, G\Delta Lipschitz and H\Delta smooth over a bounded set W. For n = \Omega L_1+H\Delta $\lambda$ log L_1+H\Delta $\lambda$

, Algorithm 6 with input as function (w, x) \to f(w; x), $\sigma$^2 = 64G2 (log(n))2 log(1/$\delta$) $\lambda$2n2\varepsilon^2, \eta = log(n) \lambdan and S ({w_t} n t=1) = 1 Pn t=1 \gammat Pn $t=1 \gammatwt for any weights \gammat satisfies (\varepsilon, \delta)-DP.$

Proof. We start with computing the sensitivity of the algorithm's output: let {w_t} and {w' t} be sequence of iterates produced by Algorithm 6 on neighbouring datasets. Note that the function w \to f(w; x) is L' $1 = L_1 + H\Delta-smooth and \lambda-strongly$ convex for all x. From the assumption on n, we have that the step size \eta \leq 1 H+H\Delta. Suppose the differing sample between neighbouring datasets is xj, then w_t = w' t for all t \leq j. Also, wj+1 - w' $j+1 = \eta \nabla \ell(wj; xj) - \nabla \ell(wj; x'$ j) \leq 2\etaL0 = 2L0 log (n) \lambdan Now, for any t > j, as in the standard stability analysis we have, w_t+1 - w' t+1 = w_t - \eta\nabla \ell(w_t; xt) - \eta\nabla \Delta(w_t) - w_t + \eta\nabla \ell(w' $t; xt) + \eta\nabla \Delta(w'$ t) = I - \eta \nabla 2 $\ell(w̃t; xt) + \nabla 2$ \Delta(w̃t)

(w_t - w' t) where the last equality follows from Taylor remainder theorem where w̃t is some intermediate point in the line joining w_t and w' t. Using the fact that \eta \leq 1 L_1+H\Delta and $\lambda$ strong convexity, we have w_t+1 - w' $t+1 \leq (1 - \eta\lambda) \|w_t - w'$ t\| \leq wj+1 - w' j+1 \leq 2L0 log (n) \lambdan Applying convexity to the weights in the definition of the selector function S, we get, $\|S({w_t}) - S({w'$ t})\| \leq 2L0 log (n) \lambdan The privacy proof now follows from the Gaussian mechanism guarantee. **Lemma D.6 (Phased SGD composite guarantee). Consider a function f(w; x) = \ell(w; x) + \Delta(w) where w \to \ell(w; x)** is convex, L_0 Lipschitz, L_1 smooth for all x, and \Delta(w) is $\lambda$ strongly convex, G\Delta Lipschitz and H\Delta smooth over a bounded set W. For n = \Omega

K(L_1+H\Delta) $\lambda$ log L_1+H\Delta $\lambda$

, Algorithm 6 with $\sigma$^2 = 64L2 0K2 (log(n))2 log(1/$\delta$) $\lambda2n2\varepsilon^2, satisfies (\varepsilon, \delta)-$ DP. Furthermore, with input as function (w, x) \to f(w; x), a dataset S of n samples drawn i.i.d. from D, \eta = log(n) \lambdan, $K = ln ln n, \gammat = (1 - \eta\lambda)$ -t and S ({w_t} n t=1) = 1 Pn t=1 \gammat Pn t=1 \gammatwt, the excess population risk of output wK is bounded as E [F(wK; D)] - min w\inW $F(w; D) = \widetilde{O}$

L^2 \lambdan + dL2 \lambdan2\varepsilon^2

Proof. The privacy proof simply follows from parallel composition. For the utility proof, we repeat the arguments in **Theorem 5.3 in (Feldman et al., 2020) substituting the variance-based bound from Lemma D.4. Note that the variance of the** stochastic gradients used, V^2 \leq L^2 0, this gives us, E [F(wK; D)] - min w\inW $F(w; D) = \widetilde{O}$

L^2 \lambdan + dL2 \lambdan2\varepsilon^2

## E. Missing Results for Generalized Linear Models
We first give the definition of oblivious subspace embedding. **Definition E.1 ((r, $\tau$, \beta)-oblivious subspace embedding). A random matrix \Phi \in \mathbb{R}^k\timesd** $is an (r, \tau, \beta)-oblivious subspace$ embedding if for any r dimensional linear subspace in \mathbb{R}^d, say V, we have that with probability at least 1 - \beta, for all x \in V, $(1 - \tau) \|x\|$ \leq \|\Phix\| $\leq (1 + \tau) \|x\|$

It is well-known that JL matrices with embedding dimension k = O

r log(2/\beta) $\tau$^2

$are (r, \tau, \beta)-oblivious subspace embeddings$ and can be constructed efficiently (Cohen, 2016). A simple example is a scaled Gaussian random matrix, \Phi = 1 \sqrt k G where entries of G are independent and distributed as \mathcal{N}(0, 1). Proof of Theorem 6.1. We first prove privacy. Let G(S) and H(S) be the bounds on the Lipschitz and smoothness constants of the family of loss functions {w \to f(w; \Phix)}x\inS. With k = \Omega(log (2n/$\delta$)), from the JL-property, it follows that with probability at least 1 - $\delta$/2, G(S) \leq 2L0 \|X\| and H(S) \leq 2L1 \|X\| $. Hence, using the fact that A is (\varepsilon, \delta/2)-DP, we$ have that Algorithm 4 is (\varepsilon, \delta)-DP. We now proceed to the utility part. Let w̃ \in \mathbb{R}^k be the output of the base algorithm in low dimensions. Note that the final output is \bar{w} = \Phi^\top w̃. The transpose of the JL matrix can only increase the norm by the polynomial factor of d and n, hence \|\bar{w}\| \leq \operatorname{poly}(n, d) \|w̃\|. By assumption, P (\|w̃\| > \operatorname{poly}(n, d, L_0, L_1)) \leq 1 \sqrt{n}. Hence we also have that $P (\|\bar{w}\| > \operatorname{poly}(n, d, L_0, L_1)) \leq 1$ \sqrt{n}. Let W \subseteq \mathbb{R}^d $denote the above set with radius \operatorname{poly}(n, d, L_0, L_1).$ We now decompose the population stationarity as, E \|\nabla F(\bar{w}; D)\| \leq E \|\nabla F(\bar{w}; D) - \nabla F(\bar{w}; S)\| + \|\nabla F(\bar{w}; S)\| \leq E sup w\inW $\|\nabla F(w; D) - \nabla F(w; S)\| +$ L_0 \|X\| \sqrt{n} $+ E \|\nabla F(\bar{w}; S)\|, (7)$ where the last inequality follows from the above reasoning that that P (\bar{w} \in W) \geq 1 - 1 \sqrt{n}. The first term is bounded from uniform convergence guarantee in Lemma E.2 noting that the dependence on \|W\| in the Lemma is only \operatorname{poly}-logarithmic. E sup w\inW $\|\nabla F(w; D) - \nabla F(w; S)\| = \widetilde{O}$

L_0 \|X\| \sqrt{n}

(8) We now prove a bound on the empirical stationarity. Note that it suffices to prove a high-probability (over the random JL matrix) bound because the norm of gradient is bounded in worst case by L_0 \|X\|. Thus the expected norm of gradient of the output is bounded by the high probability bound by considering a small enough failure probability. From the assumption on A, with probability at least 1 - $\delta$/2, $\|\nabla F(w̃; \PhiS)\| = E$ n n X i=1 \phi' y_i (\langle w̃, \Phi x_i \rangle)\Phi x_i \leq g(k, n, 2L0 \|X\|, 2L0 \|X\|, $\varepsilon$, $\delta$/2) We now use the fact that if k = O (\operatorname{rank} log (2n/$\delta$)), then the JL transform is an (\operatorname{rank}, 1/2, $\delta$/2) oblivious subspace embedding (see Definition E.1). Thus, it approximates the norm of any vector in \operatorname{span}({x_i} n i=1), and hence any gradient. Therefore, $E \|\nabla F(w̃; \PhiS)\| = E \Phi$ n n X i=1 \phi' y_i $(\langle w̃, \Phi x_i \rangle)x_i$ ! \geq 1 - r \operatorname{rank} k ! E n n X i=1 \phi' y_i $(\langle w̃, \Phi x_i \rangle)x_i$ \geq E n n X i=1 \phi' y_i $(\langle w̃, \Phi x_i \rangle)x_i =$ E n n X i=1 \phi' y_i ( \Phi^\top w̃, x_i )x_i = $E \|\nabla F(\bar{w}; S)\|$ Thus with k = O (\operatorname{rank} log (2n/\delta)), we get $E \|\nabla F(\bar{w}; S)\| \leq g(k, n, 2L0 \|X\|, 2L1 \|X\|$, \varepsilon, \delta) = g(\operatorname{rank}, n, 2L0 \|X\|, 2L1 \|X\|, $\varepsilon$, $\delta$) For the other bound, let I_d-k \in \mathbb{R}^d\timesk denote the matrix with first k diagonal entries, (I_d-k)j,j with j \in [k], are 1 and the

rest of the matrix is zero. We have, E \|\nabla F(\bar{w}; S)\| = E n n X i=1 \phi' y_i ( \Phi^\top w̃, x_i )x_i \leq E n n X i=1 \phi' y_i $(\langle w̃, \Phi x_i \rangle)I_d-k\Phi x_i + E$ " n n X i=1 \phi' y_i $(\langle w̃, \Phi x_i \rangle)x_i -$ n n X i=1 \phi' y_i $(\langle w̃, \Phi x_i \rangle)I_d-k\Phi x_i$
#
$\leq E \|I_d-k\|$ n n X i=1 \phi' y_i $(\langle w̃, \Phi x_i \rangle)\Phi x_i +$ n E n X i=1 \phi' y_i $(\langle w̃, \Phi x_i \rangle) |\|x_i - I_d-k\Phi x_i\||$ \leq E \|\nabla F(w̃; \PhiS)\| + n E n X i=1 $L_0 \|I - I_d-k\Phi\| \|x_i\|$ \leq g(k, n, 2L0 \|X\|, 2L1 \|X\| $, \varepsilon, \delta/2) + L_0 \|X\| E \|I - H\|$ where the second inequality follows from triangle inequality, the third inequality follows from L_0-Lipschitzness of the GLM, the third inequality follows from the accuracy guarantee of the base algorithm and substituting H = I_d-k\Phi. To bound E \|I - H\|, we use concentration properties of distribution used in the construction of JL matrices. Specifically, using the scaled Gaussian matrix construction, from concentration of extreme eignevalues of square Gaussian matrices, we have that $E \|I - H\| = \widetilde{O}$

\sqrt k

(Rudelson & Vershynin, 2010). This gives us, E \|\nabla F(\bar{w}; S)\| \leq g(k, n, 2L0 \|X\|, 2L1 \|X\| $, \varepsilon, \delta/2) + \widetilde{O}$

L_0 \|X\| \sqrt k

Choosing k to minimize the above yields the bound of \widetilde{O}

L_0\|X\| \sqrt k

. Combining the two cases, yields the bound of $g(k, n, 2L0 \|X\|, 2L1 \|X\|$, $\varepsilon$, $\delta$/2) on gradient norm. Plugging this and the bound in Eqn. (8) in Inequality (7) gives the claimed bound. **Lemma E.2. Let D be a probability distribution over X such that \|x\| \leq \|X\| for all x \in \operatorname{supp}(D). Let f(w; (x, y)) =** \phi_y (\langle w, x \rangle) be an L_1-smooth L_0-Lipschitz GLM. Then, with probability at least 1 - \beta, over a draw of n i.i.d. samples S from D, we have sup w\inW $\|\nabla F(w; D) - \nabla F(w; S)\| \leq$ 4L0 \|X\| log 2n3/2 $\|W\| L_1 \|X\| /L_0$

\sqrt{n} + 4L0 \|X\| p log (1/\beta) \sqrt{n} Proof. We first give a bound on the expected uniform deviation, ES\simDn supw\inW \|\nabla F(w; D) - \nabla F(w; S)\|. The gradient $of the loss function is \nabla f(w; x) = \phi'$ x (\langle w, x \rangle) x. We start with the standard symmetrization trick, ES\simDn sup w\inW $\|\nabla F(w; D) - \nabla F(w; S)\|$ = ES\simDn sup w\inW E\phi' y (\langle w, x \rangle) x - n n X i=1 \phi' x_i $(\langle w, x_i \rangle) x_i$ = ES\simDn sup w\inW E{x' i}\simDn n n X i=1 \phi' y' i (\langle w, x' i \rangle) x' i - n n X i=1 \phi' x_i $(\langle w, x_i \rangle) x_i$ \leq ES,S'\simDn sup w\inW n n X i=1 \phi' y' i (\langle w, x' i \rangle) x' i - n n X i=1 \phi' x_i $(\langle w, x_i \rangle) x_i$ = ES,S'\simDn E{\sigmai} sup w\inW n n X i=1 \sigmai

\phi' y' i (\langle w, x' i \rangle) x' i - \phi' x_i $(\langle w, x_i \rangle) x_i$

\leq 2ES\simDn E{\sigmai} sup w\inW n n X i=1 \sigmai\phi' y_i $(\langle w, x_i \rangle) x_i (9)$

where \sigmai are i.i.d. Rademacher random variables. For fixed {x_i} n i=1, consider a set W0 s.t. for all w \in W and i \in [n], there exists w_0 \in W0 such that |\langle w, x_i \rangle - \langle w_0, x_i \rangle| \leq $\tau$. Since \|w\| \leq \|W\| and \|x_i\| \leq \|X\|, we require only 2n\|W\|\|X\| $\tau$ points in W0 to satisfy the above covering condition. Therefore, ES\simDn E{\sigmai} sup w\inW n n X i=1 \sigmai\phi' y_i $(\langle w, x_i \rangle) x_i$ = ES\simDn E{\sigmai} sup w\inW,w_0\inW0 n n X i=1 \sigmai \phi' y_i $(\langle w, x_i \rangle) - \phi'$ y_i $(\langle w_0, x_i \rangle) + \phi'$ y_i $(\langle w_0, x_i \rangle)$

x_i \leq ES\simDn E{\sigmai} sup w\inW,w_0\inW0 n n X i=1 \sigmai \phi' y_i $(\langle w, x_i \rangle) - \phi'$ y_i $(\langle w_0, x_i \rangle)$

x_i + n n X i=1 \sigmai\phi' y_i $(\langle w_0, x_i \rangle) x_i$ \leq ES\simDn E{\sigmai} sup w\inW,w_0\inW0 $L_1 |\langle w, x_i \rangle - \langle w_0, x_i \rangle| \|X\| + ES\simDn E{\sigmai} sup$ w_0\inW0 n n X i=1 \sigmai\phi' y_i $(\langle w_0, x_i \rangle) x_i$ \leq L_1\tau \|X\| + ES\simDn E{\sigmai} sup w_0\inW0 n n X i=1 \sigmai\phi' y_i $(\langle w_0, x_i \rangle) x_i (10)$ where the second last inequality follows from smoothness and the last from the definition of cover W0. For fixed w_0, from standard manipulations, we have, E{\sigmai} n n X i=1 \sigmai\phi' y_i $(\langle w_0, x_i \rangle) x_i \leq$ v u u tE{\sigmai} n n X i=1 \sigmai\phi' y_i $(\langle w_0, x_i \rangle) x_i$ = v u u t 1 n^2 E{\sigmai} n X i=1 \sigmai\phi' y_i $(\langle w_0, x_i \rangle) x_i$ \leq L_0 \|X\| \sqrt{n} Using Massart's finite class lemma to handle all w_0 \in W0, and substituting the above in Eqn. (10), we get, ES\simDn E{\sigmai} sup w\inW n n X i=1 \sigmai\phi' y_i $(\langle w, x_i \rangle) x_i \leq L_1\tau \|X\| +$ G \|X\| log (2n \|W\| \|X\| /\tau) $\sqrt{n} Choosing \tau = L_0 L_1 \sqrt{n}, we get, ES\simDn E{\sigmai} sup w\inW n n X i=1 \sigmai\phi' y_i$ (\langle w, x_i \rangle) x_i \leq $2L0 \|X\| log 2n3/2$ \|W\| L_1 \|X\| /L_0

\sqrt{n} Finally, substituting the above in Eqn. (9) gives us the following in-expectation bound. ES\simDn sup w\inW \|\nabla F(w; D) - \nabla F(w; S)\| \leq $4L0 \|X\| log 2n3/2$ \|W\| L_1 \|X\| /L_0

\sqrt{n} For the high-probability bound, let ψ(S) = supw\inW \|\nabla F(w; D) - \nabla F(w; S)\| and let w^\ast \in W achieves the supremum. We can bound the increment between neighbouring datasets S and S' as,
|ψ(S) - ψ(S'
)| \leq |\|\nabla F(w^\ast $; D) - \nabla F(w^\ast$; S)\| - \|\nabla F(w^\ast $; D) - \nabla F(w^\ast$; S' )\|| $\leq \|\nabla F(w^\ast$; S) - \nabla F(w^\ast; S' )\| \leq 2L0 \|X\| n Finally, applying McDiarmid's inequality gives the claimed bound.

Proof of Corollary 6.2. The results follow from Theorem 6.1 provided we show that the conditions on the base algorithm in the Theorem statement are satisfied. The privacy and accuracy claims follow from Theorem 3.2 and 5.1 respectively. We note that even though we are given population stationarity guarantee for the convex case, the same bound for empirical stationarity guarantee simply follows from the re-sampling argument in (Bassily et al., 2019). The only thing left to show is the high-probability bound on the trajectory of the algorithm. Non-convex setting with Private Spiderboost: From the update in Algorithm 2, we have that for any t $\|\nabla t\| \leq$ t X i=1 $\|\Deltai\| +$ t X i=1 g_t \leq 2tL0 + t X i=1 g_t where the last inequality follows from the Lipschitzness assumption. Note that g_t \sim \mathcal{N}(0, $\sigma$^2 t I) where \sigma_t \leq O (max ($\sigma$1, b $\sigma^2)) = O (\operatorname{poly}(n, d, L_0, L_1)). Hence$ Pt i=1 g_t \leq p $d log (1/\beta')O (\operatorname{poly}(n, d, L_0, L_1)) with probability at least 1 - \beta'$. Taking a union bound over all t \in T gives us \|w_t\| \leq \operatorname{poly}(n, d, L_0, L_1, log (\operatorname{poly}(n, d)/\beta)) with probability at least 1 - \beta. Substituting \beta = 1 \sqrt{n} yields the guarantee of Theorem 6.1. Convex setting with Recursive Regularization: Since the iterates are restricted to the constraint set, the final output, with probability one, lies in the set of radius RT = 2T/2 \|w^\ast \| = O r L_1 $\lambda$ \|w^\ast \| ! = O L_1 \|w^\ast \| 3/2 n L_0 ! which completes the proof.