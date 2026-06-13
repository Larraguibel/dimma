# Differential Privacy for SGD: Overview

## Understanding DP-SGD

!!! note

    **DP-SGD Flow:**
    Generate Batches → batch forward → *per-sample* grad → *per-sample* clip → aggregate clipped grads (mean) → add noise → update

**Understanding DP-SGD (Differentially Private Stochastic Gradient Descent)** can be a daunting task. The literature often leaves critical questions unanswered, forcing readers to piece together the logic on their own. This document aims to bridge those gaps by posing the right questions first: Why does *how* you sample the batch matter for privacy? What exactly is an accountant, and why do different sampling strategies need different ones?  In case you just want the direct answers, I'll leave them [here](https://www.notion.so/35e6b45abb3a80ed9e83f2eb43df3163).

You can wrap up almost all DP-SGD methods follow a standard seven-step pipeline:

1. **Batch Generation:** Sample your data with an specific private methodology.
2. **Forward Pass:** Pass the batch through the model.
3. **Per-sample Gradients:** Calculate the gradient for *each* individual sample.
4. **Clipping:** Constrain the norm of each per-sample gradient to limit individual influence and bound sensitivity.
5. **Aggregation:** Sum or average the clipped gradients.
6. **Perturbation:** Add calibrated noise to the aggregated gradients.
7. **Optimization:** Update the model parameters using the privatized gradients.

The two things that sit at the center of everything are **how you sample batches** and **how you randomize gradients**. Everything else, such as the privacy accounting, the clipping strategy, the noise calibration, follows from understanding those two clearly.

## Batches and Gradients

Every differentially private stochastic gradient method protects the training data through two mechanisms applied at every step: **how batches are drawn** and **how gradients are randomized**.

### Batch Sampling

How training examples are selected at each step determines the privacy amplification by subsampling. This means, the privacy cost per step is reduced depending on how we select the training data for the current batch. If each example is included in the batch independently with probability `p = b/n`, where `b` is the expected batch size and `n` is the number of training points, we are in Poisson subsampling, which is the classic sampling strategy for DP. The intuition behind why this helps is simple: with Poisson sampling, any given individual is simply *absent* from many steps, with probability `1 - p` they are not included at all, which fundamentally limits how much information about them can leak across the training run.

In terms of privacy accounting, Poisson subsampling is the gold standard: we have very tight accountants for it, meaning we can track the privacy budget consumption precisely and with little slack. The practical problem is that the realized batch size is `Binomial(n, p)`, not a fixed number. Computers benefit from having fixed amounts of data coming in sequentially, so this is a significant implementation challenge.

The most common alternative in practice is **shuffled minibatch sampling**: partition the dataset into fixed-size batches by shuffling and iterating through epochs. This gives perfectly regular, fixed batch sizes, which is why essentially every deep learning framework does it by default. The privacy cost, however, is harder to reason about. We do not have accountants for shuffling that are as tight as those for Poisson subsampling, which means either accepting a looser privacy guarantee or making additional assumptions. We will get into privacy accounting in detail later.

### Randomizing Gradients

Randomizing gradients means adding noise to the gradient estimate at each step. The question is: how much noise should we add to guarantee an overall `(ε, δ)`-DP mechanism?

The first challenge is that deep learning models are generally not Lipschitz, so gradients can grow arbitrarily large depending on the input, the weights, and the architecture, so the *sensitivity* of the gradient is unbounded. Per-sample clipping to a global L2 norm of `C` is what artificially imposes that bound. Once every individual gradient is clipped, the sensitivity of the aggregated gradient is known and finite, and the Gaussian mechanism can be calibrated accordingly.

From there, the noise scale follows directly from the sensitivity and the privacy budget. Start with `C`. A larger `C` means fewer gradients are clipped aggressively, which preserves the direction of the gradient signal, but it also raises the sensitivity, requiring more noise to compensate. A too-small `C`, on the other hand, clips gradients heavily, distorting their direction. Heavily clipped gradients are biased: they no longer satisfy the core SGD guarantee that the expected gradient sample tends to the true gradient, which means the optimization can drift or stall entirely, because the signal itself has been corrupted.

Then there is `ε`. The smaller `ε` is, the stronger the privacy guarantee, and the more noise must be added to the already-clipped gradients. This further degrades the gradient signal and slows learning.

Both forces act in the same direction: aggressive clipping biases the gradient, and a tight privacy budget buries what signal remains under noise. Together they define the fundamental privacy-utility tradeoff, and there is no way around it.

## Hyperparameters in DP Deep Learning

One of the most disorienting aspects of DP deep learning, coming from the standard paradigm, is that hyperparameters are far more entangled with each other. In ordinary deep learning you can tune batch size, number of epochs, and architecture somewhat independently. In DP training, almost every hyperparameter feeds into the privacy accounting, and changing one forces you to reconsider all the others.

The clipping norm `C` sets the sensitivity, which determines how much noise is needed for a given `ε`. But as we just saw, `C` is also an optimization decision: clip too little and the noise explodes, clip too much and the gradient direction is corrupted. The privacy budget `ε` and failure probability `δ` then determine how much noise must be added on top of whatever sensitivity `C` has imposed. The number of training steps determines how many times that budget is spent, and in the case of Poisson subsampling, the batch size controls the sampling probability `p = b/n`, which governs privacy amplification and therefore how much each step costs. The noise scale `σ` generally is not really a free parameter. It is the output of the accountant once all the others are fixed.

Change any one of these and the rest are affected, sometimes in non-obvious ways. Doubling the number of steps does not simply train longer: it spends more privacy budget, which forces either a larger `σ`, a different `C`, or accepting a worse `ε`. The relationships are governed by the accountant.

The one hyperparameter that sits outside this web is the **learning rate**. It does not appear in the privacy accounting at all and has no effect on `ε` or `δ`. In practice it still interacts with the noise indirectly, since a heavily noised gradient requires a different learning rate regime than a clean one, but it can be tuned without touching the privacy budget. It is the closest thing to a free parameter in DP training.

## Refs

- [Quick Answers](https://www.notion.so/35e6b45abb3a80ed9e83f2eb43df3163)
