## General

**Baselines / Scope**: (UJ59) "include comparisons with dedicated noise-robust methods (e.g., Co-teaching and DivideMix)"(yzxB) "omitted evaluation against... Tanaka et al. and Zhang et al."

- Reviewers suggested comparing SANER with semi-supervised noisy-label pipelines (e.g., DivideMix, Tanaka et al.). We clarify that SANER is an optimization method, fundamentally different from these multi-stage training pipelines that perform label correction, sample selection, or co-training.

- **Fair Comparison**. The appropriate baselines for SANER are robust optimizers, not full training pipelines. Accordingly, we provide comprehensive comparisons with SGD, SAM, ASAM, GSAM, and VaSSO. Table 4 shows that SANER consistently improves existing sharpness-aware optimizers. For example, it enhances VaSSO—an optimizer specifically designed for noisy-label scenarios—by up to 5% on CIFAR-100.

- **Relevance to Semi-supervised Methods**. Methods such as [1–3] rely on the small-loss criterion (clean samples fit earlier) to identify reliable data or generate pseudo-labels. By slowing noisy-label memorization (keeping noisy samples at higher loss) and widening the clean–noisy accuracy gap (Fig. 1b), SANER preserves and amplifies precisely the training signal these pipelines depend on. Thus, SANER functions as a robust optimization backbone that can be integrated into such methods rather than compared to them as standalone systems.

References
[1] Han et al. “Co-teaching.” NeurIPS 2018.
[2] Tanaka et al. “Joint optimization framework.” CVPR 2018.
[3] Li et al. “DivideMix.” ICLR 2020."

**Theory**: (mGrR) "setup is overly simplistic and does not extend well to... deep networks"(XUjo) "rationale for this assumption's validity... is not explicitly justified"

