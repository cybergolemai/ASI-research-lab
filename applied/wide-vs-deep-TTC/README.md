# More Compute, More Confusion: The Scaling Trap of Inference-Time Reasoning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cybergolemai/ASI-research-lab/blob/main/applied/wide-vs-deep-TTC/code/CyberGolem_Blog_Wide_vs_Deep_Inference.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and analysis for an experiment designed to test a critical hypothesis in AI efficiency: can smaller, VRAM-friendly models unlock the complex reasoning capabilities of their larger cousins by being granted significantly more computational time at inference?

We explore a 2D space of inference-time compute:
1.  **"Wider" Reasoning (n):** Using parallel, stochastic forward passes to form a robust "consensus" prediction.
2.  **"Deeper" Reasoning (m):** Using sequential, recursive steps to simulate a deliberate, multi-step thought process.

The results were unambiguous and serve as a valuable case study on the limitations of small models and the importance of foundational capabilities forged during training.

---

## The Hypothesis: Can Compute Create Capability?

The core idea, inspired by recent research like arXiv:2510.04871 ("Less is More"), is that a model's performance is a function of not just its parameters, but also the compute applied during generation. We set out to map the performance of a base GPT-2 model on a simple arithmetic reasoning task across a 16x16 grid of "wide" vs. "deep" compute configurations.

-   **Width (n):** The number of parallel forward passes with dropout enabled. The logits from these passes are averaged to produce the final probability distribution for the next token.
-   **Depth (m):** The number of internal "thought" steps the model takes, refining a "scratchpad" of reasoning, before committing to generating a single token of the final answer.

## Methodology

The full experiment is contained within a single Google Colab notebook.

-   **Model:** `gpt2` (124M parameters) from Hugging Face.
-   **Task:** A simple, multi-step arithmetic problem: *"Alice has 5 apples. She gives 2 to Bob. She then eats 1 apple. How many apples does Alice have left?"* (Correct Answer: 2).
-   **Grid Search:** We iterated through `n_wide` and `m_deep` values of `[1, 2, 4, 8, 16]`, covering 25 distinct configurations.
-   **Metrics:** We meticulously logged correctness, latency (wall-clock time), total forward passes (a proxy for compute), and the model's average confidence in its chosen tokens.

## Results: A Multiplier on Zero is Still Zero

The experiment produced a clear and decisive result: **accuracy remained at 0% across all 25 configurations.**

Increasing the computational budget, sometimes by more than two orders of magnitude, had no positive effect on the model's ability to solve the problem.

### Performance Overview



The heatmaps above illustrate the core findings:
1.  **Correctness (Left):** Remained at 0.00 for every single `(n, m)` pair.
2.  **Latency (Center):** Increased exponentially with both width and depth, from ~0.5s to over 40s.
3.  **Confidence (Right):** Showed no clear trend. More compute did not lead to higher confidence, and in some cases, appeared to decrease it, suggesting the model became more confused.

### Performance vs. Computational Cost



This plot is the most telling. The y-axis represents correctness, and the x-axis (log scale) represents the total compute used. All data points are locked to the "Incorrect" baseline, demonstrating a zero return on investment for the additional computation.

## Conclusion

This experiment strongly suggests that **inference-time algorithms are multipliers on a model's foundational capability, not creators of it.**

GPT-2 lacks the emergent ability to perform the multi-step arithmetic reasoning required by the task. Our findings show that no amount of ensembling ("wider") or recursive thinking ("deeper") at inference time can create a skill that was not learned during pre-training.

Giving a model that cannot reason more time to think does not help it find the right answer; it only gives it more opportunities to explore a vast space of wrong answers, ultimately leading to more compute and more confusion. This work highlights a fundamental "scaling trap" and underscores the critical importance of a model's inherent capabilities derived from training-time scale.

## How to Run This Experiment

1.  Click the **"Open in Colab"** badge at the top of this README.
2.  In the Colab notebook, select a GPU runtime (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3.  Run the single code cell. The grid search will execute and generate the same data visualizations presented here.
