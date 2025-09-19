# Failures of Combinatorial Reasoning in Large Language Models: A De Morgan's Law Benchmark

[![Paper](https://img.shields.io/badge/Paper-Preprint%20v1-blue)](https://github.com/cybergolemai/ASI-research-lab/blob/main/pure/hallucination/paper/(preprint%20v1)%20paper.pdf)
[![Colab](https://img.shields.io/badge/Colab-Replication-yellow)](https://colab.research.google.com/drive/1yhFh-gnmjT06ELL-eOs5ITgrkX1iV_cc?usp=sharing)
[![License](https://img.shields.io/badge/License-MIT-green)]()

<a target="_blank" href="https://colab.research.google.com/github/cybergolemai/ASI-research-lab/tree/main/pure/hallucination/code/replication.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Research Question

**Are modern LLMs sensitive to function depth in recursively composed logical expressions?**

This research investigates whether Large Language Models exhibit systematic failures when applying De Morgan's laws to Boolean expressions of varying recursive depths.

## Key Findings

We evaluated **9,000+ boolean algebra expressions** through Gemini 2.5 Flash and discovered:

- **Non-monotonic error pattern**: Rather than errors increasing proportionally with depth, we observed a surprising peak at depth 3-4 (43.3% error rate)
- **Error stabilization**: After the peak, errors stabilized around 10-15% regardless of further depth increases
- **Deterministic failures**: All tests conducted at temperature=0, proving failures are systemic, not stochastic

![Error Rate Graph](https://github.com/cybergolemai/ASI-research-lab/blob/main/pure/hallucination/figures/error_rate_chart.png)
*Error rates vs. expression depth showing non-monotonic pattern with 95% CI*

### Try it in Google Colab
[Open Replication Notebook in Colab](https://colab.research.google.com/drive/1yhFh-gnmjT06ELL-eOs5ITgrkX1iV_cc?usp=sharing)

## 📁 Repository Structure

```
pure/hallucination/
├── paper/
│   ├── main.tex                   # Preprint LaTeX manuscript
│   ├── references.bib             # Preprint manuscript citations
│   └── (preprint v1) paper.pdf    # Full research paper
└── code/
    ├── replication.ipynb          # Complete experimental replication Notebook
    └── dpo_benchmark.jsonl        # DPO benchmark for future work
```

## 🔧 Methodology

### Problem Generation
- Procedurally generated Boolean expressions requiring recursive application of De Morgan's laws
- Controlled complexity via recursive depth parameter (1-31 levels)
- 300 samples per depth range

### Ground Truth Verification
- Custom Abstract Syntax Tree (AST) solver for deterministic verification
- Canonical form comparison for logical equivalence
- Automated error detection and classification

### Evaluation Protocol
- Model: Gemini 2.5 Flash (temperature=0)
- Zero-temperature sampling to eliminate stochasticity
- Detailed few-shot prompting with complex examples

## 📈 Results Summary

| Depth Range | Error Rate | Sample Size |
|------------|------------|-------------|
| 1-2        | 3.3%       | 300         |
| 3-4        | **43.3%**  | 300         |
| 5-6        | 23.0%      | 300         |
| 7-8        | 16.3%      | 300         |
| 9-10       | 11.3%      | 300         |
| 15-16      | 12.3%      | 300         |
| 29-30      | 11.7%      | 300         |

*Note: All error rates represent lower bounds due to silent API failures*

## Open Research Questions

1. **Peak Error Replication**: Does the dramatic peak at depth 3-4 replicate across different models and seeds?

2. **Cross-Model Stability**: Is the error distribution stable between models, or are some LLMs more prone to memorization vs. reasoning?

3. **Fine-tuning Potential**: Can SFT/PEFT adapters be trained on smaller models to achieve superior performance?

4. **Broader Symbolic Tasks**: What other symbolic AI tasks can torture-test transformer-based models in out-of-distribution contexts?

## 🛠️ Technical Implementation

### Example Problem
```
Input:  ¬((¬(A0) ∧ ¬(A1)) ∨ ¬((¬(A2) ∧ ¬(A3))))
Truth:  (A0 ∨ A1) ∧ (A2 ∨ A3)
LLM:    (A0 ∨ A1) ∧ A2 ∧ A3  ❌
```

### AST Solver Rules
- Double Negation: `¬(¬A) → A`
- De Morgan's (AND): `¬(A ∧ B) → (¬A ∨ ¬B)`
- De Morgan's (OR): `¬(A ∨ B) → (¬A ∧ ¬B)`

## 📚 Citation

```bibtex
@article{asi2025demorgan,
  title={Failures of Combinatorial Reasoning in Large Language Models: 
         A De Morgan's Law Benchmark},
  author={ASI Research Lab},
  institution={CyberGolem LLC},
  year={2025},
  month={September}
}
```

## 🔗 Resources

- **Paper**: [Preprint v1 (PDF)](https://github.com/cybergolemai/ASI-research-lab/blob/main/pure/hallucination/paper/(preprint%20v1)%20paper.pdf)
- **Replication Code**: [Jupyter Notebook](https://github.com/cybergolemai/ASI-research-lab/tree/main/pure/hallucination/code/replication.ipynb)
- **Google Colab**: [Interactive Replication](https://colab.research.google.com/drive/1yhFh-gnmjT06ELL-eOs5ITgrkX1iV_cc?usp=sharing)
- **DPO Benchmark**: [JSONL Dataset](https://github.com/cybergolemai/ASI-research-lab/blob/main/pure/hallucination/code/dpo_benchmark.jsonl)

## 🤝 Contributing

We welcome contributions! Areas of particular interest:
- Replication with other LLMs (GPT-4, Claude, Llama, etc.)
- Extension to other logical systems
- Fine-tuning experiments
- Error pattern analysis

## 📧 Contact

ASI Research Lab, CyberGolem LLC  
Email: asi@cybergolem.ai

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*"Hallucination is not a bug, it's the default behavior when models extrapolate into unseen combinatorial states."*
