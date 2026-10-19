# Conformal Label-Transition Auditing for Label-Flipping Federated Clients

Federated learning lets many clients train a shared model without sharing their data, but a few clients can poison training by flipping their labels. Standard defenses often compare update geometry, which can confuse such attackers with strongly non-IID honest clients. The paper "Conformal Label-Transition Auditing Against Label-Flipping Attacks in Trust-Aware Federated Learning" proposes auditing clients through label transitions under conformal prediction sets. This repository runs a small, self-contained version of that idea.

**Question:** Can a conformal label-transition audit flag a few label-flipping federated clients among strongly non-IID honest clients more reliably than a standard cosine-similarity update-outlier detector?

**Dataset:** scikit-learn's bundled digits dataset (no download). This is not the paper's own dataset (`same_as_paper: false`).

**Baseline:** a cosine-similarity update-outlier detector. Each client's update is compared with the coordinate-wise median update, and clients with a robust z-score below -2 are flagged.

**Improved model:** a conformal label-transition audit. For each client, a leave-one-client-out ensemble is built and calibrated on a held-out split. The audit computes the fraction of that client's labels that fall outside a 90% conformal prediction set, and flags clients above 0.3.

Both models share the same data split, federated training (softmax regression, FedAvg), metric, and evaluation code. The only difference is the flagging rule.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the scikit-learn digits dataset, the cosine-similarity update-outlier detector reached attacker-detection F1 of 0.93 while the conformal label-transition audit reached 0.95 over 3 seeds, a small gain well within the seed standard deviations (0.09 and 0.07).

| Metric | Value |
|---|---|
| baseline_score | 0.9333 |
| improved_score | 0.9524 |
| improvement | 0.01905 |
| baseline_score_std | 0.09428 |
| improved_score_std | 0.06734 |
| baseline_auc | 1 |
| improved_auc | 1 |
| baseline_accuracy | 0.7639 |
| improved_accuracy | 0.7481 |
| no_filter_accuracy | 0.7102 |
| n_seeds | 3 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Hyperparameters were fixed a priori and not tuned, so no CV selection was performed. The test set is used only for the final FedAvg accuracy. The paper's exact method details were not available; the improved model is a reconstruction from its name and description. Only 3 seeds and 20 clients per seed, so detection F1 is coarse. The digits dataset is a small stand-in, not the paper's data. Bundled data, so no download retries are needed.

## Reference

This project is motivated by the following paper:

> CW Wu, WC Hsu, YA Lien, SH Huang (2026). *Conformal Label-Transition Auditing Against Label-Flipping Attacks in Trust-Aware Federated Learning*. 2026 8th International. https://ieeexplore.ieee.org/abstract/document/11666546/

Paper author: Prof. Li-Der Chou, National Central University (cld@csie.ncu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
