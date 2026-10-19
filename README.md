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

## Reference

This project is motivated by the following paper:

> CW Wu, WC Hsu, YA Lien, SH Huang (2026). *Conformal Label-Transition Auditing Against Label-Flipping Attacks in Trust-Aware Federated Learning*. 2026 8th International. https://ieeexplore.ieee.org/abstract/document/11666546/

Paper author: Prof. Li-Der Chou, National Central University (cld@csie.ncu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
