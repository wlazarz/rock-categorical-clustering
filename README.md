# ROCK: Robust Clustering Using Links

A Python implementation of the **ROCK** algorithm for clustering categorical data, based on the original paper:

> **Guha, S., Rastogi, R., & Shim, K.** (2000). *ROCK: A Robust Clustering Algorithm for Categorical Attributes*. In _Proceedings of the 2000 International Conference on Management of Data_ (SIGMOD).

## Introduction

ROCK (Robust Clustering using LinKs) is a hierarchical clustering algorithm designed specifically for categorical data. Unlike traditional distance-based methods, ROCK uses a notion of **link** (common categorical co-occurrences) to group objects, providing resilience to noise and skewed distributions.


## Algorithm Overview

1. **One-hot encoding**: Transform categorical attributes into a binary sparse matrix.
2. **Link graph construction**: Compute Jaccard-based link weights between objects, and keep only pairs whose similarity ≥ ε.
3. **Merge strategy**: Use a goodness measure based on the number of cross‑links and cluster sizes to greedily merge clusters until the desired number **k** is reached.
4. **Label assignment**: Derive final cluster labels via union‑find disjoint‑set structure.


## Installation

```
# Clone repository
git clone https://[github.com/your-org/rock-clustering.git](https://github.com/wlazarz/rock-categorical-clustering)
cd rock-categorical-clustering

# Install dependencies
pip install -r requirements.txt
```

