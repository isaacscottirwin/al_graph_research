# Active Learning on Signed Graphs

Research code for studying graph-based semi-supervised learning, active learning, and spectral methods on signed graphs. The project focuses on how graph structure modifications. Particularly how deleting versus negating cross-class edges affect label propagation, spectral embeddings, uncertainty estimation, and active querying behavior.

The repository includes implementations of signed Laplace learning, signed Poisson learning, graph construction utilities, active querying experiments, edge alteration experiments, spectral analysis tools, and visualization utilities.

---

# Core Ideas

The repository studies binary classification on signed graphs where:

- Positive edges represent similarity or agreement.
- Negative edges represent dissimilarity or disagreement.
- Labels are encoded as:

```python
{-1, +1}
```

- Unlabeled nodes are represented by:

```python
0
```

The primary graph operator used throughout the repository is the signed Laplacian

```math
L_{\mathrm{signed}} = D_{|\!A\!|} - A
```

where

```math
D_{|\!A\!|}(i,i) = \sum_j |A_{ij}|.
```

Both combinatorial and normalized signed Laplacians are supported.

---

# Main Components

## Signed Laplace Learning

Implementation of graph-based semi-supervised learning using signed Laplace equations.

Key features:

- Signed combinatorial Laplacian
- Signed normalized Laplacian
- Binary label propagation
- Continuous score outputs
- Spectral embedding support
- Higher-order Laplacian support
- Reweighting options inherited from GraphLearning

Main class:

```python
AlteredLaplace
```

---

## Signed Poisson Learning

Implementation of signed Poisson learning using conjugate gradient solvers on signed Laplacians.

Key features:

- Signed Poisson propagation
- Continuous score outputs in `[-1, 1]`
- Signed combinatorial and normalized Laplacians
- Sparse linear algebra support
- Conjugate gradient solvers
- GraphLearning integration

Main class:

```python
AlteredPoisson
```

The implementation modifies the original GraphLearning Poisson solver to support signed adjacency matrices.

---

## Active Querying Experiments

Framework for studying active learning on signed graphs.

Main experiment class:

```python
ActiveQueryingExperiment
```

Each run:

 1. Find new cross-class edges among the currently labeled nodes.
 2. Apply the selected edge alteration strategy.
 3. Recompute the signed Laplacian.
 4. Recompute and store the spectral embedding.
 5. Predict labels and scores using the selected SSL model.
 6. Store predictions.
 7. Compute and store accuracy before adding new queried labels.
 8. Select new unlabeled nodes using the query rule.
 9. Add queried labels to the training set.
 10. Store the updated labeled-index history.
 11. Update eigenvalue and eigenvector histories.

Supported querying methods:

### Uncertainty Querying

Selects unlabeled nodes whose scores are closest to zero.

### Frustration Querying

Selects nodes incident to the largest number of frustrated positive edges.

---

## Edge Alteration Experiments

Framework for isolating the effect of graph structure modifications.

Main experiment class:

```python
EdgeAlterationExperiment
```

Each Run:
 1. Retrieve the current edge batch.
 2. Apply edge alterations to the current adjacency matrix.
 3. Recompute the signed Laplacian.
 4. Recompute and store the spectral embedding.
 5. Predict labels on the updated graph.
 6. Store the prediction history.
 7. Compute and store accuracy using the fixed labeled set.
 8. Store the current labeled-index history.
 9. Update eigenvalue and eigenvector histories.


Unlike active querying experiments:

- The labeled set is fixed.
- No new labels are queried.
- Only graph structure changes over time.

Positive cross-class edges are progressively:

- deleted (`zero`)
- negated (`negate`)

The framework tracks how these changes affect:

- prediction accuracy
- embeddings
- eigenvalues
- eigenvectors
- graph spectra

---

# Graph Construction

Graphs are constructed using:

```python
KNNGraph
```

Supported features:

- k-nearest-neighbor graph construction
- sparse adjacency matrices
- multiple kernels
- multiple graph construction methods
- spectral decomposition utilities

---

# Spectral Analysis

The repository contains utilities for analyzing spectral behavior under graph modifications.

Tracked quantities include:

- eigenvalues
- eigenvectors
- signed Laplacians
- embeddings
- modularity-style quantities
- frustrated edges
- graph partitions

The codebase heavily emphasizes spectral evolution over experiment rounds.

---

# Frustration

For a predicted sign function

```math
g : V \to \{-1,1\},
```

an edge is considered frustrated if its sign disagrees with the predicted node signs.

For positive edges, frustration occurs when the incident nodes are assigned opposite predicted signs.

The repository includes:

- frustration-based query strategies
- frustration counting utilities
- sparse frustration computations
- signed graph analysis tools

---

# Visualization Utilities

Visualization utilities support:

- dataset visualization
- embedding visualization
- prediction visualization
- spectral metric plotting
- eigenvector evolution
- animated embeddings
- animated prediction trajectories

Animations are implemented using:

```python
matplotlib.animation.FuncAnimation
```

Supported visualizations include both 2D and 3D embeddings.

---

# Repository Structure

```text
src/
    Core implementation.

tests/
    Unit tests and debugging experiments.

notebooks/
    Lightweight experiment drivers and exploratory analysis.

configs/
    Experiment configuration files.

results/
    Generated outputs, plots, animations, and experiment artifacts.

docs/
    LaTeX writeup and research documentation.
```

---

# Important Classes

## Models

```python
AlteredLaplace
AlteredPoisson
Model
```

## Experiments

```python
ActiveQueryingExperiment
EdgeAlterationExperiment
ExperimentResult
RunState
```

## Graph Utilities

```python
KNNGraph
SignedLaplacian
EdgeModification
GraphAnalysis
BatchSequences
Metrics
```

## Active Learning

```python
LabelPropagation
```

---

# Dependencies

Main dependencies include:

```python
numpy
scipy
matplotlib
networkx
graphlearning
```
---

# Example Workflow

Typical experiment workflow:

1. Construct dataset
2. Build kNN graph
3. Compute signed Laplacian
4. Initialize labeled nodes
5. Run Laplace or Poisson propagation
6. Query or alter graph edges
7. Recompute embeddings and predictions
8. Track metrics over time
9. Visualize spectral evolution

---

# Research Notes

This repository is an active research codebase. Some methods are experimental and still under investigation.

In particular:

- Signed Poisson learning behavior is still being studied.
- Signed modularity utilities may not yet be mathematically correct.
- Some graph alteration strategies behave differently than initially expected.
- Spectral behavior can depend strongly on graph normalization choices.

The codebase prioritizes experimentation and analysis over production stability.

---

# Installation

Clone the repository:

```bash
git clone <repo-url>
cd <repo-name>
```

Install in editable mode:

```bash
pip install -e .
```

---

# Notes

This repository was developed for research into:

- signed graph learning
- graph signal processing
- active learning
- spectral graph theory
- graph-based semi-supervised learning
- signed Laplacians
- uncertainty estimation on graphs
- graph structure modification
- spectral embeddings
- frustration dynamics