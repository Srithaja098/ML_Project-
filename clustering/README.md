# Clustering: Palmer Penguins Dataset

This module performs unsupervised clustering analysis on the **Palmer Penguins dataset** ([UCI Machine Learning Repository #690](https://archive.ics.uci.edu/dataset/690/palmer+penguins-3)).

Four clustering methodologies are systematically implemented, tuned, and compared:
1. **K-Means Clustering** (Elbow method and Silhouette analysis across $k \in [2, 10]$)
2. **Hierarchical / Agglomerative Clustering** (Ward linkage with dendrogram visualization)
3. **DBSCAN** (Density-based clustering with $k$-distance graph and parameter grid search)
4. **Gaussian Mixture Models (GMM)** (Probabilistic soft/hard clustering with BIC/AIC model selection)

Evaluations include both **unsupervised internal validation** (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index) and **external ground-truth validation** against the biological penguin species (*Adélie*, *Chinstrap*, *Gentoo*) via Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and V-Measure.

---

## Dataset Description

- **Source:** Palmer Station Antarctica LTER / Gorman et al. (2014), hosted on UCI ([Dataset #690](https://archive.ics.uci.edu/dataset/690/palmer+penguins-3)).
- **Instances:** 344 total penguins collected across 3 islands (Biscoe, Dream, Torgersen).
- **Cleaned Instances:** 342 complete observations (2 rows missing morphometric measurements dropped).
- **Features (4 continuous physical measurements):**
  - `bill_length_mm`: Culmen length in millimeters.
  - `bill_depth_mm`: Culmen depth in millimeters.
  - `flipper_length_mm`: Flipper length in millimeters.
  - `body_mass_g`: Body mass in grams.
- **Preprocessing:** All numerical measurements are standardized using `StandardScaler` (zero mean, unit variance) so that distance-based metrics are unbiased across differing measurement scales.
- **Ground Truth Target (for external validation only):** `species` (151 Adélie, 123 Gentoo, 68 Chinstrap).

---

## Experimental Results

### Quantitative Summary Comparison

| Model | Hyperparameters | Clusters Found | Noise Points | Silhouette Score (↑) | Calinski-Harabasz (↑) | Davies-Bouldin (↓) | Adjusted Rand Index (ARI ↑) | Normalized Mutual Info (NMI ↑) | V-Measure (↑) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gaussian Mixture Model (GMM)** | $k=3$, `covariance_type='full'` | **3** | **0** | **0.4545** | 425.96 | 0.8956 | **0.9603** | **0.9349** | **0.9349** |
| **Hierarchical (Agglomerative)** | $k=3$, `linkage='ward'` | **3** | **0** | 0.4541 | 420.72 | **0.8465** | **0.9159** | **0.9007** | **0.9007** |
| **K-Means** | $k=3$, `init='k-means++'` | **3** | **0** | 0.4472 | 441.68 | 0.9436 | 0.7928 | 0.7855 | 0.7855 |
| **DBSCAN** | $\varepsilon=0.6$, `min_samples=7` | **3** | 42 | 0.4933* | 458.80* | 0.7673* | 0.7662 | 0.7279 | 0.7279 |

*\*DBSCAN internal metrics computed on clustered points excluding noise.*

---

### Contingency Matrices (Mapping Clusters to Biological Species)

#### 1. Gaussian Mixture Model (GMM) — ARI: 0.9603
- **Cluster 0:** 123 Gentoo, 0 Adélie, 0 Chinstrap (**100% pure Gentoo**)
- **Cluster 1:** 149 Adélie, 3 Chinstrap, 0 Gentoo (**98.0% pure Adélie**)
- **Cluster 2:** 65 Chinstrap, 2 Adélie, 0 Gentoo (**97.0% pure Chinstrap**)
*Only 5 out of 342 penguins misassigned without ever seeing true species labels!*

#### 2. Hierarchical (Agglomerative Ward) — ARI: 0.9159
- **Cluster 0:** 151 Adélie, 11 Chinstrap, 0 Gentoo
- **Cluster 1:** 123 Gentoo, 0 Adélie, 0 Chinstrap (**100% pure Gentoo**)
- **Cluster 2:** 57 Chinstrap, 0 Adélie, 0 Gentoo (**100% pure Chinstrap**)

#### 3. K-Means ($k=3$) — ARI: 0.7928
- **Cluster 0:** 63 Chinstrap, 24 Adélie, 0 Gentoo
- **Cluster 1:** 123 Gentoo, 0 Adélie, 0 Chinstrap (**100% pure Gentoo**)
- **Cluster 2:** 127 Adélie, 5 Chinstrap, 0 Gentoo

#### 4. DBSCAN ($\varepsilon=0.6, \text{min\_samples}=7$) — ARI: 0.7662
- **Core/Dense Clusters:** Successfully discovered 3 distinct high-density regions corresponding to the 3 species.
- **Noise Points (Label -1):** 42 boundary/outlier points (20 Adélie, 14 Chinstrap, 8 Gentoo).

---

## Key Morphological Insights

1. **Gentoo Separation:** Across all algorithms (K-Means, Ward, DBSCAN, GMM), Gentoo penguins consistently form a clean, isolated cluster with **zero false positives**, driven by significantly larger body mass and longer flipper lengths.
2. **Adélie vs. Chinstrap Boundaries:** Adélie and Chinstrap penguins have overlapping body mass and flipper length distributions, but diverge in bill dimensions (Chinstrap has longer culmen with moderate depth; Adélie has shorter, deeper culmen). GMM's flexible full covariance ellipses effectively capture these non-spherical sub-populations.
3. **PCA Dimensionality Reduction:** The first two principal components capture **88.2% of total morphological variance** (PC1: 68.6%, PC2: 19.6%), confirming that the 4 standardized physical measurements naturally form 3 coherent biological clusters.

---

## Generated Visualizations and Outputs

Saved in `outputs/clustering/`:
- `kmeans_elbow_silhouette.png`: Dual Elbow (Inertia) and Silhouette score plots across $k \in [2, 10]$.
- `hierarchical_dendrogram.png`: Ward distance dendrogram showing hierarchical cluster merges.
- `dbscan_k_distance.png`: $k$-distance knee graph used to select $\varepsilon \approx 0.6$.
- `gmm_bic_aic.png`: BIC and AIC curves across component numbers and covariance structures.
- `pca_clustering_comparison.png`: 5-panel 2D PCA comparison showing Ground Truth vs. K-Means, Agglomerative, DBSCAN, and GMM.
- `feature_pairwise_clusters.png`: Bill Length vs. Depth and Flipper Length vs. Body Mass scatter plots.
- `clustering_summary_metrics.csv`: Comparative metric summary table.
- `<model>_contingency_matrix.csv`: Cluster-to-species cross-tabulations.

Saved in `artifacts/clustering/`:
- Fitted model files: `kmeans.joblib`, `agglomerative.joblib`, `dbscan.joblib`, `gmm.joblib`, `pca.joblib`, `scaler.joblib`.
- Full run metadata: `metadata.json`.

---

## How to Run

From the project root:

```powershell
# Run the complete clustering pipeline (training, evaluation, plotting, artifacts)
python -m clustering.cluster

# Run clustering unit tests
python -m unittest -v test_clustering
```
