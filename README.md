# EmbeddingDrift

**Measuring and Visualizing Concept Representation Drift Across LLM Versions**

> *Concepts encoded in LLM embeddings shift in meaningful ways across model versions and fine-tuning steps -- but this drift is invisible to end users and most researchers. EmbeddingDrift builds a framework to measure, track, and visualize this drift, with implications for model monitoring, alignment stability, and representation auditing.*

---

## Overview

EmbeddingDrift is a mechanistic analysis framework that quantifies how instruction tuning (RLHF) and quantization alter the internal concept representations of large language models. The project extracts last-layer embeddings for 235 semantically diverse concepts across three Llama-3.2-1B variants, computes pairwise cosine drift, performs k-NN neighborhood analysis, and projects the full embedding space into a shared UMAP coordinate system for trajectory visualization.

The core finding: **instruction tuning shifts demographic and identity-related concept representations ~7x more than quantization does, and ~26% more than factual/abstract concepts on average -- a fingerprint of RLHF alignment pressure that is invisible at the output level but measurable in representation space.**

---

## Models Compared

| Variant | Description | Drift Role |
|---|---|---|
| `meta-llama/Llama-3.2-1B` | Base model | Reference |
| `meta-llama/Llama-3.2-1B-Instruct` | RLHF instruction-tuned | Primary comparison |
| `meta-llama/Llama-3.2-1B` (4-bit NF4) | BitsAndBytes quantized base | Quantization noise baseline |

---

## Key Findings

### 1. Instruction Tuning Produces ~7.6x More Representation Drift Than Quantization

| Model Pair | Mean Cosine Distance | Std |
|---|---|---|
| Base → Instruct | **0.1507** | 0.0411 |
| Base → Quantized | **0.0199** | 0.0031 |
| Instruct → Quantized | **0.1639** | 0.0410 |

Quantization (4-bit NF4) is a near-identity operation on representations. Instruction tuning is not.

<img width="1790" height="617" alt="comp" src="https://github.com/user-attachments/assets/565d6e7b-3af0-48e7-a25a-4cb63bc8ad77" />

### 2. Demographic and Identity Concepts Drift Disproportionately

Category-level drift summary (base → instruct):

<img width="1189" height="590" alt="drift" src="https://github.com/user-attachments/assets/53e1a8e4-19a1-423a-a645-511e18b0d3e5" />

| Category | Mean Drift | % in Top Quartile |
|---|---|---|
| Demographic | **0.1798** | **58.3%** |
| Scientific | 0.1459 | 22.0% |
| Factual | 0.1434 | 20.0% |
| Abstract | 0.1342 | 1.5% |

Demographic concepts account for 58% of the top-quartile drifters despite comprising only 26% of the vocabulary. Abstract political concepts (democracy, liberty, justice) drift *least* -- suggesting RLHF encodes identity-sensitivity at the representation level rather than through ideological reweighting.

### 3. Top Drifted Concepts (Base → Instruct)

<img width="1390" height="790" alt="top 30" src="https://github.com/user-attachments/assets/24f5aba1-a841-405f-8cd1-4f7c9dab58ed" />

| Rank | Concept | Category | Cosine Distance |
|---|---|---|---|
| 1 | religious | demographic | 0.3951 |
| 2 | Muslim | demographic | 0.3951 |
| 3 | Jewish | demographic | 0.3368 |
| 4 | quantum | scientific | 0.3287 |
| 5 | criminal | demographic | 0.3258 |
| 6 | Buddhist | demographic | 0.2977 |
| 7 | Christian | demographic | 0.2882 |
| 8 | Asian | demographic | 0.2854 |
| 9 | secular | demographic | 0.2803 |
| 10 | neural | scientific | 0.2755 |

Note: `quantum` and `neural` appearing in the top 10 alongside religious/identity terms is a small-model (1B) effect -- compressed embedding space causes instruction-tuning signal to bleed into high-frequency technical tokens. This effect diminishes at larger model scales.

### 4. Semantic Neighborhood Restructuring

The most informative drift signal is not magnitude but *neighborhood identity change*, measured by Jaccard overlap of k-NN sets (k=10):

**`prisoner` (base → instruct) -- Jaccard overlap: 0.053 (nearly complete neighborhood replacement)**

| Rank | Base | Instruct |
|---|---|---|
| 1 | politician (0.972) | activist (0.912) |
| 2 | criminal (0.968) | bystander (0.906) |
| 3 | teacher (0.966) | agnostic (0.900) |
| 4 | minority (0.966) | immigrant (0.885) |
| 5 | court (0.965) | Feynman (0.879) |

Base model: `prisoner` is profession/institution-adjacent. Instruct model: `prisoner` is justice/identity-adjacent. This is a qualitative representation shift, not just a vector rotation.

---
## UMAP Trajectory Visualization

Each circle is a concept in the base model's embedding space; each triangle is the same concept after instruction tuning. Arrows show the displacement — **red = high drift, blue = low drift**.

<img width="950" height="680" alt="newplot" src="https://github.com/user-attachments/assets/d824e6b7-4c5a-44ed-b8e8-8757842ddfee" />

The same shared coordinate space, now colored by concept category. Demographic concepts (red) are scattered across multiple clusters, consistent with their higher variance in drift scores.

<img width="950" height="700" alt="newplot (1)" src="https://github.com/user-attachments/assets/84429762-ca53-4438-a18c-a65b08536a52" />

---
## Pipeline

```
1. Setup and Configuration          -- device, seeds, model configs
2. Concept Vocabulary (235 terms)   -- factual, abstract, demographic, scientific
3. Embedding Extraction             -- last hidden layer, last non-pad token, L2-normalized
4. Drift Measurement                -- pairwise cosine distance across all 3 variant pairs
5. Semantic Neighborhood Analysis   -- k-NN Jaccard overlap (k=10)
6. Category Drift Analysis          -- per-category statistics + violin/bar plots
7. UMAP Trajectory Visualization    -- joint-fit UMAP, arrow trajectories colored by drift
8. Gradio Interactive Demo          -- per-concept drift lookup, UMAP, k-NN panels
```

---

## Concept Vocabulary Design

235 unique terms across 4 categories, designed to produce interpretable drift signals:

| Category | N | Expected Behavior |
|---|---|---|
| **factual** | 60 | Countries, scientists, institutions, objects -- low expected drift |
| **abstract** | 66 | Political, moral, epistemic concepts -- moderate expected drift |
| **demographic** | 60 | Identity, religion, race, profession -- highest expected drift after RLHF |
| **scientific** | 50 | Physics, biology, CS, medicine -- low to moderate drift |

---

## Methodology

### Embedding Extraction

For each concept term, the prompt `"The concept of {term} is"` is constructed and tokenized. A forward pass extracts the hidden state at the final non-padding token position from the last transformer layer. Embeddings are L2-normalized so that cosine similarity equals the dot product.

```python
# Core extraction logic
hidden = outputs.hidden_states[-1]          # (batch, seq_len, hidden_dim)
last_token_idx = attention_mask.sum(dim=1) - 1
emb = hidden[b_idx, tok_idx, :].float().cpu().numpy()
embeddings = normalize(embeddings, norm="l2")
```

### Drift Metric

Cosine distance: `drift = 1 - cosine_similarity(emb_a, emb_b)`. Range [0, 2]; higher = more drift. For L2-normalized embeddings this reduces to `1 - dot(emb_a, emb_b)`.

### Neighborhood Analysis

Full N×N cosine similarity matrix computed for each variant. Jaccard overlap between k-NN sets measures semantic neighborhood stability independently of raw distance.

### UMAP Visualization

All variant embeddings concatenated and fit jointly in a single UMAP pass (cosine metric, n_neighbors=15, min_dist=0.1). Joint fitting ensures shared coordinate system so trajectory arrows reflect genuine movement, not projection artifacts.

---

## Caveats and Limitations

- **Small model effects:** Llama-3.2-1B has a compressed 2048-dim embedding space. Bleeding of alignment signal into unrelated technical tokens (`quantum`, `neural`) is expected to diminish at 7B+ scale. Results should be validated on larger models before strong mechanistic claims are made.
- **Single extraction point:** Embeddings come from the last hidden layer only. Probing intermediate layers would reveal *where* in the network drift is localized, which is the more mechanistically interesting question.
- **Prompt sensitivity:** The template `"The concept of {term} is"` standardizes extraction but may activate different token positions for multi-word concepts. Single-token terms are most reliable.
- **UMAP global structure:** At this dataset scale (235 points × 3 variants), UMAP trajectory arrows show a global shift pattern that partly reflects the systematic last-layer representation change between base and instruct models. Per-concept arrow *direction* is not reliably interpretable; arrow *length* correlates with cosine distance scores.
- **No causal claims:** Cosine distance measures correlation between alignment training and representation change, not a causal mechanism. Activation patching or path patching would be needed to establish causality.

---

## Setup

```bash
pip install transformers accelerate bitsandbytes huggingface_hub \
    umap-learn scikit-learn matplotlib seaborn plotly pandas numpy \
    gradio scipy tqdm ipywidgets
```

**Hardware:** Any CUDA GPU with 8+ GB VRAM for 1B models. The sequential load/unload pattern supports T4 (16 GB) and above. Tested on NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM).

**HuggingFace token:** Required for Llama model access. Set `HF_TOKEN` as environment variable or Colab secret.

---

## Running

```bash
# Clone and open in Jupyter / Colab
jupyter notebook EmbeddingDrift.ipynb
```

Embeddings are saved to `embedding_drift_artifacts/` after extraction. Subsequent runs can load from checkpoint by uncommenting the checkpoint reload cell (Section 3).

---

## Output Artifacts

| File | Description |
|---|---|
| `embeddings_{variant}.npy` | Raw L2-normalized embeddings (235, 2048) per variant |
| `concept_metadata.json` | Concept list and category assignments |
| `top30_drift_bar.png` | Horizontal bar chart: top 30 most-drifted concepts |
| `category_drift_comparison.png` | Mean drift by category across all 3 model pairs |
| `drift_violin_by_category.png` | Drift score distribution violin plots |
| `umap_trajectory.html` | Interactive UMAP with trajectory arrows |
| `umap_by_category.html` | Interactive UMAP colored by concept category |
| `umap_{variant}.npy` | 2D UMAP coordinates per variant |

---

## Implications

**For model monitoring:** Cosine drift of concept embeddings is a lightweight, interpretable signal for detecting alignment-induced representation changes between model versions. It requires no labeled data and runs in minutes.

**For alignment auditing:** The disproportionate drift in demographic/identity concepts suggests RLHF encodes social sensitivity at the representation level, not just at the output distribution. This is detectable without access to training data or RLHF reward signals.

**For deployment:** Quantization (4-bit NF4) introduces negligible representation drift (mean 0.0199 vs 0.1507 for instruction tuning). Monitoring pipelines should focus on fine-tuning and RLHF steps, not quantization.

---

## Future Directions

- Extend to 7B/13B/70B model scales to test whether small-model bleeding effects disappear
- Layer-wise drift analysis to localize *where* in the network alignment pressure is applied
- Temporal drift tracking via OLMo training checkpoints (see Appendix, `USE_TEMPORAL_DRIFT = True`)
- Causal intervention (activation patching) to test whether drifted representations causally affect downstream behavior
- Cross-architecture comparison: does instruction tuning drift differently in Mistral vs. Llama vs. Gemma?

---

## Author

**Jacob O** | [GitHub: agentjakey](https://github.com/agentjakey)

*ML Research | Mechanistic Interpretability | Representation Auditing*
