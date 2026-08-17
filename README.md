# MRLOD

MRLOD is a multimodal representation-learning framework for cancer subtyping.

This repository provides the source code for the paper **"MRLOD: Multimodal
Representation Learning with Omics-Specific Diffusion for Cancer Subtyping"**.

## Repository layout

```text
multimodal/        Multimodal MRLOD training and subtype-assignment code
  assign_subtypes.py
                   PCA and K-means subtype assignment from saved latents
  clustering.py    Dataset-specific cluster counts and assignment output
  clean_reports.py Remove potential label information from report text
  diffusion.py     Cosine diffusion schedule and timestep embeddings
  losses.py        Reconstruction, cluster-semantic, and entropy losses
  model.py         Multimodal MRLOD model definition
  modules.py       Encoders, fusion blocks, adapters, and decoders
  train.py         Training, latent extraction, and artifact export
  utils.py         Reproducibility, preprocessing, and Dataset utilities
omic/              MRLOD-Omics training and subtype-assignment code
data/              Placeholder for user-provided model inputs
outputs/           Reference subtype assignments
environment.yml    Reproducible Conda environment
SYSTEM_REQUIREMENTS.md
                   Hardware, CUDA, driver, and installation notes
```

## Installation

Create Conda environment:

```powershell
conda env create --name ENV_NAME -f environment.yml
conda activate ENV_NAME
```

[SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) records the reference
hardware, NVIDIA driver and environment verification commands.

## Data layout

Place prepared inputs under
`data/`:

```text
data/
  OMIC/
    tcga_blca_all_clean.csv
    tcga_brca_all_clean.csv
    ...
  TCGA_TITAN_features.pkl
  tcga_reports_embeddings.csv
```

## Training and subtype assignment

After activating Conda environment, run one or more datasets from the
repository root:

```powershell
python multimodal/train.py --datasets tcga_brca_all_clean
python multimodal/assign_subtypes.py --datasets tcga_brca_all_clean
```

Omit `--datasets` to run all ten configured TCGA datasets

## datasets and pretrained representations

### Whole-slide image representations

Download `TCGA_TITAN_features.pkl` from the official
[MahmoodLab/TITAN repository](https://huggingface.co/MahmoodLab/TITAN).
The file should be placed at `data/TCGA_TITAN_features.pkl`.

### Pathology report representations

Download `TCGA_Reports.csv.zip` from the official
[TCGA-Reports dataset](https://doi.org/10.17632/hyg5xkznpx.1).

To reproduce the report embeddings, load
[`thomas-sounack/BioClinical-ModernBERT-base`](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base)
with Hugging Face Transformers. Tokenize each report using the default
tokenizer with dynamic padding and truncation at 8,192 tokens. Run the model in
evaluation mode and obtain a 768-dimensional representation by attention-mask
mean pooling over the final hidden states.

### references

- Ding T, Wagner SJ, Song AH, et al. A multimodal whole-slide foundation model
  for pathology. *Nature Medicine*. 2025;31:3749-3761.
  [doi:10.1038/s41591-025-03982-3](https://doi.org/10.1038/s41591-025-03982-3)
- Kefeli J, Tatonetti N. TCGA-Reports: A machine-readable pathology report
  resource for benchmarking text-based AI models. *Patterns*. 2024;5:100933.
  [doi:10.1016/j.patter.2024.100933](https://doi.org/10.1016/j.patter.2024.100933)
- Sounack T, Davis J, Durieux BN, et al. BioClinical ModernBERT: A
  state-of-the-art long-context encoder for biomedical and clinical NLP.
  *arXiv*. 2025.
  [doi:10.48550/arXiv.2506.10896](https://doi.org/10.48550/arXiv.2506.10896)
