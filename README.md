<div align="center">

# E²R

### Farmland Remote-Sensing Image-Text Retrieval

**Anonymous code release · Paper under review**

</div>

E²R is a cross-modal retrieval framework for fine-grained farmland remote-sensing scenes. It combines global-local alignment with **Representation Suppression (RES)** and **Keyword-based Explicit Reasoning (KER)**. This repository contains the core research implementation and configuration needed to reproduce the method; datasets, checkpoints, logs, and unrelated baseline code are intentionally excluded.

## Framework

![E2R architecture](figures/e2r-architecture.png)

*E²R architecture. CLIP encoders extract global and token-level representations; RES suppresses low-confidence positive pairs, while KER predicts masked farmland keywords through cross-modal reasoning.*

## FRS-CMR

![FRS-CMR collection](figures/frs-cmr-collection.png)

*FRS-CMR collection. Farmland imagery is acquired from satellite, UAV, and ground-based platforms across diverse geographic regions and paired with concise retrieval-oriented descriptions.*

![FRS-CMR construction](figures/frs-cmr-construction.png)

*FRS-CMR construction. A structured prompt guides Qwen2.5-VL-7B-Instruct to describe vegetation cover, field shape, man-made structures, spatial relations, and spectral color; the figure also summarizes vocabulary and regional image counts.*

Each image is assigned three distinct captions generated with [Qwen2.5-VL-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct). A randomly sampled 40% of image-caption pairs is independently inspected by four domain experts. Incorrect visual content, spatial relations, redundant details, and inconsistent terminology are revised. The reported Fleiss' kappa range is 0.83-0.89.

| Split | Images |
|---|---:|
| Train | 18,081 |
| Validation | 2,260 |
| Test | 2,261 |

Dataset files are not distributed in this anonymous code repository.

## Core components

- **Global-local alignment:** jointly optimizes image-text correspondence at representation and token levels.
- **RES:** activates after warm-up and removes low-confidence positive anchors according to the similarity bank.
- **KER:** masks frequent farmland keywords and reconstructs them using masked text and visual tokens.

The focused implementation is organized as follows:

```text
E2R/
├── configs/frs_cmr.yaml
├── figures/
└── src/e2r/
    ├── model.py          # E²R model, global-local features, RES and KER
    ├── loss.py           # weighted retrieval and masked-keyword objective
    ├── train.py          # core optimization loop and RES threshold update
    ├── data.py           # FRS-CMR CSV dataset interface
    ├── tokenizer.py      # keyword-aware masking
    └── clip_blocks.py    # CLIP transformer components
```

## Experimental configuration

| Item | Setting |
|---|---|
| Backbone | OpenCLIP ViT-B/32 |
| Input resolution | 224 × 224 |
| Context length | 77 tokens |
| Embedding dimension | 512 |
| Cross-modal transformer | 2 layers, 8 heads |
| Training epochs | 50 |
| Batch size | 128 |
| Optimizer | AdamW |
| Initial learning rate | 1.5 × 10⁻⁵ |
| Weight decay | 0.7 |
| Schedule | 200-step linear warm-up, then cosine decay |
| Gradient clipping | 50 |
| Global / local / MLM weights | 0.7 / 0.3 / 0.5 |
| RES activation epoch | 4 |
| FRS-CMR suppression ratio | 1% |
| Keyword vocabulary | 256 words |
| Training hardware | 4 × NVIDIA GeForce RTX 3090 |

The experiments initialize both encoders from the same OpenCLIP ViT-B/32 checkpoint. Pretrained weights are not redistributed; set `pretrained_checkpoint` in `configs/frs_cmr.yaml` to a locally obtained OpenCLIP-compatible checkpoint.

## Data format

The CSV loader expects one image-caption pair per row:

```csv
filename,title
example/000001.jpg,"Rectangular green fields lie beside a narrow road."
```

Update the dataset, checkpoint, BPE, and keyword-vocabulary paths before training. The complete reported configuration is provided in `configs/frs_cmr.yaml`.

## Installation

```bash
conda create -n e2r python=3.10 -y
conda activate e2r
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Evaluation

Performance is evaluated using image-to-text and text-to-image Recall@1, Recall@5, Recall@10, and their six-metric mean recall (mR). The reported FRS-CMR result is **30.73% mR**.

## Citation

```bibtex
@article{anonymous2026e2r,
  title   = {FRS-Searching: Cross-Modal Image-Text Retrieval in Farmland Remote Sensing with Large Vision-Language Models},
  author  = {Anonymous},
  year    = {2026},
  note    = {Under review}
}
```

Author identities and affiliations are intentionally omitted during anonymous review.
