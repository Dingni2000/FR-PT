# Hierarchical Feature-level Reverse Propagation for Post-Training Neural Networks

Official PyTorch implementation of **Feature-level Reverse Propagation for
Post-Training (FR-PT)**. FR-PT propagates supervision from ground-truth labels
backward through downstream network by reconstructing label-conditioned features, which
are then used to supervise intermediate representations. See the paper on
[arXiv](https://arxiv.org/abs/2506.07188).

## Repository Structure

- [`imagecls/rvs_cpt/`](imagecls/rvs_cpt/): operator-specific feature
  reconstruction algorithms.
- [`imagecls/FRPT.py`](imagecls/FRPT.py): image classification post-training
  experiments.
  - `model_post_train_all`: runs post-training experiments across reconstructed
    feature targets.
  - `ablation_all`: compares the output-target embedding strategies used in the
    ablation study.
- [`navsim/`](navsim/): trajectory planning experiments on NAVSIM.
- [`effi_stab_exp/`](effi_stab_exp/): convolutional reconstruction experiments
  corresponding to Section IV-B(1) of the paper.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{ding2026hierarchical,
  title         = {Hierarchical Feature-level Reverse Propagation for Post-Training Neural Networks},
  author        = {Ding, Ni and Wang, Shuchang and He, Lei and Li, Shengbo Eben and Li, Keqiang},
  year          = {2026},
  eprint        = {2506.07188},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2506.07188},
  url           = {https://arxiv.org/abs/2506.07188}
}
```
