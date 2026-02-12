---
title: "Time2General: Learning Spatiotemporal Invariant Representations for Domain-Generalization Video Semantic Segmentation"
collection: publications
category: PrePrint
permalink: /publication/2026-02-10-paper-title-number-5
excerpt: "<img src='/images_papers/image.png' alt=\"Time2General overview\" />"
date: 2026-02-10
venue: 'arXiv'
paperurl: 'https://arxiv.org/abs/2602.09648'
---

Domain Generalized Video Semantic Segmentation (DGVSS) is trained on a single labeled driving domain and is directly deployed on unseen domains without target labels and test-time adaptation while maintaining temporally consistent predictions over video streams. In practice, both domain shift and temporal-sampling shift break correspondence-based propagation and fixed-stride temporal aggregation, causing severe frame-to-frame flicker even in label-stable regions. We propose Time2General, a DGVSS framework built on Stability Queries. Time2General introduces a Spatio-Temporal Memory Decoder that aggregates multi-frame context into a clip-level spatio-temporal memory and decodes temporally consistent per-frame masks without explicit correspondence propagation. To further suppress flicker and improve robustness to varying sampling rates, the Masked Temporal Consistency Loss is proposed to regularize temporal prediction discrepancies across different strides, and randomize training strides to expose the model to diverse temporal gaps. Extensive experiments on multiple driving benchmarks show that Time2General achieves a substantial improvement in cross-domain accuracy and temporal stability over prior DGSS and VSS baselines while running at up to 18 FPS. Code will be released after the review process.
