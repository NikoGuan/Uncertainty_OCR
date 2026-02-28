# Uncertainty-Aware OCR with GRPO

This repository contains the core code of uncertainty-aware OCR. The system trains OCR models to use `<C>...</C>` markers to indicate uncertain regions instead of forcing potentially incorrect outputs.

## Files

- **ocr_gt_align.py**: Implements pseudo-labeled cold start by aligning OCR predictions with ground truth and inserting uncertainty tags around detected errors (character or word level).

- **ocr_reward_word_level.py**: Computes the uncertainty-aware reward function combining transcription accuracy and tagging quality for GRPO training.

- **data_sample/**: Contains sample degraded images with corresponding ground truth texts for testing.

## Quick Start

```python
# Test alignment and tagging
python ocr_gt_align.py

# Test reward computation
python ocr_reward_word_level.py
```

Both files contain example usage in their `main()` functions demonstrating the core functionality.

## Concepts

The system prevents reward hacking through length-mismatch damping and ensures that adding errors always reduces the total reward even with perfect tagging.