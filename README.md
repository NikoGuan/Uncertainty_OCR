# Teaching VLMs to Admit Uncertainty in OCR from Lossy Visual Inputs

This project provides the core components of an uncertainty-aware OCR pipeline. Instead of forcing potentially incorrect outputs on lossy/degraded inputs, the model is trained to explicitly mark uncertain spans using **`<C>...</C>`**.

---

## Dataset: Blur-OCR

The paper introduces **Blur-OCR**, a benchmark for uncertainty-aware OCR under lossy visual conditions:

Hugging Face: https://huggingface.co/datasets/ShuhaoGuan/Blur-OCR

---

## What's inside

- **`ocr_gt_align.py`**  
  Pseudo-labeled cold start via alignment between OCR predictions and ground truth. It detects mismatches and inserts uncertainty tags around error regions (character-level or word-level).

- **`ocr_reward_word_level.py`**  
  Uncertainty-aware reward for GRPO training (word-level). It combines transcription accuracy with tagging quality and includes defenses against reward hacking (e.g., length-mismatch damping).

---

If you find our paper useful, please cite:

```bibtex
@inproceedings{guan2026teach_vlms_uncertainty_ocr,
  title     = {Teaching VLMs to Admit Uncertainty in OCR from Lossy Visual Inputs},
  author    = {Shuhao Guan and Moule Lin and Cheng Xu and Jinman Zhao and Derek Greene},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=zyCjizqOxB}
}
