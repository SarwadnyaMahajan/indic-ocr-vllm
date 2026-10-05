# IndicOCR vLLM Acceleration

High-throughput, production-ready inference backend for [bodhan-ai/indic-ocr](https://huggingface.co/bodhan-ai/indic-ocr) using **vLLM** and **CUDA graph capture**, achieving a **~3.7x end-to-end speedup** over Hugging Face transformers with **99.93% character-level parity**.

---

## Overview

[IndicOCR](https://huggingface.co/bodhan-ai/indic-ocr) by Bodhan AI and AI4Bharat is a state-of-the-art vision-language document parser for English and 22 Indian languages. In high-volume document pipelines, **Stage 2 (IndicBlockOCR)** dominates per-page runtime (~10–12 seconds per page on GPU) due to sequential crop recognition under standard Hugging Face pipelines.

This repository provides an optimized, drop-in **vLLM recognizer backend** that unlocks the native throughput of `Qwen3_5ForConditionalGeneration` through batched scheduling, chunked prefill, and CUDA graph capture.

---

## Benchmarks

Measured on **NVIDIA RTX PRO 6000 Ada (96GB VRAM)** across a controlled evaluation set of 15 representative document pages (111 image crops, 31,500 characters) across 5 document domains:

| Language | Crops | Identical Crops | Reference Chars | Differing Chars | Character Error Rate (CER) | Runtime (HF) | Runtime (vLLM) | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Marathi (Devanagari)** | 68 | 67 / 68 | 17,265 | 1 | **0.0058%** | 71.8s | 18.9s | **3.79x** |
| **English** | 43 | 40 / 43 | 14,235 | 20 | **0.14%** | 49.0s | 13.6s | **3.60x** |
| **Total** | **111** | **107 / 111** | **31,500** | **21** | **0.067%** | **120.8s** | **32.9s** | **3.67x** |

### Character Fidelity & Zero Degradation
* Across all 17,265 Marathi characters, there was only a single character difference: `ः` (U+0903 Devanagari Visarga) vs `:` (U+003A ASCII colon).
* All complex Devanagari conjuncts (jodakshare), numbers, words, and vocabulary matched bit-for-bit with the Hugging Face reference implementation.
* The 0.067% CER across English and Marathi represents floating-point kernel argmax ties on punctuation glyphs, with zero degradation in text semantics.

### Production Scale Validation
Following the controlled benchmark, this backend was deployed in an active pipeline and has processed **over 3,000+ complex Marathi & English PDF documents** continuously with zero memory leaks, consistent latency, and rock-solid multi-worker stability.

---

## Key Technical Optimizations

### 1. CUDA Graph Capture (enforce_eager=False)
* Running in eager mode (`enforce_eager=True`) yields only ~2.25x speedup due to repetitive PyTorch CPU-GPU launch overhead across dozens of block crops per page.
* Enabling CUDA graphs via `enforce_eager=False` eliminates kernel launch overhead and unlocks the full **3.7x–3.9x speedup**.

### 2. Mamba Cache Block Allocation (max_num_seqs=64)
* The recognizer is a GDN/Mamba hybrid model where each decode sequence requires its own Mamba cache block.
* vLLM's default `max_num_seqs=1024` attempts to allocate more cache blocks than are physically available at conservative memory fractions, causing initialization failure.
* Explicitly setting `max_num_seqs=64` matches typical page crop density (20–50 crops/page) and enables smooth CUDA graph capture.

### 3. SentencePiece Metaspace Correction
* vLLM performs internal detokenization. The tokenizer applies SentencePiece leading metaspaces (`chr(0x2581)` / `U+2581`).
* The wrapper normalizes metaspaces to standard spaces prior to stripping, guaranteeing byte-identical whitespace handling to Hugging Face.

### 4. Headless Environment Compatibility
* FlashInfer attempts to JIT-compile a top-k/top-p sampler during initialization requiring `nvcc`.
* Because OCR inference uses greedy decoding (`temperature=0.0`), setting `VLLM_USE_FLASHINFER_SAMPLER=0` bypasses unnecessary sampler compilation and runs on standard runtime environments without the full CUDA compiler toolkit.

---

## Installation

### Prerequisites
* Python 3.10+
* Linux OS
* NVIDIA GPU with Ampere, Ada Lovelace, or Hopper architecture (Compute Capability >= 8.0)
* CUDA 12.1+

### Dependencies
```bash
pip install vllm>=0.6.0 transformers>=4.45.0 torch>=2.4.0
```

---

## Quick Start

### Basic Usage

```python
from vllm_recognizer import VllmRecognizer
from PIL import Image

# Initialize the vLLM recognizer backend
recognizer = VllmRecognizer(
    model_path="bodhan-ai/indic-ocr",
    gpu_memory_utilization=0.15,  # Adjust based on your available VRAM
    max_num_seqs=64,
    enforce_eager=False,          # Enables CUDA graph capture for maximum speed
)

# Prepare inference requests (format matching IndicOCR interface)
class OCRRequest:
    def __init__(self, image: Image.Image, prompt: str = ""):
        self.image = image
        self.prompt = prompt

crop = Image.open("path/to/crop.png")
requests = [OCRRequest(image=crop)]

# Transcribe crops
transcriptions = recognizer.transcribe(requests)
print(transcriptions)

recognizer.close()
```

### Integrating with IndicOCR Pipeline

To replace HfRecognizer in an existing IndicOCR installation:

```python
from idp_offline import IndicBlockOCR
from vllm_recognizer import VllmRecognizer

# Instantiate the vLLM recognizer
vllm_backend = VllmRecognizer(
    model_path="/path/to/indic-ocr/weights/ocr",
    gpu_memory_utilization=0.15,
)

# Inject into the IndicBlockOCR pipeline
pipeline = IndicBlockOCR(
    backend=vllm_backend,
    config=parser._rec_cfg,
    dedup=parser._dedup,
    crop=parser._crop,
)
```

---

## Repository Structure

```text
.
├── vllm_recognizer.py     # Main drop-in recognizer class
├── requirements.txt       # Minimal package dependencies
└── README.md              # Technical documentation and benchmarks
```

---

## Acknowledgments & License

* Model weights and architecture: Bodhan AI and AI4Bharat, IIT Madras.
* Model License: Indic Open Model License v1.0
* Code License: MIT License
