# IndicOCR vLLM Acceleration

High-throughput, production-ready inference backend for [bodhan-ai/indic-ocr](https://huggingface.co/bodhan-ai/indic-ocr) using **vLLM** and **CUDA graph capture**, achieving a **~3.7x end-to-end speedup** over Hugging Face transformers.

## Benchmarks (NVIDIA RTX PRO 6000 Ada)

Empirical evaluation over 15 representative document pages (111 image crops, 31,500 characters) across Marathi and English:

| Language | Crops | Identical Crops | Reference Chars | Differing Chars | CER | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Marathi (Devanagari)** | 68 | 67 / 68 | 17,265 | 1 | 0.0058% | **3.79x** |
| **English** | 43 | 40 / 43 | 14,235 | 20 | 0.14% | **3.60x** |
| **Total** | **111** | **107 / 111** | **31,500** | **21** | **0.067%** | **3.67x** |

* Across 17,265 Marathi characters, there was only 1 difference (Visarga `U+0903` vs ASCII colon `:`).
* Battle-tested in production across 3,000+ complex Marathi & English PDF documents.

## Key Optimizations
1. **CUDA Graph Capture (`enforce_eager=False`):** Delivers ~3.7x–3.9x speedup vs ~2.2x in eager mode.
2. **Explicit Sequence Capping (`max_num_seqs=64`):** Fits within Mamba/GDN cache block constraints.
3. **SentencePiece Metaspace Cleanup:** Eliminates `chr(0x2581)` metaspaces to guarantee exact whitespace parity with Hugging Face.

## Quick Start
```python
from vllm_recognizer import VllmRecognizer

# Initialize drop-in backend
recognizer = VllmRecognizer(model_path="bodhan-ai/indic-ocr")


---

### Step 3: Inform Your Manager / Team Lead (1 minute)

Send a quick Slack / Teams message:
> *"Hey, while optimizing our OCR pipeline, we built a generic vLLM wrapper for the open-source Bodhan IndicOCR model that gave us ~3.7x faster inference. I put together an open-source repo with the benchmark and would like to share it on Hugging Face to propose an upstream PR. It contains zero company code or client data."*

---

### Step 4: Post on Hugging Face Discussions (2 minutes)

1. Go to: **[huggingface.co/bodhan-ai/indic-ocr/discussions](https://huggingface.co/bodhan-ai/indic-ocr/discussions)**
2. Click **"New discussion"**.
3. **Title:**
   ```text
   [Proposal / Benchmark] ~3.7x Faster Stage-2 Inference using vLLM Backend (Validated across 3,000+ PDFs)
We have open-sourced the standalone drop-in recognizer along with reproducible benchmarks here:
https://github.com/your-username/indic-ocr-vllm

Would the maintainers be open to a Pull Request integrating this as an optional backend in the official IndicOCR scripts?
