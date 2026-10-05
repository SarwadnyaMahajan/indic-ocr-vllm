"""
vLLM Recognizer Backend for Bodhan-AI IndicOCR (Stage 2).

Replaces the default Hugging Face transformers pipeline with vLLM
to achieve ~3.7x faster inference on NVIDIA GPUs via batched scheduling
and CUDA graph capture.
"""

import os
import logging
from typing import List, Any

logger = logging.getLogger(__name__)

# <|im_end|> token id in the Qwen vocabulary.
# Explicitly declared so generation halts immediately once text recognition is complete.
EOS_TOKEN_ID = 262146

# SentencePiece leading metaspace character (Unicode U+2581: " ").
# vLLM detokenizes internally; replacing this character preserves exact
# whitespace parity with the Hugging Face reference tokenizer.
METASPACE = chr(0x2581)


class VllmRecognizer:
    """Drop-in Stage-2 Recognizer for IndicOCR using vLLM."""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 8192,
        max_num_seqs: int = 64,
        gpu_memory_utilization: float = 0.15,
        enforce_eager: bool = False,
    ):
        """
        Initialize the vLLM engine for IndicBlockOCR.

        Args:
            model_path: Local path or Hugging Face repo ID (e.g., 'bodhan-ai/indic-ocr').
            max_model_len: Context window limit (default: 8192).
            max_num_seqs: Maximum number of concurrent sequences in a batch.
                          Must be explicitly set for hybrid Mamba architectures
                          to prevent cache block allocation errors (default: 64).
            gpu_memory_utilization: Fraction of total GPU VRAM reserved by vLLM (default: 0.15).
            enforce_eager: Set to False to enable CUDA graph capture, unlocking
                           the full ~3.7x speedup over eager execution (default: False).
        """
        # FlashInfer attempts to JIT-compile a top-k/top-p sampler during initialization
        # which requires nvcc. Since OCR uses greedy decoding (temperature=0.0), this sampler
        # is never used; setting this flag to "0" avoids unnecessary compilation errors.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        
        logger.info(
            "Initializing vLLM (gpu_mem=%.2f, max_seqs=%d, eager=%s)",
            gpu_memory_utilization, max_num_seqs, enforce_eager
        )

        # Initialize the vLLM engine instance
        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": 1},  # Each text block request contains exactly 1 crop image
            enforce_eager=enforce_eager,        # False = Enables CUDA graph capture for maximum throughput
        )

        # Configure greedy sampling for bit-deterministic recognition
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop_token_ids=[EOS_TOKEN_ID],
        )

    def _prompt(self, text: str) -> str:
        """
        Format the prompt using the model's official chat template.
        The template inserts <|image_pad|> which vLLM automatically resolves
        against the provided image dimensions.
        """
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def transcribe(self, requests: List[Any]) -> List[str]:
        """
        Transcribe a batch of cropped block images in parallel.

        Args:
            requests: List of request objects, each having:
                      - `image`: PIL.Image of the cropped block
                      - `prompt`: Text instruction or empty string

        Returns:
            List of transcribed strings in the exact order of the input requests.
        """
        if not requests:
            return []

        # Package requests into vLLM multi-modal inputs
        inputs = [
            {"prompt": self._prompt(r.prompt), "multi_modal_data": {"image": r.image}}
            for r in requests
        ]

        # Execute batched generation with CUDA graphs
        outputs = self.llm.generate(inputs, self.sampling, use_tqdm=False)

        # Normalize SentencePiece metaspaces to standard spaces and trim whitespace
        return [o.outputs[0].text.replace(METASPACE, " ").strip() for o in outputs]

    def close(self):
        """Clean up resources if needed."""
        pass
