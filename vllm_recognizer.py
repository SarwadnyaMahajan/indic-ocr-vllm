import os
import logging
from typing import List, Any

logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 262146
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
        # Disable FlashInfer sampler compilation when decoding greedily
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        
        logger.info(
            "Initializing vLLM (gpu_mem=%.2f, max_seqs=%d, eager=%s)",
            gpu_memory_utilization, max_num_seqs, enforce_eager
        )
        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=enforce_eager,
        )
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop_token_ids=[EOS_TOKEN_ID],
        )

    def _prompt(self, text: str) -> str:
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def transcribe(self, requests: List[Any]) -> List[str]:
        if not requests:
            return []
        inputs = [
            {"prompt": self._prompt(r.prompt), "multi_modal_data": {"image": r.image}}
            for r in requests
        ]
        outputs = self.llm.generate(inputs, self.sampling, use_tqdm=False)
        return [o.outputs[0].text.replace(METASPACE, " ").strip() for o in outputs]

    def close(self):
        pass
