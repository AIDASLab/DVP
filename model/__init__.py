# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from ValueBench (https://github.com/ValueByte-AI/ValueBench).

"""Small, lazily loaded local-model interface for the annotation scripts."""

from .models import VLLMChatModel

__all__ = ["LLMModel"]


class LLMModel:
    """Accept a compatible Hugging Face model ID or a local chat-model directory.

    Calls return one stripped string per input prompt, in input order. The
    underlying adapter's system_prompt can be changed between parser stages
    without loading another model.
    """

    def __init__(
        self,
        model,
        max_new_tokens=20,
        temperature=0,
        device="cuda",
        dtype="auto",
        model_dir=None,
        system_prompt=None,
        tensor_parallel_size=1,
        top_p=1.0,
        chat_template_kwargs=None,
        engine_kwargs=None,
    ):
        self.model_name = model
        self.tensor_parallel_size = tensor_parallel_size
        self.model = VLLMChatModel(
            model_name=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            model_dir=model_dir,
            system_prompt=system_prompt,
            tensor_parallel_size=tensor_parallel_size,
            top_p=top_p,
            chat_template_kwargs=chat_template_kwargs,
            engine_kwargs=engine_kwargs,
        )

    def __call__(self, input_texts, **kwargs):
        if isinstance(input_texts, str):
            raise TypeError("Pass a sequence of prompts, not a single string.")
        return [self.model.predict(text, **kwargs) for text in input_texts]
