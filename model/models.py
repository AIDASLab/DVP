# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from ValueBench (https://github.com/ValueByte-AI/ValueBench).

"""Local chat inference for models supported by the installed vLLM version."""


class VLLMChatModel:
    """Load one chat model, preserving the annotation prompts and sampling options.

    model_dir overrides the model source. engine_kwargs passes optional engine
    settings to vLLM; its model, dtype, and tensor-parallel settings come from the
    explicit arguments. chat_template_kwargs supplies model-specific template
    options. Model compatibility and dtype support are checked by vLLM.

    Select GPUs outside this process (for example, with CUDA_VISIBLE_DEVICES).
    This adapter does not implement remote APIs or a CPU fallback.
    """

    def __init__(
        self,
        model_name,
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
        if device not in {"auto", "cuda"}:
            raise ValueError(
                "The local vLLM adapter accepts device='auto' or 'cuda'; "
                "select GPUs with CUDA_VISIBLE_DEVICES before starting the process."
            )
        source = model_dir if model_dir is not None else model_name
        if source is None or not str(source).strip():
            raise ValueError("Provide a model ID or local model directory.")
        if tensor_parallel_size is None:
            tensor_parallel_size = 1
        if int(tensor_parallel_size) < 1:
            raise ValueError("tensor_parallel_size must be positive.")

        engine_options = dict(engine_kwargs or {})
        reserved = {"model", "dtype", "tensor_parallel_size"} & engine_options.keys()
        if reserved:
            raise ValueError("Use the explicit arguments for: " + ", ".join(sorted(reserved)))
        template_options = dict(chat_template_kwargs or {})
        reserved = {"tokenize", "add_generation_prompt"} & template_options.keys()
        if reserved:
            raise ValueError("Chat template options cannot override: " + ", ".join(sorted(reserved)))

        # Keep short structured outputs for named Qwen3 checkpoints. Other model
        # families receive no Qwen options or hand-written control tokens.
        if str(source).rstrip("/").rsplit("/", 1)[-1].lower().startswith("qwen3"):
            template_options.setdefault("enable_thinking", False)

        from vllm import LLM, SamplingParams

        self.model_name = str(source)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt if system_prompt is not None else "You are a helpful assistant."
        self.chat_template_kwargs = template_options
        self._sampling_params = SamplingParams
        self.model = LLM(
            model=self.model_name,
            dtype=dtype,
            tensor_parallel_size=int(tensor_parallel_size),
            **engine_options,
        )
        self.tokenizer = self.model.get_tokenizer()

    def predict(self, input_text, **kwargs):
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        sampling_options = dict(kwargs)
        sampling_options.setdefault("temperature", self.temperature)
        sampling_options.setdefault("max_tokens", self.max_new_tokens)
        if self.top_p is not None:
            sampling_options.setdefault("top_p", self.top_p)
        sampling_params = self._sampling_params(**sampling_options)
        outputs = self.model.generate(prompt, sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()
