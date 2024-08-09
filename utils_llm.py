import os
import logging

from openai import OpenAI
from http import HTTPStatus
import requests
import dashscope

import tiktoken
from transformers import pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def is_openai_model(model_name: str) -> bool:
#     return model_name and ("gpt-3.5" in model_name or "gpt-4" in model_name)

# def check_openai_api():
#     if not os.getenv("OPENAI_API_KEY"):
#         raise ValueError("Please set the OPENAI_API_KEY environment variable. See https://platform.openai.com/docs/quickstart")


class LLM:
    def __init__(
        self, 
        model_name: str, 
        use_api: bool = False,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.use_api = use_api
            
        if not use_api:
            self.generator = pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": 'auto'}, device=device)


    def count_tokens(self, text: str, encoding='cl100k_base') -> int:
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
        

    def generate(
        self,
        prompt: str,
        do_sample: bool = True,
        temperature: float = None,
        max_tokens: int = 300,
        seed: int = 42
    ) -> str:
        # local model
        if not self.use_api:
            return self._generate_hf(prompt, do_sample, temperature, max_tokens)
        # openai api
        if 'gpt' in self.model_name.lower():
            return self._generate_openai_api(prompt, do_sample, temperature, max_tokens, seed)
        # qwen api
        elif 'qwen' in self.model_name.lower() or 'llama' in self.model_name.lower():
            return self._generate_dashscope_api(prompt, do_sample, temperature, max_tokens, seed)
        # deepbricks api
        # elif 'llama' in self.model_name.lower():
        #     return self._generate_deepbricks_api(prompt, do_sample, temperature, max_tokens, seed)
        else:
            return NotImplementedError

    def _generate_hf(
        self,
        prompt: str,
        do_sample: bool = True,
        temperature: float = None,
        max_tokens: int = 300,
    ) -> str:
        gen_kwargs = {"do_sample": do_sample, "max_tokens": max_tokens}
        if temperature:
            gen_kwargs["temperature"] = temperature

        res = self.generator(prompt, return_tensors=True, **gen_kwargs)
        generated_text = res['generated_text'][len(prompt):].strip()

        total_tokens = len(res['generated_token_ids'])
        prompt_tokens = len(self.generator.tokenizer(prompt)['input_ids'])
        completion_tokens = total_tokens - prompt_tokens
        usage = {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens
        }

        return generated_text
    
    def _generate_openai_api(
        self,
        prompt: str,
        do_sample: bool = True,
        temperature: float = None,
        max_tokens: int = 300,
        seed: int = 42
    ) -> str:
        self.client = OpenAI()
        if not do_sample and temperature != 0:
            temperature = 0
            logger.warning("`do_sample` is False, setting temperature to 0 for openai api.")
        gen_kwargs = {"max_tokens": max_tokens}
        if temperature:
            gen_kwargs["temperature"] = temperature

        messages = [{"role": "user", "content": prompt}]

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            seed=seed,
            **gen_kwargs
        )

        return res.choices[0].message.content
    
    def _generate_dashscope_api(
        self,
        prompt: str,
        do_sample: bool = True,
        temperature: float = 1,
        max_tokens: int = 300,
        seed: int = 42
    ) -> str:
        if not do_sample and temperature != 0:
            temperature = 0
            logger.warning("`do_sample` is False, setting temperature to 0 for openai api.")
        gen_kwargs = {"max_tokens": max_tokens}
        if temperature:
            gen_kwargs["temperature"] = temperature

        messages = [{"role": "user", "content": prompt}]

        res = dashscope.Generation.call(
            model=self.model_name,
            messages=messages,
            seed=seed,
            **gen_kwargs
        )

        if res.status_code != HTTPStatus.OK:
            raise RuntimeError('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                res.request_id, res.status_code, res.code, res.message
            ))

        return res.output.text
    
    def _generate_deepbricks_api(
        self,
        prompt: str,
        do_sample: bool = True,
        temperature: float = 1,
        max_tokens: int = 300,
        seed: int = 42
    ) -> str:
        DEEPBRICKS_API_KEY = os.environ.get("DEEPBRICKS_API_KEY")
        if not DEEPBRICKS_API_KEY:
            raise ValueError("Please set the DEEPBRICKS_API_KEY environment variable.")
        messages = [{"role": "user", "content": prompt}]

        url = "https://api.deepbricks.ai/v1/chat/completions"
        body = {
            "model": self.model_name,
            "messages": messages,
        }
        res = requests.post(url, headers={"Authorization": f"Bearer {DEEPBRICKS_API_KEY}"}, json=body)
        if res.status_code != HTTPStatus.OK:
            raise RuntimeError(res.json())
        return res.json()['choices'][0]['message']['content']
        

