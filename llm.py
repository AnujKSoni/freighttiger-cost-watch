"""Minimal Groq client (OpenAI-compatible REST) with fixed sampling and token accounting."""
import json
import os
import time

import requests

MODEL = "openai/gpt-oss-120b"                      # open-weight (Apache-2.0), Groq free tier
PRICE_USD_PER_M = {"input": 0.15, "output": 0.60}  # Groq published rate for this model
URL = "https://api.groq.com/openai/v1/chat/completions"
SAMPLING = {"temperature": 0, "seed": 7, "reasoning_effort": "low"}


class LLM:
    def __init__(self, api_key=None):
        self.api_key = os.environ.get("GROQ_API_KEY", "") if api_key is None else api_key
        self.calls = self.input_tokens = self.output_tokens = self.reasoning_tokens = 0

    @property
    def available(self):
        return bool(self.api_key)

    def json(self, system, user):
        return json.loads(self._chat(system, user, {"response_format": {"type": "json_object"}}))

    def text(self, system, user):
        return self._chat(system, user, {})

    def _chat(self, system, user, extra):
        body = {"model": MODEL, **SAMPLING, **extra,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        for attempt in range(4):
            r = requests.post(URL, json=body, timeout=120, headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code != 429 or attempt == 3:
                break
            time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))  # free-tier rate limit
        r.raise_for_status()
        data = r.json()
        usage = data["usage"]
        self.calls += 1
        self.input_tokens += usage["prompt_tokens"]
        self.output_tokens += usage["completion_tokens"]
        self.reasoning_tokens += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        return data["choices"][0]["message"]["content"]

    def cost_log(self):
        usd = (self.input_tokens * PRICE_USD_PER_M["input"] + self.output_tokens * PRICE_USD_PER_M["output"]) / 1e6
        return {"model": MODEL, "sampling": SAMPLING, "llm_calls": self.calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "of_which_reasoning_tokens": self.reasoning_tokens,
                "rate_usd_per_million_tokens": PRICE_USD_PER_M, "cost_usd_at_published_rate": round(usd, 6)}
