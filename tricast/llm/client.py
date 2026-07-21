"""Provider-agnostic structured-output LLM calls over a Pydantic schema.

Shared by the analyst (produces analyses) and the judge (scores analyses).
Ollama grammar-constrains JSON via `format`; Anthropic uses messages.parse.
Returns (parsed_model, model_label, cost_usd).
"""

import json
from typing import TypeVar

import requests
from pydantic import BaseModel

from tricast import config

T = TypeVar("T", bound=BaseModel)


def ollama_structured(system: str, payload: dict, schema: type[T],
                      model: str | None = None) -> tuple[T, str, float]:
    model = model or config.OLLAMA_MODEL
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "stream": False,
                "format": schema.model_json_schema(),
                "options": {"num_predict": config.MAX_TOKENS},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, sort_keys=True)},
                ],
            },
            timeout=600,  # local models can be slow on long prompts
        )
        resp.raise_for_status()
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_HOST}. Start it (`ollama serve`) "
            "or set OLLAMA_HOST in .env if it runs on another machine."
        ) from e
    content = resp.json()["message"]["content"]
    return schema.model_validate_json(content), f"ollama/{model}", 0.0


def anthropic_structured(system: str, payload: dict, schema: type[T],
                         model: str | None = None) -> tuple[T, str, float]:
    import anthropic  # deferred: only needed when this provider is selected

    model = model or config.MODEL_ID
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=config.MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, sort_keys=True)}],
        output_format=schema,
    )
    usage = response.usage
    # Sonnet 4.6 pricing: $3/M input, $15/M output
    cost = round(usage.input_tokens * 3e-6 + usage.output_tokens * 15e-6, 4)
    return response.parsed_output, model, cost


def structured_call(provider: str, system: str, payload: dict, schema: type[T],
                    model: str | None = None) -> tuple[T, str, float]:
    if provider == "ollama":
        return ollama_structured(system, payload, schema, model)
    if provider == "anthropic":
        return anthropic_structured(system, payload, schema, model)
    raise ValueError(f"Unknown provider {provider!r} (use 'ollama' or 'anthropic')")
