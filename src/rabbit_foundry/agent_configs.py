from __future__ import annotations

import json


def opencode_config(model_id: str = "rabbit-local", port: int = 8765) -> str:
    return json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "model": f"rabbit/{model_id}",
        "providers": {
            "rabbit": {
                "name": "Rabbit Code Compatibility",
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": f"http://127.0.0.1:{port}/v1"},
                "models": {
                    model_id: {
                        "modelID": model_id,
                        "capabilities": {
                            "tools": False,
                            "input": ["text"],
                            "output": ["text"],
                        },
                        "limit": {"context": 4096, "output": 1024},
                    }
                },
            }
        },
    }, indent=2)


def kilo_config(model_id: str = "rabbit-local", port: int = 8765) -> str:
    return json.dumps({
        "$schema": "https://app.kilo.ai/config.json",
        "model": f"rabbit/{model_id}",
        "provider": {
            "rabbit": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "apiKey": "none",
                    "baseURL": f"http://127.0.0.1:{port}/v1",
                },
                "models": {
                    model_id: {
                        "name": "Rabbit Code Compatibility",
                        "tool_call": False,
                        "limit": {"context": 4096, "output": 1024},
                    }
                },
            }
        },
    }, indent=2)
