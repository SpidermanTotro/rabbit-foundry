"""
Connect Alpha Rabbit / OpenAI-compatible clients to local DreamRabbit server.
"""
import os
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path


def generate_alpha_rabbit_config(
    server_url: str = "http://localhost:8080/v1",
    model_name: str = "dreamrabbit",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate configuration for Alpha Rabbit to connect to local DreamRabbit.
    
    Alpha Rabbit (and most agent frameworks) use OpenAI-compatible endpoints.
    """
    config = {
        "model": model_name,
        "base_url": server_url,
        "api_key": "not-needed",  # Local server doesn't require auth
        "timeout": 120,
        "max_retries": 3,
    }
    
    if output_path:
        Path(output_path).write_text(json.dumps(config, indent=2))
        print(f"Config written to {output_path}")
        
    return config


def generate_kilo_config(
    server_url: str = "http://localhost:8080/v1",
    model_name: str = "dreamrabbit",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Generate Kilo configuration with local DreamRabbit provider."""
    config = {
        "$schema": "https://app.kilo.ai/config.json",
        "model": f"local-dreamrabbit/{model_name}",
        "small_model": f"local-dreamrabbit/{model_name}",
        "provider": {
            "local-dreamrabbit": {
                "options": {
                    "baseURL": server_url,
                    "apiKey": "not-needed"
                },
                "models": {
                    model_name: {"name": "DreamRabbit Local"}
                }
            }
        },
        "enabled_providers": ["local-dreamrabbit"],
        "default_agent": "code",
    }
    
    if output_path:
        Path(output_path).write_text(json.dumps(config, indent=2))
        print(f"Kilo config written to {output_path}")
        
    return config


def generate_openrouter_config(
    server_url: str = "http://localhost:8080/v1",
    model_name: str = "dreamrabbit",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Generate OpenRouter-compatible configuration."""
    config = {
        "name": "DreamRabbit Local",
        "model": model_name,
        "endpoint": server_url,
        "api_key": "local",
        "context_length": 4096,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
    }
    
    if output_path:
        Path(output_path).write_text(json.dumps(config, indent=2))
        print(f"OpenRouter config written to {output_path}")
        
    return config


def test_connection(server_url: str = "http://localhost:8080/v1", model_name: str = "dreamrabbit"):
    """Test connection to DreamRabbit server."""
    import requests
    
    try:
        # Test models endpoint
        resp = requests.get(f"{server_url}/models", timeout=5)
        print(f"Models endpoint: {resp.status_code}")
        print(f"Available models: {resp.json()}")
        
        # Test chat completion
        resp = requests.post(
            f"{server_url}/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello, DreamRabbit!"}],
                "max_tokens": 50,
                "temperature": 0.7,
            },
            timeout=30
        )
        print(f"Chat endpoint: {resp.status_code}")
        result = resp.json()
        print(f"Response: {result['choices'][0]['message']['content']}")
        
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate configs for connecting to DreamRabbit server")
    parser.add_argument("--url", default="http://localhost:8080/v1", help="Server URL")
    parser.add_argument("--model", default="dreamrabbit", help="Model name")
    parser.add_argument("--test", action="store_true", help="Test connection")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    if args.test:
        test_connection(args.url, args.model)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        generate_alpha_rabbit_config(args.url, args.model, f"{args.output_dir}/alpha_rabbit_config.json")
        generate_kilo_config(args.url, args.model, f"{args.output_dir}/kilo.jsonc")
        generate_openrouter_config(args.url, args.model, f"{args.output_dir}/openrouter_config.json")
        print("\nConfigs generated. To use with Alpha Rabbit:")
        print("  Set ALPHA_RABBIT_MODEL_ENDPOINT=http://localhost:8080/v1")
        print("  Set ALPHA_RABBIT_MODEL=dreamrabbit")