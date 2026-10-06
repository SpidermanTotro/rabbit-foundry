"""
Local Inference Server for DreamRabbit - OpenAI-compatible API.
"""
import json
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import torch


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    max_tokens: int = 1024
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(__import__('time').time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(__import__('time').time()))
    model: str
    choices: List[Dict[str, Any]]


@dataclass
class ModelState:
    model: Any = None
    tokenizer: Any = None
    config: Any = None
    device: str = "cpu"


class DreamRabbitServer:
    """OpenAI-compatible inference server for DreamRabbit."""
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, device: str = "auto"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = self._resolve_device(device)
        self.state = ModelState()
        self.app = FastAPI(title="DreamRabbit Inference Server")
        self._setup_routes()
        
    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    def load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from src.model import DreamRabbit, get_tiny_config
        from src.config import DreamRabbitConfig
        import torch
        
        print(f"Loading model from {self.model_path} on {self.device}...")
        
        # Load tokenizer
        self.state.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.state.tokenizer.pad_token is None:
            self.state.tokenizer.pad_token = self.state.tokenizer.eos_token
        
        # Load config
        if self.model_path.endswith('.json'):
            with open(self.model_path) as f:
                config_dict = json.load(f)
            config = DreamRabbitConfig(**config_dict)
        else:
            # Default to tiny config for now
            config = get_tiny_config()
            
        # Create model
        self.state.model = DreamRabbit(config)
        self.state.model.to(self.device)
        self.state.model.eval()
        
        # Load weights if available
        if self.model_path.endswith('.pt') or self.model_path.endswith('.pth') or self.model_path.endswith('.bin'):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.state.model.load_state_dict(state_dict)
            
        self.state.config = config
        print(f"Model loaded: {sum(p.numel() for p in self.state.model.parameters()):,} params")
        
    def _setup_routes(self):
        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [{
                    "id": "dreamrabbit",
                    "object": "model",
                    "created": int(__import__('time').time()),
                    "owned_by": "rabbit-foundry",
                }]
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest, http_request: Request):
            if self.state.model is None:
                raise HTTPException(503, "Model not loaded")
                
            if request.stream:
                return StreamingResponse(
                    self._stream_chat(request),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            return await self._generate_chat(request)
    
    async def _generate_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate non-streaming completion."""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(request.messages)
        
        # Tokenize
        inputs = self.state.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.state.model.generate(
                inputs["input_ids"],
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )
        
        # Decode
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.state.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Apply stop sequences
        if request.stop:
            for stop_seq in request.stop:
                if stop_seq in text:
                    text = text[:text.index(stop_seq)]
        
        return ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(generated),
                "total_tokens": inputs["input_ids"].shape[1] + len(generated)
            }
        )
    
    async def _stream_chat(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Stream chat completion."""
        prompt = self._messages_to_prompt(request.messages)
        inputs = self.state.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        # For streaming, generate token by token
        generated_ids = []
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            for _ in range(request.max_tokens):
                outputs = self.state.model(input_ids)
                next_token_logits = outputs["lm_logits"][:, -1, :] / request.temperature
                
                # Top-k sampling
                if request.top_k > 0:
                    topk_vals, topk_idx = torch.topk(next_token_logits, request.top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(-1, topk_idx, topk_vals)
                
                # Top-p sampling
                if request.top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True)
                    cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_idx_to_remove = cumprobs > request.top_p
                    sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                    sorted_idx_to_remove[..., 0] = 0
                    idx_to_remove = sorted_idx_to_remove.scatter(1, sorted_idx, sorted_idx_to_remove)
                    next_token_logits[idx_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids.append(next_token.item())
                
                # Check stop
                if request.stop:
                    current_text = self.state.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    if any(stop in current_text for stop in request.stop):
                        break
                
                # Yield chunk
                chunk = ChatCompletionChunk(
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {"content": self.state.tokenizer.decode([next_token.item()], skip_special_tokens=True)},
                        "finish_reason": None
                    }]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                
                # Append for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Final chunk
        final_chunk = ChatCompletionChunk(
            model=request.model,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to prompt string."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"<|system|>\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"<|user|>\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"<|assistant|>\n{msg.content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the server."""
        self.load_model()
        uvicorn.run(self.app, host=host, port=port)


def create_server(model_path: str, tokenizer_path: Optional[str] = None, device: str = "auto") -> DreamRabbitServer:
    """Factory function to create inference server."""
    return DreamRabbitServer(model_path, tokenizer_path, device)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/dreamrabbit-tiny"
    server = create_server(model_path)
    server.run()