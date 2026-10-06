"""
DreamRabbit - Main model integrating all components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from config import DreamRabbitConfig, get_default_config
from recursive_core import RecursiveCore
from experts import ExpertPool, MoELayer
from verifier import NeuralVerifier, VerifierLoss
from memory import CompressedMemory, WorkingMemory


class DreamRabbit(nn.Module):
    """
    DreamRabbit: Adaptive Recursive Model
    
    Architecture:
    Input → Encoder → RECURSIVE CORE → Router → Expert Pool
                      ↑      │          │
                      │      ↓          ↓
                      │   Verifier   Tool/Code Head
                      │      │          │
                      └──── retry ←─────┘
                            │
                         Output
    """
    
    def __init__(self, config: DreamRabbitConfig = None):
        super().__init__()
        self.config = config or get_default_config()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.core.d_model)
        self.position_embedding = nn.Embedding(self.config.max_seq_len, self.config.core.d_model)
        self.dropout = nn.Dropout(self.config.core.dropout)
        
        # Recursive core
        self.recursive_core = RecursiveCore(self.config.core)
        
        # Expert pool and MoE layer
        # Update expert configs with core's d_model
        for expert_config in self.config.experts:
            expert_config.d_model = self.config.core.d_model
        self.expert_pool = ExpertPool(self.config.experts)
        # Update router config with core's d_model
        router_config = self.config.router
        router_config.d_model = self.config.core.d_model
        router_config.n_experts = len(self.config.experts)
        self.moe_layer = MoELayer(self.expert_pool, router_config)
        
        # Neural verifier
        # Use core's d_model for verifier
        verifier_config = self.config.verifier
        verifier_config.d_model = self.config.core.d_model
        self.verifier = NeuralVerifier(verifier_config)
        
        # Memory systems
        # Use core's d_model for memory
        memory_config = self.config.memory
        memory_config.d_model = self.config.core.d_model
        self.episodic_memory = CompressedMemory(memory_config)
        self.working_memory = WorkingMemory(self.config.core.d_model)
        
        # Output heads
        self.lm_head = nn.Linear(self.config.core.d_model, self.config.vocab_size, bias=False)
        if self.config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
            
        # Tool/code head for agent actions
        self.tool_head = nn.Sequential(
            nn.Linear(self.config.core.d_model, self.config.core.d_model),
            nn.GELU(),
            nn.Linear(self.config.core.d_model, 256),  # Tool/action vocabulary
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_verification: bool = False,
        return_routing_info: bool = False,
        max_recursion_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through DreamRabbit.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
            return_verification: Whether to return verification signals
            return_routing_info: Whether to return MoE routing info
            max_recursion_depth: Override max recursion depth
            
        Returns:
            Dict with logits, verification, routing_info, etc.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Add working memory context
        wm_context = self.working_memory.get_context()
        if wm_context.shape[1] > 0:
            # Prepend working memory to sequence
            x = torch.cat([wm_context.expand(batch_size, -1, -1), x], dim=1)
            if attention_mask is not None:
                wm_mask = torch.ones(batch_size, wm_context.shape[1], device=device)
                attention_mask = torch.cat([wm_mask, attention_mask], dim=1)
        
        # Actual sequence length after working memory
        actual_seq_len = x.shape[1]
        
        # Create causal mask
        if attention_mask is not None:
            causal_mask = torch.tril(torch.ones(actual_seq_len, actual_seq_len, device=device))
            full_mask = attention_mask.unsqueeze(1) * causal_mask.unsqueeze(0)
        else:
            causal_mask = torch.tril(torch.ones(actual_seq_len, actual_seq_len, device=device))
            full_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Recursive core with dynamic depth
        core_output, depths = self.recursive_core(x, full_mask)
        
        # Update working memory with core output
        self.working_memory.update(core_output[:, -seq_len:])
        
        # Verification
        verification = self.verifier(core_output[:, -seq_len:])
        
        # Determine if we should retry (recurse more)
        should_retry = self.verifier.should_retry(verification)
        
        # Route to experts
        moe_output, router_logits, routing_info = self.moe_layer(core_output[:, -seq_len:])
        
        # Combine core output with MoE output
        combined = core_output[:, -seq_len:] + moe_output
        
        # Language modeling head
        lm_logits = self.lm_head(combined)
        
        # Tool head
        tool_logits = self.tool_head(combined)
        
        # Prepare outputs
        outputs = {
            "lm_logits": lm_logits,
            "tool_logits": tool_logits,
            "hidden_states": combined,
            "core_output": core_output,
            "recursion_depths": depths,
            "verification": verification,
            "should_retry": should_retry,
        }
        
        if return_routing_info:
            outputs["routing_info"] = routing_info
            outputs["router_logits"] = router_logits
            
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token_ids: Optional[List[int]] = None,
        use_verifier: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens with optional verifier-guided recursion.
        """
        self.eval()
        device = input_ids.device
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids, return_verification=use_verifier)
                
                # Get next token logits
                next_token_logits = outputs["lm_logits"][:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_values)
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check stop conditions
                if stop_token_ids and next_token.item() in stop_token_ids:
                    break
                    
                # Check verifier retry signal
                if use_verifier and outputs["should_retry"].any():
                    # Could trigger additional recursion here
                    pass
        
        return input_ids
    
    def store_experience(self, hidden_states: torch.Tensor, importance: Optional[torch.Tensor] = None):
        """Store experience in episodic memory."""
        # Compress and store
        compressed = self.episodic_memory.compress(hidden_states.mean(dim=1))
        self.episodic_memory.write(compressed, hidden_states.mean(dim=1), importance)
    
    def retrieve_memory(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve relevant memories for query."""
        return self.episodic_memory(query)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_active_params(self) -> int:
        """Estimate active parameters per forward pass."""
        # Core params (always active)
        core_params = sum(p.numel() for p in self.recursive_core.parameters())
        # Embedding params
        embed_params = self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        # Output head params
        head_params = sum(p.numel() for m in [self.lm_head, self.tool_head] for p in m.parameters())
        # Router params
        router_params = sum(p.numel() for p in self.moe_layer.router.parameters())
        # Verifier params
        verifier_params = sum(p.numel() for p in self.verifier.parameters())
        # Memory params
        memory_params = sum(p.numel() for m in [self.episodic_memory.compressor, self.episodic_memory.decompressor, self.episodic_memory.retrieval_proj] for p in m.parameters())
        memory_params += sum(p.numel() for p in self.working_memory.parameters())
        # Expert params (only top-k active)
        expert_params_per_token = sum(
            sum(p.numel() for p in expert.parameters()) 
            for expert in self.expert_pool.experts.values()
        ) * (self.config.router.top_k / len(self.expert_pool.experts))
        
        return int(core_params + embed_params + head_params + router_params + verifier_params + memory_params + expert_params_per_token)


def create_dreamrabbit(config: DreamRabbitConfig = None) -> DreamRabbit:
    """Factory function to create DreamRabbit model."""
    return DreamRabbit(config)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by component."""
    counts = {}
    for name, module in model.named_children():
        counts[name] = sum(p.numel() for p in module.parameters())
    counts["total"] = sum(counts.values())
    return counts