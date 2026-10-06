"""
Expert Pool - Specialized neural modules for different capabilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from config import ExpertConfig


class Expert(nn.Module):
    """A single expert module with specialized capability."""
    
    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        self.name = config.name
        
        # Expert-specific layers
        layers = []
        for i in range(config.n_layers):
            layers.extend([
                nn.Linear(config.d_model, config.d_expert),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_expert, config.d_model),
                nn.Dropout(config.dropout),
            ])
        self.net = nn.Sequential(*layers)
        
        # Specialization marker (learned)
        self.specialization_bias = nn.Parameter(torch.zeros(config.d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through expert."""
        out = self.net(x)
        return out + self.specialization_bias


class ExpertPool(nn.Module):
    """
    Pool of specialized experts that can grow over time.
    
    Experts are specialists rather than copies:
    - Python: code generation/understanding
    - Debugging: error analysis, fix generation
    - Planning: task decomposition, strategy
    - Shell: command generation, execution reasoning
    - Git: version control operations
    - Math: mathematical reasoning
    - Language: natural language understanding
    - Tool_use: API calls, tool orchestration
    - Memory: retrieval, storage decisions
    - General: fallback/unspecialized
    """
    
    def __init__(self, expert_configs: List[ExpertConfig]):
        super().__init__()
        self.experts = nn.ModuleDict()
        self.expert_configs = {c.name: c for c in expert_configs}
        
        for config in expert_configs:
            self.experts[config.name] = Expert(config)
            
    def forward(self, x: torch.Tensor, expert_names: List[str]) -> Dict[str, torch.Tensor]:
        """Run selected experts on input."""
        outputs = {}
        for name in expert_names:
            if name in self.experts:
                outputs[name] = self.experts[name](x)
        return outputs
    
    def get_expert(self, name: str) -> Optional[Expert]:
        """Get a specific expert by name."""
        return self.experts.get(name)
    
    def add_expert(self, config: ExpertConfig):
        """Add a new expert to the pool (for growth)."""
        if config.name in self.experts:
            raise ValueError(f"Expert {config.name} already exists")
        self.experts[config.name] = Expert(config)
        self.expert_configs[config.name] = config
    
    def remove_expert(self, name: str):
        """Remove an expert from the pool."""
        if name in self.experts:
            del self.experts[name]
            del self.expert_configs[name]
    
    def list_experts(self) -> List[str]:
        """List all available expert names."""
        return list(self.experts.keys())
    
    def get_expert_info(self) -> Dict:
        """Get information about all experts."""
        return {
            name: {
                "specialization": config.specialization,
                "d_model": config.d_model,
                "d_expert": config.d_expert,
                "n_layers": config.n_layers,
            }
            for name, config in self.expert_configs.items()
        }


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with top-k routing.
    
    Routes tokens to top-k experts and combines their outputs.
    """
    
    def __init__(self, expert_pool: ExpertPool, router_config):
        super().__init__()
        self.expert_pool = expert_pool
        self.router_config = router_config
        self.n_experts = len(expert_pool.experts)
        self.top_k = router_config.top_k
        self.capacity_factor = router_config.capacity_factor
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(router_config.d_model, router_config.d_model),
            nn.GELU(),
            nn.Dropout(router_config.router_dropout),
            nn.Linear(router_config.d_model, self.n_experts),
        )
        
        # Expert names in fixed order
        self.expert_names = list(expert_pool.experts.keys())
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Route input to experts and combine outputs.
        
        Returns:
            output: Combined expert outputs
            router_logits: Raw router logits for load balancing
            routing_info: Dict with routing statistics
        """
        batch_size, seq_len, d_model = x.shape
        
        # Flatten for routing - use reshape to handle non-contiguous tensors
        x_flat = x.reshape(-1, d_model)  # (batch*seq, d_model)
        
        # Get router logits
        router_logits = self.router(x_flat)  # (batch*seq, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k routing
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process through selected experts
        routing_info = {
            "expert_usage": torch.zeros(self.n_experts, device=x.device),
            "router_entropy": -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean().item(),
        }
        
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # (batch*seq,)
            expert_weight = topk_probs[:, k].unsqueeze(-1)  # (batch*seq, 1)
            
            # Count usage
            for idx in expert_idx:
                routing_info["expert_usage"][idx] += 1
            
            # Group by expert for efficient processing
            for expert_id in range(self.n_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]  # (n_tokens, d_model)
                    expert_name = self.expert_names[expert_id]
                    expert_out = self.expert_pool.experts[expert_name](expert_input)
                    output[mask] += expert_weight[mask] * expert_out
        
        output = output.view(batch_size, seq_len, d_model)
        
        return output, router_logits, routing_info