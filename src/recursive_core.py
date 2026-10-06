"""
Recursive Core - The reasoning engine that can be reused dynamically.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from config import RecursiveCoreConfig


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional flash attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RecursiveCoreBlock(nn.Module):
    """A single block of the recursive core that can be reused."""
    
    def __init__(self, config: RecursiveCoreConfig):
        super().__init__()
        self.config = config
        
        self.attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = x + residual
        
        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual
        
        return x


class RecursiveCore(nn.Module):
    """
    The recursive reasoning core.
    
    Instead of a fixed stack of layers, we have a small set of core blocks
    that can be reused dynamically. Easy problems use fewer recursions,
    hard problems use more.
    """
    
    def __init__(self, config: RecursiveCoreConfig):
        super().__init__()
        self.config = config
        
        # Shared core blocks - these are reused
        self.core_blocks = nn.ModuleList([
            RecursiveCoreBlock(config) for _ in range(config.n_layers)
        ])
        
        # Recursion controller - decides whether to continue
        self.recursion_controller = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Depth embedding for each recursion step
        self.depth_embeddings = nn.Embedding(config.max_recursion_depth + 1, config.d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_all_depths: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Forward pass with dynamic recursion.
        
        Returns:
            final_output: The output after dynamic recursion
            all_depths: List of outputs at each recursion depth (if return_all_depths)
            recursion_depths: Tensor of shape (batch_size,) with actual depths used
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        all_outputs = []
        current_x = x
        
        # Track recursion depth per sequence
        depths = torch.ones(batch_size, dtype=torch.long, device=device)
        continue_recursion = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        for depth in range(self.config.max_recursion_depth):
            # Add depth embedding
            depth_emb = self.depth_embeddings(torch.full((batch_size,), depth, dtype=torch.long, device=device))
            depth_emb = depth_emb.unsqueeze(1).expand(-1, seq_len, -1)
            current_x = current_x + depth_emb
            
            # Apply all core blocks
            for block in self.core_blocks:
                current_x = block(current_x, mask)
            
            all_outputs.append(current_x)
            
            # Check if we should continue (except at max depth)
            if depth < self.config.max_recursion_depth - 1:
                # Compute continuation probability from pooled representation
                pooled = current_x.mean(dim=1)  # (batch, d_model)
                continue_prob = self.recursion_controller(pooled).squeeze(-1)  # (batch,)
                
                # Stochastic decision during training, deterministic during eval
                if self.training:
                    continue_recursion = continue_recursion & (torch.rand_like(continue_prob) < continue_prob)
                else:
                    continue_recursion = continue_recursion & (continue_prob > 0.5)
                    
                depths = depths + continue_recursion.long()
                
                if not continue_recursion.any():
                    break
        
        final_output = current_x
        if return_all_depths:
            return final_output, all_outputs, depths
        return final_output, depths
    
    def forward_fixed_depth(self, x: torch.Tensor, depth: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with fixed recursion depth (for evaluation/debugging)."""
        depth = min(depth, self.config.max_recursion_depth)
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        current_x = x
        for d in range(depth):
            depth_emb = self.depth_embeddings(torch.full((batch_size,), d, dtype=torch.long, device=device))
            depth_emb = depth_emb.unsqueeze(1).expand(-1, seq_len, -1)
            current_x = current_x + depth_emb
            
            for block in self.core_blocks:
                current_x = block(current_x, mask)
                
        return current_x