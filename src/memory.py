"""
Episodic Memory - Working context with retrieval and compression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from config import MemoryConfig


class CompressedMemory(nn.Module):
    """
    Compressed episodic memory store.
    
    Stores important experiences in compressed form.
    Uses learned compression to fit more memories.
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.memory_size
        self.d_model = config.d_model
        self.compression_ratio = config.compression_ratio
        self.compressed_dim = int(config.d_model * config.compression_ratio)
        
        # Compression encoder/decoder
        self.compressor = nn.Sequential(
            nn.Linear(config.d_model, self.compressed_dim),
            nn.GELU(),
            nn.Linear(self.compressed_dim, self.compressed_dim),
        )
        self.decompressor = nn.Sequential(
            nn.Linear(self.compressed_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        
        # Memory storage (registered as buffer for persistence)
        self.register_buffer("memory_keys", torch.zeros(config.memory_size, self.compressed_dim))
        self.register_buffer("memory_values", torch.zeros(config.memory_size, config.d_model))
        self.register_buffer("memory_timestamps", torch.zeros(config.memory_size, dtype=torch.long))
        self.register_buffer("memory_importance", torch.zeros(config.memory_size))
        self.register_buffer("write_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("num_stored", torch.zeros(1, dtype=torch.long))
        
        # Retrieval network
        self.retrieval_proj = nn.Linear(config.d_model, self.compressed_dim)
        
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress hidden state for storage."""
        return self.compressor(x)
    
    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        """Decompress stored memory."""
        return self.decompressor(x)
    
    def write(self, keys: torch.Tensor, values: torch.Tensor, importance: Optional[torch.Tensor] = None):
        """
        Write memories to the store.
        
        Args:
            keys: Compressed keys (batch, compressed_dim)
            values: Original values (batch, d_model)
            importance: Importance scores (batch,) - higher = more important
        """
        batch_size = keys.shape[0]
        device = keys.device
        
        if importance is None:
            importance = torch.ones(batch_size, device=device)
            
        ptr = self.write_ptr.item()
        num_stored = self.num_stored.item()
        
        for i in range(batch_size):
            idx = (ptr + i) % self.memory_size
            
            # Only overwrite if new memory is more important
            if num_stored < self.memory_size or importance[i] > self.memory_importance[idx]:
                self.memory_keys[idx] = keys[i]
                self.memory_values[idx] = values[i]
                self.memory_importance[idx] = importance[i]
                self.memory_timestamps[idx] = num_stored + i
                
        self.write_ptr[0] = (ptr + batch_size) % self.memory_size
        self.num_stored[0] = min(self.num_stored[0] + batch_size, self.memory_size)
    
    def retrieve(self, query: torch.Tensor, top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve memories relevant to query.
        
        Returns:
            retrieved_values: (batch, top_k, d_model)
            retrieved_keys: (batch, top_k, compressed_dim)
            scores: (batch, top_k)
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k
            
        batch_size = query.shape[0]
        num_stored = self.num_stored.item()
        
        if num_stored == 0:
            # Return zeros if no memories
            return (
                torch.zeros(batch_size, top_k, self.d_model, device=query.device),
                torch.zeros(batch_size, top_k, self.compressed_dim, device=query.device),
                torch.zeros(batch_size, top_k, device=query.device)
            )
            
        # Project query to compressed space
        query_compressed = self.retrieval_proj(query)  # (batch, compressed_dim)
        
        # Compute similarities with stored keys (only up to num_stored)
        stored_keys = self.memory_keys[:num_stored]  # (num_stored, compressed_dim)
        stored_values = self.memory_values[:num_stored]  # (num_stored, d_model)
        stored_importance = self.memory_importance[:num_stored]  # (num_stored,)
        
        # Cosine similarity
        query_norm = F.normalize(query_compressed, dim=-1)
        keys_norm = F.normalize(stored_keys, dim=-1)
        similarities = torch.matmul(query_norm, keys_norm.t())  # (batch, num_stored)
        
        # Weight by importance
        similarities = similarities * stored_importance.unsqueeze(0)
        
        # Top-k retrieval
        topk_scores, topk_indices = torch.topk(similarities, min(top_k, num_stored), dim=-1)
        
        # Gather retrieved memories
        retrieved_keys = stored_keys[topk_indices]  # (batch, top_k, compressed_dim)
        retrieved_values = stored_values[topk_indices]  # (batch, top_k, d_model)
        
        return retrieved_values, retrieved_keys, topk_scores
    
    def forward(self, query: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        """Retrieve and return aggregated memory context."""
        retrieved_values, _, scores = self.retrieve(query, top_k)
        
        # Weighted aggregation
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, top_k, 1)
        aggregated = (retrieved_values * weights).sum(dim=1)  # (batch, d_model)
        
        return aggregated


class WorkingMemory(nn.Module):
    """
    Working memory that maintains context across recursive steps.
    """
    
    def __init__(self, d_model: int, max_context_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.max_context_len = max_context_len
        
        # Context buffer
        self.register_buffer("context", torch.zeros(1, max_context_len, d_model))
        self.register_buffer("context_len", torch.zeros(1, dtype=torch.long))
        
        # Context compression for long sequences
        self.compress = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        
    def update(self, new_context: torch.Tensor):
        """Update working memory with new context."""
        batch_size, seq_len, _ = new_context.shape
        
        # Simple strategy: keep most recent
        if seq_len >= self.max_context_len:
            self.context[0] = new_context[0, -self.max_context_len:]
            self.context_len[0] = self.max_context_len
        else:
            current_len = self.context_len.item()
            remaining = self.max_context_len - current_len
            
            if seq_len <= remaining:
                # Append
                self.context[0, current_len:current_len + seq_len] = new_context[0]
                self.context_len[0] = current_len + seq_len
            else:
                # Shift and append
                keep = remaining
                self.context[0, :keep] = self.context[0, current_len - keep:current_len]
                self.context[0, keep:keep + seq_len] = new_context[0]
                self.context_len[0] = min(keep + seq_len, self.max_context_len)
    
    def get_context(self) -> torch.Tensor:
        """Get current working memory context."""
        length = self.context_len.item()
        if length == 0:
            return torch.zeros(1, 1, self.d_model, device=self.context.device)
        return self.context[0:1, :length]
    
    def clear(self):
        """Clear working memory."""
        self.context.zero_()
        self.context_len.zero_()