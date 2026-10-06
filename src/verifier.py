"""
Neural Verifier - Self-correction as part of the architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from config import VerifierConfig


class VerifierHead(nn.Module):
    """Single verification head for a specific prediction."""
    
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralVerifier(nn.Module):
    """
    Neural verifier that predicts multiple verification signals before producing an action.
    
    Predicts:
    1. confidence - How confident is the model in its current reasoning?
    2. contradiction - Is there a contradiction in the reasoning?
    3. tool_needed - Does this require a tool call?
    4. retry - Should we recurse deeper in the core?
    5. hallucination - Is the model likely hallucinating?
    
    Low confidence or high contradiction triggers recursive core retry.
    """
    
    def __init__(self, config: VerifierConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Multiple verification heads
        self.heads = nn.ModuleDict({
            "confidence": VerifierHead(config.d_hidden, config.d_hidden, config.dropout),
            "contradiction": VerifierHead(config.d_hidden, config.d_hidden, config.dropout),
            "tool_needed": VerifierHead(config.d_hidden, config.d_hidden, config.dropout),
            "retry": VerifierHead(config.d_hidden, config.d_hidden, config.dropout),
            "hallucination": VerifierHead(config.d_hidden, config.d_hidden, config.dropout),
        })
        
        # Decision threshold parameters (learnable)
        self.confidence_threshold = nn.Parameter(torch.tensor(0.8))
        self.contradiction_threshold = nn.Parameter(torch.tensor(0.3))
        self.retry_threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run verification on hidden states.
        
        Args:
            x: Hidden states of shape (batch, seq, d_model) or (batch, d_model)
            
        Returns:
            Dict of verification predictions
        """
        # Pool sequence if needed
        if x.dim() == 3:
            x = x.mean(dim=1)  # (batch, d_model)
            
        # Encode
        encoded = self.encoder(x)  # (batch, d_hidden)
        
        # Run all heads
        results = {}
        for name, head in self.heads.items():
            results[name] = head(encoded)
            
        return results
    
    def should_retry(self, verification: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Determine if recursive core should retry based on verification signals.
        
        Returns boolean tensor of shape (batch,)
        """
        confidence = verification["confidence"]
        contradiction = verification["contradiction"]
        retry_signal = verification["retry"]
        
        # Retry if low confidence OR high contradiction OR explicit retry signal
        should_retry = (
            (confidence < self.confidence_threshold) |
            (contradiction > self.contradiction_threshold) |
            (retry_signal > self.retry_threshold)
        )
        
        return should_retry
    
    def get_verification_summary(self, verification: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get scalar summary for logging."""
        return {k: v.mean().item() for k, v in verification.items()}


class VerifierLoss(nn.Module):
    """Loss functions for training the verifier."""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self, 
        verification: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute verification loss.
        
        Args:
            verification: Predicted verification signals
            targets: Ground truth targets for each signal
            weights: Optional loss weights per signal
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses per signal
        """
        if weights is None:
            weights = {k: 1.0 for k in verification.keys()}
            
        losses = {}
        total = torch.tensor(0.0, device=next(iter(verification.values())).device)
        
        for name, pred in verification.items():
            if name in targets:
                target = targets[name]
                # Binary cross entropy
                loss = F.binary_cross_entropy(pred, target)
                losses[name] = loss
                total = total + weights.get(name, 1.0) * loss
                
        losses["total"] = total
        return total, losses