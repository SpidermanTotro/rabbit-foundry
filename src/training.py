"""
Training infrastructure - losses, metrics, and training loop utilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    compile_model: bool = False


class DreamRabbitLoss(nn.Module):
    """
    Combined loss for DreamRabbit training.
    
    Includes:
    - Language modeling loss (next token prediction)
    - Verification loss (self-correction signals)
    - Router load balancing loss
    - Memory reconstruction loss
    """
    
    def __init__(
        self,
        vocab_size: int,
        lm_weight: float = 1.0,
        verification_weight: float = 0.5,
        router_weight: float = 0.1,
        memory_weight: float = 0.2,
        verification_target_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.lm_weight = lm_weight
        self.verification_weight = verification_weight
        self.router_weight = router_weight
        self.memory_weight = memory_weight
        
        if verification_target_weights is None:
            verification_target_weights = {
                "confidence": 1.0,
                "contradiction": 1.0,
                "tool_needed": 1.0,
                "retry": 1.0,
                "hallucination": 1.0,
            }
        self.verification_target_weights = verification_target_weights
        
        self.verifier_loss_fn = nn.BCELoss(reduction="none")
        
    def forward(
        self,
        outputs: Dict,
        targets: Dict,
        verification_targets: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dict
            targets: Target dict with 'input_ids', 'attention_mask'
            verification_targets: Optional ground truth for verifier
            
        Returns:
            total_loss, loss_dict
        """
        lm_logits = outputs["lm_logits"]  # (batch, seq, vocab)
        input_ids = targets["input_ids"]  # (batch, seq)
        attention_mask = targets.get("attention_mask")
        
        # Language modeling loss (next token prediction)
        shift_logits = lm_logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
        else:
            shift_mask = torch.ones_like(shift_labels, dtype=torch.float)
            
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            reduction="none"
        )
        lm_loss = (lm_loss * shift_mask.view(-1)).sum() / shift_mask.sum().clamp(min=1)
        
        losses = {"lm_loss": lm_loss}
        total_loss = self.lm_weight * lm_loss
        
        # Router load balancing loss
        if "router_logits" in outputs:
            router_logits = outputs["router_logits"]  # (batch*seq, n_experts)
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Encourage uniform expert usage
            mean_probs = router_probs.mean(dim=0)
            router_loss = (mean_probs * torch.log(mean_probs * router_probs.shape[1] + 1e-10)).sum()
            losses["router_loss"] = router_loss
            total_loss = total_loss + self.router_weight * router_loss
            
        # Verification loss
        if verification_targets is not None and "verification" in outputs:
            verification = outputs["verification"]
            ver_loss = torch.tensor(0.0, device=lm_logits.device)
            
            for name, pred in verification.items():
                if name in verification_targets:
                    target = verification_targets[name]
                    # BCE loss per signal
                    signal_loss = F.binary_cross_entropy(pred, target, reduction="mean")
                    weight = self.verification_target_weights.get(name, 1.0)
                    ver_loss = ver_loss + weight * signal_loss
                    losses[f"verifier_{name}_loss"] = signal_loss
                    
            losses["verification_loss"] = ver_loss
            total_loss = total_loss + self.verification_weight * ver_loss
            
        # Memory reconstruction loss (if storing experiences)
        if "memory_reconstruction" in outputs:
            mem_loss = outputs["memory_reconstruction"]
            losses["memory_loss"] = mem_loss
            total_loss = total_loss + self.memory_weight * mem_loss
            
        losses["total_loss"] = total_loss
        return total_loss, losses


class GradientAccumulator:
    """Handles gradient accumulation for large effective batch sizes."""
    
    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.step = 0
        
    def __call__(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> bool:
        """
        Accumulate gradients. Returns True when optimizer should step.
        """
        loss = loss / self.accumulation_steps
        loss.backward()
        self.step += 1
        
        if self.step % self.accumulation_steps == 0:
            return True
        return False
    
    def reset(self):
        self.step = 0


def get_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer with weight decay for non-bias/norm params."""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name.lower() or "embedding" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(
        optimizer_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )


def get_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """Create learning rate scheduler with warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(step: int):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        # Cosine decay
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    loss_fn: DreamRabbitLoss,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_lm_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    for batch in dataloader:
        if max_batches and num_batches >= max_batches:
            break
            
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        targets = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        outputs = model(input_ids, attention_mask=attention_mask, return_routing_info=True)
        loss, loss_dict = loss_fn(outputs, targets)
        
        total_loss += loss_dict["total_loss"].item() * input_ids.shape[0]
        total_lm_loss += loss_dict["lm_loss"].item() * input_ids.shape[0]
        total_tokens += attention_mask.sum().item() if attention_mask is not None else input_ids.numel()
        num_batches += 1
        
    model.train()
    
    return {
        "eval_loss": total_loss / max(1, num_batches),
        "eval_lm_loss": total_lm_loss / max(1, num_batches),
        "eval_ppl": torch.exp(torch.tensor(total_lm_loss / max(1, total_tokens))).item(),
    }