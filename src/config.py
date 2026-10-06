"""
DreamRabbit Core Configuration
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RecursiveCoreConfig:
    """Configuration for the recursive reasoning core."""
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_recursion_depth: int = 4
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Recursive control
    min_recursion_depth: int = 1
    recursion_temperature: float = 1.0
    confidence_threshold: float = 0.8


@dataclass
class ExpertConfig:
    """Configuration for a single expert module."""
    name: str
    d_model: int = 512
    d_expert: int = 1024
    n_layers: int = 2
    dropout: float = 0.1
    specialization: str = "general"


@dataclass
class RouterConfig:
    """Configuration for the expert router."""
    d_model: int = 512
    n_experts: int = 16
    top_k: int = 2
    capacity_factor: float = 1.25
    router_dropout: float = 0.1
    load_balance_weight: float = 0.01


@dataclass
class VerifierConfig:
    """Configuration for the neural verifier."""
    d_model: int = 512
    d_hidden: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    n_verifier_heads: int = 5  # confidence, contradiction, tool_needed, retry, hallucination


@dataclass
class MemoryConfig:
    """Configuration for episodic memory."""
    d_model: int = 512
    memory_size: int = 10000
    retrieval_top_k: int = 8
    compression_ratio: float = 0.5


def _default_experts() -> List[ExpertConfig]:
    return [
        ExpertConfig(name="python", specialization="python"),
        ExpertConfig(name="debugging", specialization="debugging"),
        ExpertConfig(name="planning", specialization="planning"),
        ExpertConfig(name="shell", specialization="shell"),
        ExpertConfig(name="git", specialization="git"),
        ExpertConfig(name="math", specialization="math"),
        ExpertConfig(name="language", specialization="language"),
        ExpertConfig(name="tool_use", specialization="tool_use"),
        ExpertConfig(name="memory", specialization="memory"),
        ExpertConfig(name="general_1", specialization="general"),
        ExpertConfig(name="general_2", specialization="general"),
        ExpertConfig(name="general_3", specialization="general"),
        ExpertConfig(name="general_4", specialization="general"),
        ExpertConfig(name="general_5", specialization="general"),
        ExpertConfig(name="general_6", specialization="general"),
        ExpertConfig(name="general_7", specialization="general"),
    ]


@dataclass
class DreamRabbitConfig:
    """Main configuration for DreamRabbit model."""
    core: RecursiveCoreConfig = field(default_factory=RecursiveCoreConfig)
    experts: List[ExpertConfig] = field(default_factory=_default_experts)
    router: RouterConfig = field(default_factory=RouterConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    vocab_size: int = 32000
    max_seq_len: int = 4096
    tie_weights: bool = True


def get_default_config() -> DreamRabbitConfig:
    """Get default DreamRabbit configuration (~500M params)."""
    return DreamRabbitConfig()


def get_small_config() -> DreamRabbitConfig:
    """Get small DreamRabbit configuration (~100M params)."""
    config = DreamRabbitConfig()
    config.core.d_model = 256
    config.core.n_layers = 6
    config.core.n_heads = 4
    config.core.d_ff = 1024
    config.router.n_experts = 8
    config.verifier.d_hidden = 128
    config.experts = config.experts[:8]
    return config


def get_tiny_config() -> DreamRabbitConfig:
    """Get tiny DreamRabbit configuration (~30M params) for fast testing."""
    config = DreamRabbitConfig()
    config.core.d_model = 128
    config.core.n_layers = 4
    config.core.n_heads = 2
    config.core.d_ff = 512
    config.router.n_experts = 4
    config.verifier.d_hidden = 64
    config.experts = [
        ExpertConfig(name="general_1", specialization="general"),
        ExpertConfig(name="general_2", specialization="general"),
        ExpertConfig(name="general_3", specialization="general"),
        ExpertConfig(name="general_4", specialization="general"),
    ]
    return config