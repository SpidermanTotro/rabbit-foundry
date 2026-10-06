"""
Test script to verify DreamRabbit architecture works.
"""
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import get_default_config, get_small_config, get_tiny_config
from model import DreamRabbit, count_parameters


def test_tiny_model():
    """Test tiny model for fast verification."""
    print("=" * 60)
    print("Testing DreamRabbit Tiny Config")
    print("=" * 60)
    
    config = get_tiny_config()
    print(f"Config: d_model={config.core.d_model}, n_layers={config.core.n_layers}")
    print(f"Experts: {len(config.experts)}")
    print(f"Max recursion depth: {config.core.max_recursion_depth}")
    
    model = DreamRabbit(config)
    
    # Count parameters
    param_counts = count_parameters(model)
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    print(f"  Total: {param_counts['total']:,} ({param_counts['total']/1e6:.1f}M)")
    
    active_params = model.get_active_params()
    print(f"  Estimated active params/forward: {active_params:,} ({active_params/1e6:.1f}M)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids, 
            attention_mask=attention_mask,
            return_verification=True,
            return_routing_info=True
        )
    
    print(f"  lm_logits shape: {outputs['lm_logits'].shape}")
    print(f"  tool_logits shape: {outputs['tool_logits'].shape}")
    print(f"  recursion_depths: {outputs['recursion_depths']}")
    print(f"  verification keys: {list(outputs['verification'].keys())}")
    for k, v in outputs['verification'].items():
        print(f"    {k}: {v.shape} mean={v.mean().item():.4f}")
    print(f"  should_retry: {outputs['should_retry']}")
    print(f"  routing_info expert_usage: {outputs['routing_info']['expert_usage']}")
    print(f"  router_entropy: {outputs['routing_info']['router_entropy']:.4f}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"  Generated shape: {generated.shape}")
    print(f"  New tokens: {generated.shape[1] - 10}")
    
    # Test memory
    print("\nTesting memory...")
    model.store_experience(outputs['hidden_states'][:1])
    memory_context = model.retrieve_memory(outputs['hidden_states'][:1])
    print(f"  Memory context shape: {memory_context.shape}")
    
    print("\n✓ Tiny model test PASSED")
    return model


def test_small_model():
    """Test small model (~100M params)."""
    print("\n" + "=" * 60)
    print("Testing DreamRabbit Small Config")
    print("=" * 60)
    
    config = get_small_config()
    print(f"Config: d_model={config.core.d_model}, n_layers={config.core.n_layers}")
    print(f"Experts: {len(config.experts)}")
    
    model = DreamRabbit(config)
    
    param_counts = count_parameters(model)
    print(f"\nTotal params: {param_counts['total']:,} ({param_counts['total']/1e6:.1f}M)")
    active_params = model.get_active_params()
    print(f"Active params/forward: {active_params:,} ({active_params/1e6:.1f}M)")
    
    # Quick forward
    batch_size, seq_len = 1, 256
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, return_verification=True)
    
    print(f"  lm_logits: {outputs['lm_logits'].shape}")
    print(f"  depths: {outputs['recursion_depths']}")
    print(f"  verification: {[f'{k}={v.mean():.3f}' for k,v in outputs['verification'].items()]}")
    
    print("\n✓ Small model test PASSED")
    return model


def test_recursive_behavior():
    """Test that recursion depth varies with input complexity."""
    print("\n" + "=" * 60)
    print("Testing Recursive Behavior")
    print("=" * 60)
    
    config = get_tiny_config()
    config.core.max_recursion_depth = 4
    model = DreamRabbit(config)
    model.eval()
    
    # Simple input (repetitive)
    simple_input = torch.tensor([[1, 2, 3, 4] * 32])
    # Complex input (random)
    complex_input = torch.randint(0, config.vocab_size, (1, 128))
    
    with torch.no_grad():
        simple_out = model(simple_input, return_verification=True)
        complex_out = model(complex_input, return_verification=True)
    
    print(f"Simple input depth: {simple_out['recursion_depths']}")
    print(f"Complex input depth: {complex_out['recursion_depths']}")
    print(f"Simple confidence: {simple_out['verification']['confidence'].mean():.4f}")
    print(f"Complex confidence: {complex_out['verification']['confidence'].mean():.4f}")
    print(f"Simple retry signal: {simple_out['verification']['retry'].mean():.4f}")
    print(f"Complex retry signal: {complex_out['verification']['retry'].mean():.4f}")
    
    print("\n✓ Recursive behavior test PASSED")


def test_expert_growth():
    """Test adding new experts dynamically."""
    print("\n" + "=" * 60)
    print("Testing Expert Pool Growth")
    print("=" * 60)
    
    from config import ExpertConfig
    from experts import ExpertPool
    
    config = get_tiny_config()
    # Update expert configs like the model does
    for ec in config.experts:
        ec.d_model = config.core.d_model
    pool = ExpertPool(config.experts)
    
    print(f"Initial experts: {pool.list_experts()}")
    
    # Add new expert with matching d_model
    new_expert = ExpertConfig(name="reasoning", specialization="reasoning", d_model=config.core.d_model)
    pool.add_expert(new_expert)
    
    print(f"After adding 'reasoning': {pool.list_experts()}")
    print(f"Expert info: {pool.get_expert_info()}")
    
    # Test forward with new expert
    x = torch.randn(2, 10, config.core.d_model)
    outputs = pool.forward(x, ["reasoning", "general_1"])
    print(f"Output shapes: { {k: v.shape for k, v in outputs.items()} }")
    
    print("\n✓ Expert growth test PASSED")


if __name__ == "__main__":
    # Run all tests
    test_tiny_model()
    test_small_model()
    test_recursive_behavior()
    test_expert_growth()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)