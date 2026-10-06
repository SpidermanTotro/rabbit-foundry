"""
GGUF Export for DreamRabbit - Convert to llama.cpp format.
"""
import struct
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


class GGUFWriter:
    """Write model weights to GGUF format."""
    
    # GGUF constants
    MAGIC = b"GGUF"
    VERSION = 3
    
    # Value types
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "wb")
        self.tensor_info = []
        self.tensor_data = {}
        self.metadata = {}
        
    def write_header(self):
        """Write GGUF header."""
        self.f.write(self.MAGIC)
        self.f.write(struct.pack("<I", self.VERSION))
        # Tensor count and metadata count will be written later
        self.tensor_count_pos = self.f.tell()
        self.f.write(struct.pack("<Q", 0))  # placeholder
        self.metadata_count_pos = self.f.tell()
        self.f.write(struct.pack("<Q", 0))  # placeholder
        
    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair."""
        self.metadata[key] = value
        
    def add_tensor(self, name: str, tensor: torch.Tensor, dtype: str = "F32"):
        """Add tensor to be written."""
        # Convert to numpy
        if tensor.dtype == torch.bfloat16:
            arr = tensor.to(torch.float32).numpy()
        elif tensor.dtype == torch.float16:
            arr = tensor.to(torch.float32).numpy()
        else:
            arr = tensor.numpy()
            
        self.tensor_info.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": dtype,
            "offset": 0,  # Will be filled later
        })
        self.tensor_data[name] = arr
        
    def _write_string(self, s: str):
        """Write length-prefixed string."""
        encoded = s.encode('utf-8')
        self.f.write(struct.pack("<Q", len(encoded)))
        self.f.write(encoded)
        
    def _write_value(self, value: Any):
        """Write a metadata value."""
        if isinstance(value, bool):
            self.f.write(struct.pack("<B", self.BOOL))
            self.f.write(struct.pack("<?", value))
        elif isinstance(value, int):
            if value < 0:
                self.f.write(struct.pack("<B", self.INT64))
                self.f.write(struct.pack("<q", value))
            else:
                self.f.write(struct.pack("<B", self.UINT64))
                self.f.write(struct.pack("<Q", value))
        elif isinstance(value, float):
            self.f.write(struct.pack("<B", self.FLOAT64))
            self.f.write(struct.pack("<d", value))
        elif isinstance(value, str):
            self.f.write(struct.pack("<B", self.STRING))
            self._write_string(value)
        elif isinstance(value, list):
            self.f.write(struct.pack("<B", self.ARRAY))
            self.f.write(struct.pack("<I", len(value)))
            if value:
                # Determine type from first element
                first = value[0]
                if isinstance(first, bool):
                    self.f.write(struct.pack("<B", self.BOOL))
                    for v in value:
                        self.f.write(struct.pack("<?", v))
                elif isinstance(first, int):
                    self.f.write(struct.pack("<B", self.INT64))
                    for v in value:
                        self.f.write(struct.pack("<q", v))
                elif isinstance(first, float):
                    self.f.write(struct.pack("<B", self.FLOAT64))
                    for v in value:
                        self.f.write(struct.pack("<d", v))
                elif isinstance(first, str):
                    self.f.write(struct.pack("<B", self.STRING))
                    for v in value:
                        self._write_string(v)
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")
            
    def finalize(self):
        """Write all metadata and tensors, finalize file."""
        # Write metadata
        metadata_start = self.f.tell()
        self.f.seek(self.metadata_count_pos)
        self.f.write(struct.pack("<Q", len(self.metadata)))
        self.f.seek(metadata_start)
        
        for key, value in self.metadata.items():
            self._write_string(key)
            self._write_value(value)
            
        # Write tensor info
        tensor_info_start = self.f.tell()
        self.f.seek(self.tensor_count_pos)
        self.f.write(struct.pack("<Q", len(self.tensor_info)))
        self.f.seek(tensor_info_start)
        
        # Align to 32 bytes for tensor data
        tensor_data_start = (tensor_info_start + len(self.tensor_info) * 64 + 31) // 32 * 32
        # We'll need to write tensor info first, then pad, then tensor data
        
        # Write tensor info with offsets
        current_offset = tensor_data_start
        for info in self.tensor_info:
            self._write_string(info["name"])
            self.f.write(struct.pack("<I", len(info["shape"])))
            for dim in info["shape"]:
                self.f.write(struct.pack("<Q", dim))
            self.f.write(struct.pack("<I", self._dtype_to_id(info["dtype"])))
            self.f.write(struct.pack("<Q", current_offset))
            # Calculate size
            dtype_size = self._dtype_size(info["dtype"])
            tensor_size = dtype_size
            for dim in info["shape"]:
                tensor_size *= dim
            # Align to 32 bytes
            current_offset += (tensor_size + 31) // 32 * 32
            
        # Pad to tensor data start
        self.f.write(b'\x00' * (tensor_data_start - self.f.tell()))
        
        # Write tensor data
        for info in self.tensor_info:
            name = info["name"]
            arr = self.tensor_data[name]
            dtype = info["dtype"]
            
            if dtype == "F32":
                data = arr.astype(np.float32).tobytes()
            elif dtype == "F16":
                data = arr.astype(np.float16).tobytes()
            elif dtype == "Q8_0":
                data = self._quantize_q8_0(arr)
            elif dtype == "Q4_0":
                data = self._quantize_q4_0(arr)
            else:
                data = arr.astype(np.float32).tobytes()
                
            self.f.write(data)
            # Pad to 32 bytes
            padding = (32 - len(data) % 32) % 32
            self.f.write(b'\x00' * padding)
            
        self.f.close()
        
    def _dtype_to_id(self, dtype: str) -> int:
        """Map dtype string to GGUF type ID."""
        mapping = {
            "F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7,
            "Q8_0": 8, "Q8_1": 9, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13,
            "Q6_K": 14, "Q8_K": 15, "I8": 16, "I16": 17, "I32": 18, "I64": 19,
            "F64": 20, "BOOL": 21,
        }
        return mapping.get(dtype, 0)
        
    def _dtype_size(self, dtype: str) -> int:
        """Get size in bytes for dtype."""
        sizes = {
            "F32": 4, "F16": 2, "Q4_0": 0.5, "Q4_1": 0.5, "Q5_0": 0.625, "Q5_1": 0.625,
            "Q8_0": 1, "Q8_1": 1, "Q2_K": 0.25, "Q3_K": 0.375, "Q4_K": 0.5, "Q5_K": 0.625,
            "Q6_K": 0.75, "Q8_K": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "F64": 8, "BOOL": 1,
        }
        return int(sizes.get(dtype, 4))
        
    def _quantize_q8_0(self, arr: np.ndarray) -> bytes:
        """Quantize to Q8_0 (8-bit per weight with block scaling)."""
        block_size = 32
        arr_flat = arr.flatten()
        n_blocks = (len(arr_flat) + block_size - 1) // block_size
        result = bytearray()
        
        for i in range(n_blocks):
            block = arr_flat[i*block_size:(i+1)*block_size]
            if len(block) == 0:
                continue
            abs_max = np.max(np.abs(block))
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            result.extend(struct.pack("<f", scale))
            quantized = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
            result.extend(quantized.tobytes())
            # Pad block to 32
            if len(quantized) < block_size:
                result.extend(b'\x00' * (block_size - len(quantized)))
        return bytes(result)
        
    def _quantize_q4_0(self, arr: np.ndarray) -> bytes:
        """Quantize to Q4_0 (4-bit per weight with block scaling)."""
        block_size = 32
        arr_flat = arr.flatten()
        n_blocks = (len(arr_flat) + block_size - 1) // block_size
        result = bytearray()
        
        for i in range(n_blocks):
            block = arr_flat[i*block_size:(i+1)*block_size]
            if len(block) == 0:
                continue
            abs_max = np.max(np.abs(block))
            scale = abs_max / 7.0 if abs_max > 0 else 1.0
            result.extend(struct.pack("<f", scale))
            quantized = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
            # Pack 2 values per byte
            packed = np.zeros(block_size // 2, dtype=np.uint8)
            for j in range(0, len(quantized), 2):
                low = quantized[j] & 0xF
                high = (quantized[j+1] & 0xF) << 4 if j+1 < len(quantized) else 0
                packed[j//2] = low | high
            result.extend(packed.tobytes())
        return bytes(result)


def export_to_gguf(model, tokenizer, output_path: str, quantization: str = "F16"):
    """
    Export DreamRabbit model to GGUF format.
    
    Args:
        model: DreamRabbit model
        tokenizer: Tokenizer
        output_path: Output GGUF file path
        quantization: Quantization type (F32, F16, Q8_0, Q4_0, Q4_K, Q8_K)
    """
    writer = GGUFWriter(output_path)
    writer.write_header()
    
    # Metadata
    writer.add_metadata("general.architecture", "dreamrabbit")
    writer.add_metadata("general.name", "DreamRabbit")
    writer.add_metadata("general.version", "0.1")
    writer.add_metadata("general.author", "Rabbit Foundry")
    writer.add_metadata("general.description", "Adaptive Recursive Model with Expert Pool and Neural Verifier")
    writer.add_metadata("dreamrabbit.vocab_size", model.config.vocab_size)
    writer.add_metadata("dreamrabbit.max_seq_len", model.config.max_seq_len)
    writer.add_metadata("dreamrabbit.d_model", model.config.core.d_model)
    writer.add_metadata("dreamrabbit.n_layers", model.config.core.n_layers)
    writer.add_metadata("dreamrabbit.n_heads", model.config.core.n_heads)
    writer.add_metadata("dreamrabbit.n_experts", len(model.config.experts))
    writer.add_metadata("dreamrabbit.top_k", model.config.router.top_k)
    writer.add_metadata("dreamrabbit.max_recursion_depth", model.config.core.max_recursion_depth)
    writer.add_metadata("tokenizer.ggml.model", "llama")
    writer.add_metadata("tokenizer.ggml.tokens", tokenizer.get_vocab())
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Map parameter names to GGUF tensor names
    tensor_mapping = {
        "token_embedding.weight": "token_embd.weight",
        "position_embedding.weight": "pos_embd.weight",
        "recursive_core.core_blocks.": "blk.",
        "recursive_core.recursion_controller.": "recursion_controller.",
        "recursive_core.depth_embeddings.weight": "depth_embd.weight",
        "expert_pool.experts.": "expert.",
        "moe_layer.router.": "router.",
        "verifier.encoder.": "verifier_encoder.",
        "verifier.heads.": "verifier_head_",
        "episodic_memory.compressor.": "mem_compressor.",
        "episodic_memory.decompressor.": "mem_decompressor.",
        "episodic_memory.retrieval_proj.weight": "mem_retrieval_proj.weight",
        "working_memory.compress.": "wm_compress.",
        "lm_head.weight": "output.weight",
        "tool_head.": "tool_head.",
    }
    
    for name, param in state_dict.items():
        if not param.requires_grad:
            continue
            
        # Map name
        gguf_name = name
        for old, new in tensor_mapping.items():
            if old in gguf_name:
                gguf_name = gguf_name.replace(old, new)
                break
                
        # Determine quantization
        if "weight" in name and param.dim() >= 2 and quantization != "F32":
            tensor_dtype = quantization
        else:
            tensor_dtype = "F32"
            
        writer.add_tensor(gguf_name, param.detach().cpu(), tensor_dtype)
        
    writer.finalize()
    print(f"Exported to {output_path}")
    

def verify_gguf(gguf_path: str, original_model, tokenizer, test_prompts: List[str] = None):
    """
    Verify GGUF export by comparing with original model.
    """
    if test_prompts is None:
        test_prompts = [
            "def hello():",
            "The capital of France is",
            "import torch\n",
        ]
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama_cpp not installed, skipping verification")
        return
        
    # Load GGUF
    llm = Llama(model_path=gguf_path, n_ctx=2048, verbose=False)
    
    original_model.eval()
    device = next(original_model.parameters()).device
    
    print("Verifying GGUF export...")
    for prompt in test_prompts:
        # Original model
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            orig_out = original_model.generate(inputs["input_ids"], max_new_tokens=20, temperature=0.0)
        orig_text = tokenizer.decode(orig_out[0], skip_special_tokens=True)
        
        # GGUF model
        gguf_text = llm(prompt, max_tokens=20, temperature=0.0, echo=False)["choices"][0]["text"]
        
        print(f"Prompt: {prompt}")
        print(f"  Original: {orig_text[len(prompt):]}")
        print(f"  GGUF:     {gguf_text}")
        print()
        
    print("Verification complete")


if __name__ == "__main__":
    import sys
    from src.model import DreamRabbit, get_tiny_config
    from transformers import AutoTokenizer
    
    # Quick test export
    config = get_tiny_config()
    model = DreamRabbit(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    export_to_gguf(model, tokenizer, "/tmp/test_dreamrabbit.gguf", "F16")