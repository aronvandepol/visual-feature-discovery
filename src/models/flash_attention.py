# src/models/flash_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash-attn not available. Install with: pip install flash-attn")


class FlashAttention(nn.Module):
    """
    Flash Attention implementation for efficient attention computation.
    Falls back to optimized standard attention if flash-attn is not available.
    """
    
    def __init__(self, L: int, D: int, dropout: float = 0.0):
        super(FlashAttention, self).__init__()
        self.L = L  # Input feature dimension
        self.D = D  # Attention dimension
        self.dropout = dropout
        
        # Query and Key projections
        self.query = nn.Linear(L, D)
        self.key = nn.Linear(L, D)
        
        # Optional: Add value projection for more flexibility
        self.value = nn.Linear(L, D)
        
        # Output projection to get back to original dimension
        self.output_proj = nn.Linear(D, 1)
        
        # Scale factor for attention
        self.scale = D ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Flash Attention.
        
        Args:
            x: Input tensor of shape (B, N, L) where
               B = batch size, N = number of patches, L = feature dimension
               
        Returns:
            attention: Attention weights of shape (B, 1, N)
        """
        batch_size, num_patches, _ = x.shape
        
        # Project to Q, K, V
        Q = self.query(x)  # (B, N, D)
        K = self.key(x)    # (B, N, D)
        V = self.value(x)  # (B, N, D)
        
        if FLASH_ATTN_AVAILABLE and x.is_cuda:
            # Use Flash Attention for efficient computation
            attn_output = self._flash_attention(Q, K, V)
        else:
            # Fallback to optimized standard attention
            attn_output = self._standard_attention(Q, K, V)
        
        # Project to get attention weights
        attention_weights = self.output_proj(attn_output)  # (B, N, 1)
        attention_weights = attention_weights.transpose(1, 2)  # (B, 1, N)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        return attention_weights
    
    def _flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Use Flash Attention for memory-efficient computation.
        """
        batch_size, num_patches, head_dim = Q.shape
        
        # Flash attention expects (batch, seqlen, nheads, headdim)
        # We have single head attention, so add head dimension
        Q = Q.unsqueeze(2)  # (B, N, 1, D)
        K = K.unsqueeze(2)  # (B, N, 1, D)
        V = V.unsqueeze(2)  # (B, N, 1, D)
        
        # Apply flash attention
        attn_output = flash_attn_func(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False  # Not causal attention
        )
        
        # Remove head dimension: (B, N, 1, D) -> (B, N, D)
        attn_output = attn_output.squeeze(2)
        
        return attn_output
    
    def _standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient attention using PyTorch's scaled_dot_product_attention.
        This automatically uses the most efficient kernel available (flash, mem-efficient, or math).
        Never materializes the full N×N attention matrix.
        """
        batch_size, num_patches, head_dim = Q.shape

        # scaled_dot_product_attention expects (B, num_heads, N, D)
        # We use 1 head, so add the head dimension
        Q = Q.unsqueeze(1)  # (B, 1, N, D)
        K = K.unsqueeze(1)  # (B, 1, N, D)
        V = V.unsqueeze(1)  # (B, 1, N, D)

        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=dropout_p,
            scale=self.scale,
        )  # (B, 1, N, D)

        # Remove head dimension
        attn_output = attn_output.squeeze(1)  # (B, N, D)
        return attn_output


class OptimizedAttention(nn.Module):
    """
    Drop-in replacement for the original Attention class with Flash Attention optimization.
    Maintains the same interface as the original for easy replacement.
    """
    
    def __init__(self, L: int, D: int):
        super(OptimizedAttention, self).__init__()
        self.L = L
        self.D = D
        
        # Use Flash Attention internally
        self.flash_attn = FlashAttention(L, D)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that matches the original Attention class interface.
        
        Args:
            x: Input tensor of shape (B, N, L)
            
        Returns:
            attention: Attention weights of shape (B, 1, N)
        """
        return self.flash_attn(x)


# Utility function to replace attention in existing models
def upgrade_attention_to_flash(model, attention_attr_name: str = 'patch_attention'):
    """
    Utility function to upgrade existing attention modules to Flash Attention.
    
    Args:
        model: The model containing the attention module
        attention_attr_name: Name of the attention attribute to replace
    """
    if hasattr(model, attention_attr_name):
        old_attention = getattr(model, attention_attr_name)
        
        # Create new Flash Attention with same parameters
        new_attention = OptimizedAttention(old_attention.L, old_attention.D)
        
        # Replace the attention module
        setattr(model, attention_attr_name, new_attention)
        
        print(f"✓ Upgraded {attention_attr_name} to Flash Attention")
        if FLASH_ATTN_AVAILABLE:
            print("✓ Using Flash Attention kernels for optimal performance")
        else:
            print("! Using optimized standard attention (install flash-attn for best performance)")
    else:
        print(f"Warning: {attention_attr_name} not found in model")