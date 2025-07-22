import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class CacheManager(nn.Module):
    """Base class for cache management strategies"""
    
    def __init__(self, max_cache_length: int, nhid: int):
        super().__init__()
        self.max_cache_length = max_cache_length
        self.nhid = nhid
    
    def manage_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor, 
                    hidden: torch.Tensor, attention_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Override in subclasses"""
        raise NotImplementedError

class SlidingWindowCache(CacheManager):
    """Strategy 1: Simple sliding window - keep only recent tokens"""
    
    def __init__(self, max_cache_length: int, nhid: int):
        super().__init__(max_cache_length, nhid)
    
    def manage_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor, 
                    hidden: torch.Tensor, attention_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        current_length = key_cache.size(1)
        
        if current_length <= self.max_cache_length:
            return key_cache, value_cache, hidden
        
        # Keep only the most recent tokens
        new_key_cache = key_cache[:, -self.max_cache_length:]
        new_value_cache = value_cache[:, -self.max_cache_length:]
        new_hidden = hidden[-self.max_cache_length-1:]  # +1 for indexing
        
        return new_key_cache, new_value_cache, new_hidden

class HierarchicalCompressionCache(CacheManager):
    """Strategy 2: Hierarchical compression - recent detailed + old compressed"""
    
    def __init__(self, max_cache_length: int, nhid: int, compression_ratio: float = 0.7):
        super().__init__(max_cache_length, nhid)
        self.compression_ratio = compression_ratio
        self.compressed_size = int(max_cache_length * compression_ratio)
        self.recent_size = max_cache_length - self.compressed_size
        
        # Learnable compression layers
        self.key_compressor = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.LayerNorm(nhid),
            nn.Tanh()
        )
        self.value_compressor = nn.Sequential(
            nn.Linear(nhid, nhid), 
            nn.LayerNorm(nhid),
            nn.Tanh()
        )
    
    def compress_segment(self, keys: torch.Tensor, values: torch.Tensor, 
                        target_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress a segment of cache to target length"""
        if keys.size(1) <= target_length:
            return keys, values
        
        # Apply learnable compression
        compressed_keys = self.key_compressor(keys)
        compressed_values = self.value_compressor(values)
        
        # Use adaptive average pooling to reduce sequence length
        # Transpose for pooling: [batch, seq, hidden] -> [batch, hidden, seq]
        compressed_keys = F.adaptive_avg_pool1d(
            compressed_keys.transpose(1, 2), target_length
        ).transpose(1, 2)
        
        compressed_values = F.adaptive_avg_pool1d(
            compressed_values.transpose(1, 2), target_length  
        ).transpose(1, 2)
        
        return compressed_keys, compressed_values
    
    def manage_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor,
                    hidden: torch.Tensor, attention_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        current_length = key_cache.size(1)
        
        if current_length <= self.max_cache_length:
            return key_cache, value_cache, hidden
        
        # Split into old and recent parts
        split_point = current_length - self.recent_size
        
        # Recent part (keep as-is)
        recent_keys = key_cache[:, split_point:]
        recent_values = value_cache[:, split_point:]
        
        # Old part (compress)
        old_keys = key_cache[:, :split_point]
        old_values = value_cache[:, :split_point]
        
        compressed_keys, compressed_values = self.compress_segment(
            old_keys, old_values, self.compressed_size
        )
        
        # Combine compressed + recent
        new_key_cache = torch.cat([compressed_keys, recent_keys], dim=1)
        new_value_cache = torch.cat([compressed_values, recent_values], dim=1)
        
        # Adjust hidden state
        hidden_trim = min(len(hidden) - 1, new_key_cache.size(1))
        new_hidden = hidden[-hidden_trim-1:]
        
        return new_key_cache, new_value_cache, new_hidden

class AttentionBasedSelectionCache(CacheManager):
    """Strategy 3: Keep tokens that received high attention, discard unused ones"""
    
    def __init__(self, max_cache_length: int, nhid: int, attention_threshold: float = 0.01):
        super().__init__(max_cache_length, nhid)
        self.attention_threshold = attention_threshold
        self.attention_history = None  # Track cumulative attention
    
    def manage_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor,
                    hidden: torch.Tensor, attention_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        current_length = key_cache.size(1)
        
        if current_length <= self.max_cache_length:
            # Update attention history
            if attention_weights is not None:
                if self.attention_history is None:
                    self.attention_history = attention_weights.mean(dim=0).mean(dim=0)  # Average over batch and heads
                else:
                    # Pad or trim to match current length
                    if self.attention_history.size(0) < current_length:
                        pad_size = current_length - self.attention_history.size(0)
                        self.attention_history = torch.cat([
                            self.attention_history, 
                            torch.zeros(pad_size, device=self.attention_history.device)
                        ])
                    
                    # Add current attention (with decay)
                    current_attn = attention_weights.mean(dim=0).mean(dim=0)
                    self.attention_history[:current_length] = (
                        0.9 * self.attention_history[:current_length] + 0.1 * current_attn
                    )
            
            return key_cache, value_cache, hidden
        
        # Need to trim cache based on attention scores
        if self.attention_history is not None and attention_weights is not None:
            # Keep tokens with highest cumulative attention
            _, top_indices = torch.topk(
                self.attention_history[:current_length], 
                k=self.max_cache_length,
                sorted=True
            )
            
            # Select based on attention scores
            new_key_cache = key_cache[:, top_indices]
            new_value_cache = value_cache[:, top_indices]
            
            # Update attention history
            self.attention_history = self.attention_history[top_indices]
            
            # Adjust hidden state (approximate)
            new_hidden = hidden[-self.max_cache_length-1:]
            
        else:
            # Fallback to sliding window if no attention info
            new_key_cache = key_cache[:, -self.max_cache_length:]
            new_value_cache = value_cache[:, -self.max_cache_length:]
            new_hidden = hidden[-self.max_cache_length-1:]
        
        return new_key_cache, new_value_cache, new_hidden

class ExponentialDecayCache(CacheManager):
    """Strategy 4: Gradually fade older representations"""
    
    def __init__(self, max_cache_length: int, nhid: int, decay_factor: float = 0.95):
        super().__init__(max_cache_length, nhid)
        self.decay_factor = decay_factor
    
    def manage_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor,
                    hidden: torch.Tensor, attention_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        current_length = key_cache.size(1)
        
        # Apply exponential decay to older tokens
        if current_length > 1:
            # Create decay weights: [decay^(n-1), decay^(n-2), ..., decay^1, 1.0]
            decay_weights = torch.pow(
                self.decay_factor, 
                torch.arange(current_length - 1, -1, -1, device=key_cache.device)
            ).unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1]
            
            # Apply decay
            key_cache = key_cache * decay_weights
            value_cache = value_cache * decay_weights
        
        # If still too long, use sliding window
        if current_length > self.max_cache_length:
            key_cache = key_cache[:, -self.max_cache_length:]
            value_cache = value_cache[:, -self.max_cache_length:]
            hidden = hidden[-self.max_cache_length-1:]
        
        return key_cache, value_cache, hidden

class ClusteringSummarizationCache(CacheManager):
    """Strategy 5: Group similar tokens and replace with prototypes"""
    
    def __init__(self, max_cache_length: int, nhid: int, n_clusters: int = None):
        super().__init__(max_cache_length, nhid)
        self.n_clusters = n_clusters or (max_cache_length // 4)
        self.recent_window = max_cache_length // 3  # Keep recent tokens as-is
    
    def cluster_and_summarize(self, keys: torch.Tensor, values: torch.Tensor, 
                             n_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cluster similar representations and create prototypes"""
        batch_size, seq_len, hidden_dim = keys.shape
        
        if seq_len <= n_clusters:
            return keys, values
        
        # Simple k-means style clustering (differentiable approximation)
        # Initialize cluster centers
        indices = torch.randperm(seq_len, device=keys.device)[:n_clusters]
        cluster_centers = keys[:, indices]  # [batch, n_clusters, hidden]
        
        # Compute similarities and soft assignments
        similarities = torch.einsum('bsh,bch->bsc', keys, cluster_centers)  # [batch, seq, clusters]
        assignments = F.softmax(similarities / 0.1, dim=2)  # Temperature = 0.1
        
        # Create cluster prototypes (weighted averages)
        prototype_keys = torch.einsum('bsc,bsh->bch', assignments, keys)
        prototype_values = torch.einsum('bsc,bsh->bch', assignments, values)
        
        return prototype_keys, prototype_values
    
    def manage_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor,
                    hidden: torch.Tensor, attention_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        current_length = key_cache.size(1)
        
        if current_length <= self.max_cache_length:
            return key_cache, value_cache, hidden
        
        # Split into old (to be clustered) and recent (keep as-is)
        split_point = current_length - self.recent_window
        
        old_keys = key_cache[:, :split_point]
        old_values = value_cache[:, :split_point]
        recent_keys = key_cache[:, split_point:]
        recent_values = value_cache[:, split_point:]
        
        # Cluster the old part
        clustered_keys, clustered_values = self.cluster_and_summarize(
            old_keys, old_values, self.n_clusters
        )
        
        # Combine clustered + recent
        new_key_cache = torch.cat([clustered_keys, recent_keys], dim=1)
        new_value_cache = torch.cat([clustered_values, recent_values], dim=1)
        
        # Adjust hidden state
        new_cache_length = new_key_cache.size(1)
        new_hidden = hidden[-new_cache_length-1:]
        
        return new_key_cache, new_value_cache, new_hidden

class AdaptiveCache(CacheManager):
    """Strategy 6: Combination approach - adapts strategy based on context"""
    
    def __init__(self, max_cache_length: int, nhid: int):
        super().__init__(max_cache_length, nhid)
        
        # Initialize all strategies
        self.sliding_window = SlidingWindowCache(max_cache_length, nhid)
        self.hierarchical = HierarchicalCompressionCache(max_cache_length, nhid)
        self.attention_based = AttentionBasedSelectionCache(max_cache_length, nhid)
        self.exponential_decay = ExponentialDecayCache(max_cache_length, nhid)
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Linear(nhid, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 strategies
            nn.Softmax(dim=-1)
        )
    
    def manage_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor,
                    hidden: torch.Tensor, attention_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        current_length = key_cache.size(1)
        
        if current_length <= self.max_cache_length:
            return key_cache, value_cache, hidden
        
        # Decide which strategy to use based on current context
        context_repr = hidden[-1].mean(dim=0)  # Average over batch
        strategy_weights = self.strategy_selector(context_repr)
        
        # Apply each strategy
        results = []
        results.append(self.sliding_window.manage_cache(key_cache, value_cache, hidden, attention_weights))
        results.append(self.hierarchical.manage_cache(key_cache, value_cache, hidden, attention_weights))
        results.append(self.attention_based.manage_cache(key_cache, value_cache, hidden, attention_weights))
        results.append(self.exponential_decay.manage_cache(key_cache, value_cache, hidden, attention_weights))
        
        # Weight the results (simplified - just use the top strategy)
        best_strategy = torch.argmax(strategy_weights).item()
        
        return results[best_strategy]

# Usage example in CBR_RNN
class CBR_RNN_WithCacheManagement(nn.Module):
    def __init__(self, cache_strategy_name: str = 'hierarchical', **kwargs):
        super().__init__()
        
        # ... (other CBR_RNN initialization) ...
        
        # Initialize cache management strategy
        cache_strategies = {
            'sliding_window': SlidingWindowCache,
            'hierarchical': HierarchicalCompressionCache, 
            'attention_based': AttentionBasedSelectionCache,
            'exponential_decay': ExponentialDecayCache,
            'clustering': ClusteringSummarizationCache,
            'adaptive': AdaptiveCache
        }
        
        max_cache_length = kwargs.get('max_cache_length', 256)
        nhid = kwargs.get('nhid', 256)
        
        self.cache_manager = cache_strategies[cache_strategy_name](max_cache_length, nhid)
    
    def update_cache_with_management(self, key_cache, value_cache, hidden, 
                                   key_cache_i, value_cache_i, hidden_i, attention_weights=None):
        """Update cache and apply management strategy"""
        
        # Standard cache update
        hidden_i = hidden_i.unsqueeze(0)
        hidden = torch.cat((hidden, hidden_i), dim=0)
        key_cache_i = key_cache_i.unsqueeze(1)
        value_cache_i = value_cache_i.unsqueeze(1)
        key_cache = torch.cat((key_cache, key_cache_i), dim=1)
        value_cache = torch.cat((value_cache, value_cache_i), dim=1)
        
        # Apply cache management
        key_cache, value_cache, hidden = self.cache_manager.manage_cache(
            key_cache, value_cache, hidden, attention_weights
        )
        
        return key_cache, value_cache, hidden

# Performance comparison
def compare_strategies():
    """Compare different cache management strategies"""
    
    strategies = {
        'Sliding Window': 'Simple, constant memory, hard cutoffs',
        'Hierarchical Compression': 'Psychologically plausible, learnable forgetting',
        'Attention-Based Selection': 'Keep important info, discard unused',
        'Exponential Decay': 'Gradual forgetting, smooth degradation', 
        'Clustering/Summarization': 'Semantic compression, prototype formation',
        'Adaptive': 'Context-dependent strategy selection'
    }
    
    print("Cache Management Strategy Comparison:")
    print("=" * 50)
    
    for strategy, description in strategies.items():
        print(f"{strategy:25}: {description}")
    
    print("\nRecommendation for CBR-RNN:")
    print("• Start with Hierarchical Compression (most psycholinguistically motivated)")
    print("• Consider Attention-Based Selection for modeling 'importance' effects")
    print("• Use Adaptive for maximum flexibility (but more complex)")

if __name__ == "__main__":
    compare_strategies()