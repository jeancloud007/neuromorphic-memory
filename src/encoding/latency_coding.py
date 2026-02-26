"""
Latency Coding (Time-To-First-Spike) Encoder

Converts continuous feature vectors into spike trains where
feature magnitude maps to spike timing: stronger features
fire earlier, weaker features fire later or not at all.

This is the input encoding scheme used in Miruvor.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LatencyParams:
    """Parameters for latency coding."""
    t_window: float = 100.0     # Encoding window (ms)
    t_min: float = 1.0          # Minimum spike time
    v_thresh: float = 0.1       # Threshold for generating spike
    normalize: bool = True       # Normalize input to [0,1]
    separate_channels: bool = True  # Use separate +/- channels


class LatencyEncoder:
    """
    Encodes feature vectors as spike trains using latency coding.
    
    For each feature dimension:
    - High value -> early spike
    - Low value -> late spike
    - Below threshold -> no spike
    
    Produces sparse, temporally precise representations.
    """
    
    def __init__(self, n_features: int, params: Optional[LatencyParams] = None):
        self.n_features = n_features
        self.params = params or LatencyParams()
        
        # With separate channels, we have 2x neurons (positive and negative)
        if self.params.separate_channels:
            self.n_neurons = n_features * 2
        else:
            self.n_neurons = n_features
    
    def encode(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a feature vector as spike times.
        
        Args:
            features: Input feature vector (shape: n_features) or batch
                     (shape: batch_size x n_features)
                     
        Returns:
            spike_neurons: Which neurons spike (indices or bool array)
            spike_times: When they spike (ms from start of window)
        """
        p = self.params
        
        # Handle batch dimension
        if features.ndim == 1:
            features = features.reshape(1, -1)
        batch_size = features.shape[0]
        
        # Normalize if requested
        if p.normalize:
            # Normalize each sample to [0, 1]
            f_min = features.min(axis=1, keepdims=True)
            f_max = features.max(axis=1, keepdims=True)
            f_range = f_max - f_min
            f_range = np.where(f_range == 0, 1, f_range)  # Avoid divide by zero
            features = (features - f_min) / f_range
        
        if p.separate_channels:
            # Split into positive and negative channels
            pos_features = np.maximum(features, 0)
            neg_features = np.maximum(-features, 0)  # Already normalized, so this handles bipolar input
            
            # For normalized [0,1] input, just use positive channel
            # The negative channel handles cases where original input was negative
            all_features = np.concatenate([pos_features, neg_features], axis=1)
        else:
            all_features = np.abs(features)
        
        # Convert magnitude to time: higher magnitude -> earlier spike
        # t = t_min + (1 - magnitude) * (t_window - t_min)
        spike_times = p.t_min + (1 - all_features) * (p.t_window - p.t_min)
        
        # Threshold: features below threshold don't spike
        spike_mask = all_features >= p.v_thresh
        
        # Set non-spiking neurons to infinity (or NaN)
        spike_times = np.where(spike_mask, spike_times, np.inf)
        
        if batch_size == 1:
            return spike_mask[0], spike_times[0]
        return spike_mask, spike_times
    
    def to_spike_train(self, features: np.ndarray, 
                       dt: float = 1.0) -> np.ndarray:
        """
        Convert features to a full spike train matrix.
        
        Args:
            features: Input feature vector
            dt: Time resolution (ms)
            
        Returns:
            spikes: Binary matrix (n_timesteps x n_neurons)
        """
        p = self.params
        n_timesteps = int(p.t_window / dt)
        
        spike_mask, spike_times = self.encode(features)
        
        # Create spike train matrix
        spikes = np.zeros((n_timesteps, self.n_neurons), dtype=bool)
        
        for i in range(self.n_neurons):
            if spike_mask[i] and spike_times[i] < p.t_window:
                t_idx = int(spike_times[i] / dt)
                if 0 <= t_idx < n_timesteps:
                    spikes[t_idx, i] = True
        
        return spikes


class LatencyDecoder:
    """
    Decodes readout neuron activity back to feature space.
    
    Uses the timing of output spikes or integrated activity
    to reconstruct the retrieved memory.
    """
    
    def __init__(self, n_features: int, params: Optional[LatencyParams] = None):
        self.n_features = n_features
        self.params = params or LatencyParams()
    
    def decode_from_times(self, spike_times: np.ndarray) -> np.ndarray:
        """
        Decode spike times back to features.
        
        Inverse of encoding: early spike -> high value
        """
        p = self.params
        
        # Handle no-spike case
        valid = spike_times < np.inf
        
        # Inverse of encoding formula
        features = np.zeros(len(spike_times))
        features[valid] = 1 - (spike_times[valid] - p.t_min) / (p.t_window - p.t_min)
        
        return features
    
    def decode_from_rates(self, spike_counts: np.ndarray, 
                          duration: float) -> np.ndarray:
        """
        Decode from spike counts/rates (simpler method).
        
        Higher spike count -> higher feature value
        """
        # Normalize by duration and max count
        rates = spike_counts / duration
        max_rate = rates.max()
        if max_rate > 0:
            return rates / max_rate
        return rates


class EmbeddingEncoder:
    """
    Specialized encoder for text/image embeddings.
    
    Takes high-dimensional embedding vectors (e.g., from BERT, CLIP)
    and converts them to spike trains suitable for the SNN reservoir.
    """
    
    def __init__(self, embedding_dim: int, 
                 reduction_dim: Optional[int] = None,
                 params: Optional[LatencyParams] = None):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            reduction_dim: Optional dimensionality reduction
            params: Latency coding parameters
        """
        self.embedding_dim = embedding_dim
        self.reduction_dim = reduction_dim or embedding_dim
        self.params = params or LatencyParams()
        
        # Random projection for dimensionality reduction (if needed)
        if reduction_dim and reduction_dim < embedding_dim:
            self.projection = np.random.randn(embedding_dim, reduction_dim)
            self.projection /= np.sqrt(embedding_dim)  # Scale
        else:
            self.projection = None
        
        self.encoder = LatencyEncoder(self.reduction_dim, self.params)
    
    def encode(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode an embedding vector.
        
        Args:
            embedding: Dense embedding vector
            
        Returns:
            spike_mask, spike_times
        """
        # Optional dimensionality reduction
        if self.projection is not None:
            embedding = embedding @ self.projection
        
        return self.encoder.encode(embedding)
    
    def to_spike_train(self, embedding: np.ndarray, 
                       dt: float = 1.0) -> np.ndarray:
        """Convert embedding to full spike train."""
        if self.projection is not None:
            embedding = embedding @ self.projection
        return self.encoder.to_spike_train(embedding, dt)


if __name__ == "__main__":
    # Test basic latency coding
    print("Testing latency encoder...")
    encoder = LatencyEncoder(n_features=10)
    
    # Create test features
    features = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.2, 0.4, 0.6, 0.8])
    
    spike_mask, spike_times = encoder.encode(features)
    print(f"Features: {features}")
    print(f"Spike mask: {spike_mask[:10]}")  # First 10 (positive channel)
    print(f"Spike times: {np.round(spike_times[:10], 1)}")
    
    print("\nSpike train visualization:")
    train = encoder.to_spike_train(features)
    print(f"Shape: {train.shape} (timesteps x neurons)")
    print(f"Total spikes: {train.sum()}")
    print(f"Sparsity: {1 - train.mean():.4f}")
    
    # Test embedding encoder
    print("\nTesting embedding encoder (BERT-like)...")
    emb_encoder = EmbeddingEncoder(embedding_dim=768, reduction_dim=256)
    fake_embedding = np.random.randn(768)
    spike_mask, spike_times = emb_encoder.encode(fake_embedding)
    print(f"Input dim: 768, Reduced dim: 256")
    print(f"Neurons that spike: {spike_mask.sum()} / {len(spike_mask)}")
