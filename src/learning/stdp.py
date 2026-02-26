"""
Spike-Timing-Dependent Plasticity (STDP)

The core learning rule for the neuromorphic memory system.
STDP adjusts synaptic weights based on the precise timing
of pre- and post-synaptic spikes:

- Pre fires before post → Potentiate (strengthen)
- Post fires before pre → Depress (weaken)

This is how the system forms and refines memory attractors.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class STDPParams:
    """Parameters for STDP learning rule."""
    a_plus: float = 0.005      # Potentiation amplitude
    a_minus: float = 0.005     # Depression amplitude
    tau_plus: float = 20.0     # Potentiation time constant (ms)
    tau_minus: float = 20.0    # Depression time constant (ms)
    w_max: float = 1.0         # Maximum weight
    w_min: float = 0.0         # Minimum weight
    

class STDPLearning:
    """
    Implements spike-timing-dependent plasticity.
    
    The STDP rule:
        Δw = A+ * exp(-Δt / τ+)   if Δt > 0 (pre before post)
        Δw = -A- * exp(Δt / τ-)   if Δt < 0 (post before pre)
        
    Where Δt = t_post - t_pre
    """
    
    def __init__(self, n_pre: int, n_post: int, 
                 params: Optional[STDPParams] = None,
                 connectivity: float = 0.05):
        """
        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            params: STDP parameters
            connectivity: Sparse connectivity fraction (0-1)
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or STDPParams()
        
        # Initialize sparse weight matrix
        self.connectivity = connectivity
        self.weights = self._init_weights()
        
        # Track spike times for STDP computation
        self.pre_traces = np.zeros(n_pre)   # Eligibility traces
        self.post_traces = np.zeros(n_post)
    
    def _init_weights(self) -> np.ndarray:
        """Initialize sparse random weights."""
        p = self.params
        
        # Create sparse connectivity mask
        mask = np.random.rand(self.n_pre, self.n_post) < self.connectivity
        
        # Initialize weights randomly where connected
        weights = np.random.uniform(0.3 * p.w_max, 0.7 * p.w_max, 
                                    (self.n_pre, self.n_post))
        weights = weights * mask
        
        return weights
    
    def compute_stdp_update(self, delta_t: float) -> float:
        """
        Compute STDP weight change for a given spike timing difference.
        
        Args:
            delta_t: t_post - t_pre (ms)
            
        Returns:
            Weight change (can be positive or negative)
        """
        p = self.params
        
        if delta_t > 0:  # Pre before post -> potentiate
            return p.a_plus * np.exp(-delta_t / p.tau_plus)
        elif delta_t < 0:  # Post before pre -> depress
            return -p.a_minus * np.exp(delta_t / p.tau_minus)
        else:
            return 0.0
    
    def update_traces(self, pre_spikes: np.ndarray, 
                      post_spikes: np.ndarray, dt: float = 1.0):
        """
        Update eligibility traces based on current spikes.
        
        Traces decay exponentially and jump up on spikes.
        """
        p = self.params
        
        # Decay traces
        self.pre_traces *= np.exp(-dt / p.tau_plus)
        self.post_traces *= np.exp(-dt / p.tau_minus)
        
        # Jump up on spikes
        self.pre_traces += pre_spikes.astype(float)
        self.post_traces += post_spikes.astype(float)
    
    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
             dt: float = 1.0) -> np.ndarray:
        """
        Apply STDP update for one timestep.
        
        Uses trace-based implementation for efficiency:
        - When post spikes, potentiate based on pre trace
        - When pre spikes, depress based on post trace
        
        Args:
            pre_spikes: Boolean array of presynaptic spikes
            post_spikes: Boolean array of postsynaptic spikes
            dt: Timestep
            
        Returns:
            Updated weight matrix
        """
        p = self.params
        
        # Potentiation: post spike + pre trace
        # Only where both connected AND post spiked AND pre has trace
        if post_spikes.any():
            # Weight change = A+ * pre_trace for spiking post neurons
            dw_pot = np.outer(self.pre_traces, post_spikes.astype(float)) * p.a_plus
            self.weights += dw_pot * (self.weights > 0)  # Only update existing connections
        
        # Depression: pre spike + post trace
        if pre_spikes.any():
            # Weight change = -A- * post_trace for spiking pre neurons
            dw_dep = np.outer(pre_spikes.astype(float), self.post_traces) * p.a_minus
            self.weights -= dw_dep * (self.weights > 0)
        
        # Clip weights to bounds
        self.weights = np.clip(self.weights, p.w_min, p.w_max)
        
        # Update traces
        self.update_traces(pre_spikes, post_spikes, dt)
        
        return self.weights


class HebbianLearning:
    """
    Simple Hebbian learning for initial structure formation.
    
    "Neurons that fire together wire together."
    
    Used in the offline imprinting phase before STDP refinement.
    """
    
    def __init__(self, n_neurons: int, 
                 learning_rate: float = 0.01,
                 connectivity: float = 0.05):
        self.n_neurons = n_neurons
        self.lr = learning_rate
        
        # Sparse weight matrix (self-connections excluded)
        mask = np.random.rand(n_neurons, n_neurons) < connectivity
        np.fill_diagonal(mask, False)  # No self-connections
        
        self.weights = np.random.uniform(0.1, 0.3, (n_neurons, n_neurons)) * mask
    
    def step(self, spikes: np.ndarray) -> np.ndarray:
        """
        Apply Hebbian update based on co-activation.
        
        Δw_ij = η * x_i * x_j
        
        Args:
            spikes: Boolean array of which neurons spiked
            
        Returns:
            Updated weight matrix
        """
        # Outer product of spikes
        spike_float = spikes.astype(float)
        dw = self.lr * np.outer(spike_float, spike_float)
        
        # Only update existing connections
        self.weights += dw * (self.weights > 0)
        
        # Normalize to prevent unbounded growth
        max_w = self.weights.max()
        if max_w > 1.0:
            self.weights /= max_w
        
        return self.weights


class EWCConsolidation:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.
    
    Protects important weights (those critical to existing memories)
    during online learning of new patterns.
    """
    
    def __init__(self, importance_decay: float = 0.99):
        self.importance_decay = importance_decay
        self.fisher_info = None  # Importance of each weight
        self.optimal_weights = None  # Weights after learning task
    
    def compute_fisher(self, weights: np.ndarray, 
                       gradients: List[np.ndarray]) -> np.ndarray:
        """
        Estimate Fisher information matrix (diagonal approximation).
        
        Fisher info tells us which weights are important for current task.
        """
        if not gradients:
            return np.zeros_like(weights)
        
        # Squared gradients averaged over samples
        fisher = np.mean([g ** 2 for g in gradients], axis=0)
        return fisher
    
    def consolidate(self, weights: np.ndarray, 
                    gradients: List[np.ndarray]):
        """
        Mark current weights as important for consolidation.
        
        Called after learning a new memory pattern.
        """
        new_fisher = self.compute_fisher(weights, gradients)
        
        if self.fisher_info is None:
            self.fisher_info = new_fisher
        else:
            # Accumulate importance with decay
            self.fisher_info = (self.importance_decay * self.fisher_info + 
                               new_fisher)
        
        self.optimal_weights = weights.copy()
    
    def penalty(self, weights: np.ndarray, 
                lambda_ewc: float = 100.0) -> float:
        """
        Compute EWC penalty for current weights.
        
        Penalizes deviation from optimal weights proportional
        to their importance.
        """
        if self.fisher_info is None:
            return 0.0
        
        deviation = weights - self.optimal_weights
        return lambda_ewc * 0.5 * np.sum(self.fisher_info * deviation ** 2)
    
    def regularized_update(self, weights: np.ndarray, 
                          gradient: np.ndarray,
                          learning_rate: float = 0.01,
                          lambda_ewc: float = 100.0) -> np.ndarray:
        """
        Apply gradient update with EWC regularization.
        
        Protects important weights from large changes.
        """
        if self.fisher_info is None:
            # No consolidation yet, just regular update
            return weights + learning_rate * gradient
        
        # EWC gradient: penalize deviation from optimal
        ewc_grad = lambda_ewc * self.fisher_info * (weights - self.optimal_weights)
        
        # Combined update
        return weights + learning_rate * (gradient - ewc_grad)


if __name__ == "__main__":
    # Test STDP
    print("Testing STDP learning...")
    stdp = STDPLearning(n_pre=100, n_post=100, connectivity=0.1)
    
    print(f"Initial weights: mean={stdp.weights.mean():.4f}, "
          f"nonzero={np.count_nonzero(stdp.weights)}")
    
    # Simulate correlated spiking (pre tends to fire before post)
    np.random.seed(42)
    for t in range(1000):
        # Correlated pattern: pre fires, then post fires shortly after
        pre_spikes = np.random.rand(100) < 0.05
        post_spikes = np.roll(pre_spikes, 2)  # Post fires 2ms later
        stdp.step(pre_spikes, post_spikes)
    
    print(f"After correlated training: mean={stdp.weights.mean():.4f}, "
          f"nonzero={np.count_nonzero(stdp.weights)}")
    
    # Test Hebbian
    print("\nTesting Hebbian learning...")
    hebb = HebbianLearning(n_neurons=100, connectivity=0.1)
    print(f"Initial: mean={hebb.weights.mean():.4f}")
    
    # Train on co-activation pattern
    pattern = np.zeros(100, dtype=bool)
    pattern[10:20] = True  # Neurons 10-19 fire together
    
    for _ in range(100):
        hebb.step(pattern)
    
    print(f"After training: mean={hebb.weights.mean():.4f}")
    print(f"Weights within pattern (10-19): {hebb.weights[10:20, 10:20].mean():.4f}")
    print(f"Weights outside pattern: {hebb.weights[:10, :10].mean():.4f}")
