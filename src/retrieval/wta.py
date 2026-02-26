"""
Winner-Take-All (WTA) Selection Layer

Implements competitive selection to identify the winning
neuronal assembly (memory) from reservoir activity.

In the Miruvor architecture, WTA is how we determine which
memory was recalled - the assembly with maximal activation wins.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class WTAParams:
    """Parameters for winner-take-all selection."""
    n_memories: int = 100           # Number of distinct memories
    n_neurons_per_memory: int = 50  # Neurons per memory assembly
    threshold_factor: float = 2.0   # Std devs above mean to be "active"
    min_confidence: float = 0.3     # Minimum confidence to return match
    temporal_window: int = 20       # Timesteps to integrate over


class WinnerTakeAll:
    """
    Winner-take-all layer for memory selection.
    
    Maps reservoir activity to memory IDs by:
    1. Tracking which neurons belong to which memory assemblies
    2. Counting spikes per assembly over a temporal window
    3. Selecting the assembly with maximum activation
    4. Computing confidence based on activation margin
    """
    
    def __init__(self, n_reservoir: int, params: Optional[WTAParams] = None):
        """
        Args:
            n_reservoir: Number of neurons in reservoir
            params: WTA parameters
        """
        self.n_reservoir = n_reservoir
        self.params = params or WTAParams()
        p = self.params
        
        # Assembly membership: which neurons belong to each memory
        # Initially random assignment (will be learned during imprinting)
        self.assemblies: List[np.ndarray] = []
        self._init_random_assemblies()
        
        # Activity tracking
        self.activity_buffer: List[np.ndarray] = []
        self.assembly_counts = np.zeros(p.n_memories)
    
    def _init_random_assemblies(self):
        """Initialize random assembly assignments."""
        p = self.params
        
        # Each memory gets a random subset of neurons
        # Assemblies can overlap (polychronous groups do in biology)
        all_neurons = np.arange(self.n_reservoir)
        
        for _ in range(p.n_memories):
            # Random subset of neurons for this memory
            assembly = np.random.choice(
                all_neurons, 
                size=min(p.n_neurons_per_memory, self.n_reservoir),
                replace=False
            )
            self.assemblies.append(assembly)
    
    def assign_assembly(self, memory_id: int, neuron_indices: np.ndarray):
        """
        Manually assign neurons to a memory assembly.
        
        Called during imprinting to set which neurons encode a memory.
        
        Args:
            memory_id: Memory index
            neuron_indices: Which reservoir neurons form this assembly
        """
        if memory_id >= len(self.assemblies):
            # Extend assemblies list
            while len(self.assemblies) <= memory_id:
                self.assemblies.append(np.array([]))
        
        self.assemblies[memory_id] = neuron_indices
    
    def update(self, spikes: np.ndarray):
        """
        Update activity tracking with new spike data.
        
        Args:
            spikes: Boolean array of current reservoir spikes
        """
        p = self.params
        
        # Add to buffer
        self.activity_buffer.append(spikes.copy())
        
        # Trim to window size
        if len(self.activity_buffer) > p.temporal_window:
            self.activity_buffer.pop(0)
        
        # Recompute assembly counts
        self._compute_assembly_activity()
    
    def _compute_assembly_activity(self):
        """Compute spike counts per assembly over the temporal window."""
        p = self.params
        
        if not self.activity_buffer:
            self.assembly_counts = np.zeros(p.n_memories)
            return
        
        # Sum spikes over window
        total_spikes = np.zeros(self.n_reservoir)
        for spikes in self.activity_buffer:
            total_spikes += spikes.astype(float)
        
        # Count per assembly
        self.assembly_counts = np.zeros(len(self.assemblies))
        for i, assembly in enumerate(self.assemblies):
            if len(assembly) > 0:
                self.assembly_counts[i] = total_spikes[assembly].sum()
    
    def select_winner(self) -> Tuple[int, float]:
        """
        Select the winning memory based on assembly activity.
        
        Returns:
            memory_id: Index of the winning memory (-1 if no clear winner)
            confidence: Confidence score (0-1)
        """
        p = self.params
        
        if len(self.assembly_counts) == 0 or self.assembly_counts.max() == 0:
            return -1, 0.0
        
        # Find winner
        winner = np.argmax(self.assembly_counts)
        max_count = self.assembly_counts[winner]
        
        # Compute confidence based on margin over second place
        sorted_counts = np.sort(self.assembly_counts)[::-1]
        if len(sorted_counts) > 1 and sorted_counts[0] > 0:
            margin = (sorted_counts[0] - sorted_counts[1]) / sorted_counts[0]
        else:
            margin = 1.0
        
        # Also factor in absolute activation level
        expected_max = p.n_neurons_per_memory * p.temporal_window * 0.1  # ~10% firing
        activation_score = min(max_count / expected_max, 1.0)
        
        # Combined confidence
        confidence = margin * activation_score
        
        if confidence < p.min_confidence:
            return -1, confidence
        
        return int(winner), float(confidence)
    
    def get_top_k(self, k: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-k memory candidates with scores.
        
        Useful for debugging or when confidence is low.
        
        Returns:
            List of (memory_id, score) tuples
        """
        # Get indices sorted by activity
        sorted_indices = np.argsort(self.assembly_counts)[::-1]
        
        results = []
        max_count = self.assembly_counts.max() if self.assembly_counts.max() > 0 else 1
        
        for i in sorted_indices[:k]:
            score = self.assembly_counts[i] / max_count
            results.append((int(i), float(score)))
        
        return results
    
    def reset(self):
        """Clear activity buffer."""
        self.activity_buffer.clear()
        self.assembly_counts = np.zeros(len(self.assemblies))


class SoftWTA(WinnerTakeAll):
    """
    Soft winner-take-all with lateral inhibition.
    
    Instead of hard selection, provides probability distribution
    over memories. More biologically plausible and useful for
    uncertain recalls.
    """
    
    def __init__(self, n_reservoir: int, params: Optional[WTAParams] = None,
                 temperature: float = 1.0):
        super().__init__(n_reservoir, params)
        self.temperature = temperature
    
    def get_probabilities(self) -> np.ndarray:
        """
        Get softmax probability distribution over memories.
        
        Returns:
            Probability for each memory (sums to 1)
        """
        if self.assembly_counts.max() == 0:
            # Uniform if no activity
            return np.ones(len(self.assemblies)) / len(self.assemblies)
        
        # Softmax with temperature
        scaled = self.assembly_counts / self.temperature
        scaled = scaled - scaled.max()  # Numerical stability
        exp_counts = np.exp(scaled)
        
        return exp_counts / exp_counts.sum()
    
    def sample_winner(self) -> Tuple[int, float]:
        """
        Sample a winner from the probability distribution.
        
        Useful for exploration or when multiple memories are plausible.
        """
        probs = self.get_probabilities()
        winner = np.random.choice(len(probs), p=probs)
        return int(winner), float(probs[winner])


if __name__ == "__main__":
    print("Testing Winner-Take-All layer...")
    
    # Create WTA for a 1000-neuron reservoir with 10 memories
    params = WTAParams(n_memories=10, n_neurons_per_memory=50)
    wta = WinnerTakeAll(n_reservoir=1000, params=params)
    
    print(f"Initialized {params.n_memories} memory assemblies")
    print(f"Neurons per assembly: {params.n_neurons_per_memory}")
    
    # Simulate some reservoir activity
    # Memory 3's assembly is highly active
    np.random.seed(42)
    
    for t in range(30):
        spikes = np.random.rand(1000) < 0.02  # 2% baseline
        
        # Boost activity in memory 3's assembly
        if t >= 10:
            spikes[wta.assemblies[3]] = np.random.rand(50) < 0.3  # 30% in assembly 3
        
        wta.update(spikes)
    
    # Check winner
    winner, confidence = wta.select_winner()
    print(f"\nWinner: Memory {winner} (confidence: {confidence:.2f})")
    
    # Get top candidates
    print("\nTop 5 candidates:")
    for mem_id, score in wta.get_top_k(5):
        print(f"  Memory {mem_id}: {score:.3f}")
    
    # Test soft WTA
    print("\nTesting Soft WTA...")
    soft_wta = SoftWTA(n_reservoir=1000, params=params, temperature=0.5)
    soft_wta.activity_buffer = wta.activity_buffer.copy()
    soft_wta._compute_assembly_activity()
    
    probs = soft_wta.get_probabilities()
    print(f"Probability distribution (top 5):")
    top_indices = np.argsort(probs)[::-1][:5]
    for i in top_indices:
        print(f"  Memory {i}: {probs[i]:.3f}")
    
    print("\n✅ WTA test complete!")
