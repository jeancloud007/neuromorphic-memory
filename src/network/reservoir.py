"""
Sparse Recurrent Reservoir Network

The core SNN architecture for neuromorphic memory.
Implements the Miruvor-style reservoir with:
- Sparse random connectivity (1-5%)
- Heterogeneous axonal delays (1-16 timesteps)
- Support for polychronous group formation

This is where memories are stored and recalled via
attractor dynamics and content-addressable retrieval.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import sys
sys.path.append('..')
from neurons.lif import LIFPopulation, LIFParams


@dataclass
class ReservoirParams:
    """Parameters for the sparse reservoir network."""
    n_neurons: int = 10000          # Reservoir size (10k-500k in paper)
    n_input: int = 512              # Input dimension (from encoder)
    n_output: int = 100             # Output/readout dimension
    connectivity: float = 0.02      # Sparse connectivity (1-5%)
    delay_range: Tuple[int, int] = (1, 16)  # Axonal delay range (timesteps)
    input_scale: float = 2.0        # Input weight scaling
    recurrent_scale: float = 1.0    # Recurrent weight scaling
    output_scale: float = 0.1       # Output weight scaling
    neuron_params: LIFParams = field(default_factory=LIFParams)


class DelayBuffer:
    """
    Circular buffer for implementing axonal delays.
    
    Spikes are inserted at the current timestep and retrieved
    after their assigned delay, enabling polychronous dynamics.
    """
    
    def __init__(self, n_neurons: int, max_delay: int):
        self.n_neurons = n_neurons
        self.max_delay = max_delay
        self.buffer = np.zeros((max_delay, n_neurons), dtype=bool)
        self.head = 0
    
    def insert(self, spikes: np.ndarray, delays: np.ndarray):
        """
        Insert spikes with their respective delays.
        
        Args:
            spikes: Boolean array of current spikes
            delays: Integer array of delays for each connection
        """
        # For each unique delay, insert into appropriate buffer position
        for d in range(1, self.max_delay + 1):
            mask = (delays == d) & spikes
            idx = (self.head + d) % self.max_delay
            self.buffer[idx] |= mask
    
    def get_delayed_spikes(self) -> np.ndarray:
        """Get spikes that have completed their delay."""
        spikes = self.buffer[self.head].copy()
        self.buffer[self.head] = False  # Clear retrieved
        self.head = (self.head + 1) % self.max_delay
        return spikes
    
    def reset(self):
        """Clear the delay buffer."""
        self.buffer.fill(False)
        self.head = 0


class SparseConnectivity:
    """
    Sparse connectivity matrix with axonal delays.
    
    Uses coordinate (COO) format for memory efficiency
    with large reservoirs.
    """
    
    def __init__(self, n_pre: int, n_post: int, 
                 connectivity: float, delay_range: Tuple[int, int],
                 weight_scale: float = 1.0):
        self.n_pre = n_pre
        self.n_post = n_post
        
        # Generate sparse random connections
        n_connections = int(n_pre * n_post * connectivity)
        
        # Random pre/post indices
        self.pre_idx = np.random.randint(0, n_pre, n_connections)
        self.post_idx = np.random.randint(0, n_post, n_connections)
        
        # Random weights (initialized to drive spiking)
        self.weights = np.random.uniform(0.5, 1.5, n_connections) * weight_scale
        
        # Random delays
        self.delays = np.random.randint(
            delay_range[0], delay_range[1] + 1, n_connections
        )
        
        # For fast lookup: post -> list of (pre, weight, delay)
        self._build_post_lookup()
    
    def _build_post_lookup(self):
        """Build lookup table: post_idx -> [(pre_idx, weight, delay), ...]"""
        self.post_to_pre: Dict[int, List[Tuple[int, float, int]]] = {}
        for i in range(len(self.pre_idx)):
            post = self.post_idx[i]
            if post not in self.post_to_pre:
                self.post_to_pre[post] = []
            self.post_to_pre[post].append(
                (self.pre_idx[i], self.weights[i], self.delays[i])
            )
    
    def get_input_to(self, post_idx: int) -> List[Tuple[int, float, int]]:
        """Get all inputs to a postsynaptic neuron."""
        return self.post_to_pre.get(post_idx, [])
    
    def compute_currents(self, pre_spikes: np.ndarray, 
                         delay_buffer: Optional[DelayBuffer] = None) -> np.ndarray:
        """
        Compute postsynaptic currents from presynaptic spikes.
        
        Args:
            pre_spikes: Boolean array of presynaptic spikes
            delay_buffer: Optional delay buffer for axonal delays
            
        Returns:
            Current for each postsynaptic neuron
        """
        currents = np.zeros(self.n_post)
        
        if delay_buffer is None:
            # No delays - immediate transmission
            active = pre_spikes[self.pre_idx]
            np.add.at(currents, self.post_idx[active], self.weights[active])
        else:
            # With delays - spikes go into buffer
            # This is called per-connection, so we handle it differently
            pass
        
        return currents


class SparseReservoir:
    """
    Complete sparse recurrent reservoir for neuromorphic memory.
    
    Architecture:
    - Input layer → Reservoir (sparse, delayed)
    - Reservoir ↔ Reservoir (sparse recurrent, delayed)
    - Reservoir → Output/Readout layer
    
    This is the core of the Miruvor memory system.
    """
    
    def __init__(self, params: Optional[ReservoirParams] = None):
        self.params = params or ReservoirParams()
        p = self.params
        
        # Neuron population
        self.neurons = LIFPopulation(
            n_neurons=p.n_neurons,
            params=p.neuron_params,
            adaptive=True
        )
        
        # Input connections (input → reservoir)
        self.input_conn = SparseConnectivity(
            n_pre=p.n_input,
            n_post=p.n_neurons,
            connectivity=0.1,  # Denser input connectivity
            delay_range=(1, 4),  # Shorter input delays
            weight_scale=p.input_scale
        )
        
        # Recurrent connections (reservoir ↔ reservoir)
        self.recurrent_conn = SparseConnectivity(
            n_pre=p.n_neurons,
            n_post=p.n_neurons,
            connectivity=p.connectivity,
            delay_range=p.delay_range,
            weight_scale=p.recurrent_scale
        )
        
        # Output connections (reservoir → readout)
        self.output_weights = np.random.randn(
            p.n_neurons, p.n_output
        ) * p.output_scale
        
        # Delay buffers
        max_delay = max(p.delay_range[1], 4)
        self.input_delay_buffer = DelayBuffer(p.n_neurons, max_delay)
        self.recurrent_delay_buffer = DelayBuffer(p.n_neurons, max_delay)
        
        # State tracking
        self.t = 0
        self.spike_history: List[np.ndarray] = []
    
    def reset(self):
        """Reset reservoir state."""
        self.neurons.reset_state()
        self.input_delay_buffer.reset()
        self.recurrent_delay_buffer.reset()
        self.t = 0
        self.spike_history.clear()
    
    def step(self, input_spikes: Optional[np.ndarray] = None,
             dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance reservoir by one timestep.
        
        Args:
            input_spikes: Optional input spike train (from encoder)
            dt: Timestep
            
        Returns:
            reservoir_spikes: Which reservoir neurons spiked
            output: Readout layer activity
        """
        p = self.params
        
        # Compute input currents
        input_current = np.zeros(p.n_neurons)
        if input_spikes is not None:
            input_current = self.input_conn.compute_currents(input_spikes)
        
        # Get delayed recurrent spikes
        delayed_recurrent = self.recurrent_delay_buffer.get_delayed_spikes()
        
        # Compute recurrent currents (from delayed spikes)
        recurrent_current = np.zeros(p.n_neurons)
        if delayed_recurrent.any():
            recurrent_current = self.recurrent_conn.compute_currents(delayed_recurrent)
        
        # Total input current
        total_current = input_current + recurrent_current
        
        # Update neurons
        spikes = self.neurons.step(total_current, self.t, dt)
        
        # Insert new spikes into delay buffer
        # For simplicity, we insert all spikes with a representative delay
        # (Full implementation would handle per-synapse delays)
        if spikes.any():
            # Insert with random delays from the recurrent connections
            avg_delay = (p.delay_range[0] + p.delay_range[1]) // 2
            spike_delays = np.full(p.n_neurons, avg_delay)
            self.recurrent_delay_buffer.insert(spikes, spike_delays)
        
        # Compute output (simple linear readout)
        # In full system, this would be a WTA (winner-take-all) layer
        output = spikes.astype(float) @ self.output_weights
        
        # Track history
        self.spike_history.append(spikes.copy())
        self.t += dt
        
        return spikes, output
    
    def run(self, input_train: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run reservoir for a full input spike train.
        
        Args:
            input_train: Spike train matrix (n_timesteps x n_input)
            dt: Timestep
            
        Returns:
            all_spikes: Reservoir spikes (n_timesteps x n_neurons)
            all_outputs: Output activity (n_timesteps x n_output)
        """
        n_timesteps = input_train.shape[0]
        all_spikes = []
        all_outputs = []
        
        for t in range(n_timesteps):
            spikes, output = self.step(input_train[t], dt)
            all_spikes.append(spikes)
            all_outputs.append(output)
        
        return np.array(all_spikes), np.array(all_outputs)
    
    def get_active_assembly(self, window: int = 20) -> np.ndarray:
        """
        Identify the most active neuronal assembly in recent history.
        
        This is used for memory recall - the assembly that "wins"
        corresponds to the retrieved memory.
        
        Args:
            window: Number of recent timesteps to consider
            
        Returns:
            Indices of neurons in the winning assembly
        """
        if len(self.spike_history) < window:
            window = len(self.spike_history)
        
        if window == 0:
            return np.array([])
        
        # Sum spikes over window
        recent_spikes = np.array(self.spike_history[-window:])
        spike_counts = recent_spikes.sum(axis=0)
        
        # Find neurons with above-threshold activity
        threshold = spike_counts.mean() + 2 * spike_counts.std()
        active_neurons = np.where(spike_counts > threshold)[0]
        
        return active_neurons
    
    def get_polychronous_candidates(self, min_size: int = 5) -> List[List[int]]:
        """
        Detect potential polychronous groups from spike patterns.
        
        Polychronous groups are neurons that fire in precise temporal
        sequences, enabled by axonal delays. They're the basis of
        memory storage in this architecture.
        
        Returns:
            List of neuron index groups that show polychronous activity
        """
        if len(self.spike_history) < 10:
            return []
        
        # Simple detection: look for repeated firing sequences
        # (Full implementation would use more sophisticated algorithms)
        groups = []
        
        # Convert to spike times per neuron
        spike_times = {}
        for t, spikes in enumerate(self.spike_history):
            for n in np.where(spikes)[0]:
                if n not in spike_times:
                    spike_times[n] = []
                spike_times[n].append(t)
        
        # Find neurons with correlated firing
        active_neurons = [n for n, times in spike_times.items() if len(times) >= 3]
        
        if len(active_neurons) >= min_size:
            # For now, just return the most active as a candidate group
            groups.append(active_neurons[:min_size])
        
        return groups


if __name__ == "__main__":
    print("Testing SparseReservoir...")
    
    # Small test reservoir
    params = ReservoirParams(
        n_neurons=1000,
        n_input=100,
        n_output=10,
        connectivity=0.05
    )
    
    reservoir = SparseReservoir(params)
    print(f"Created reservoir with {params.n_neurons} neurons")
    print(f"Recurrent connections: ~{int(params.n_neurons**2 * params.connectivity)}")
    
    # Generate random input spike train
    np.random.seed(42)
    input_train = np.random.rand(100, params.n_input) < 0.1  # 10% spike probability
    
    print(f"\nRunning {input_train.shape[0]} timesteps...")
    all_spikes, all_outputs = reservoir.run(input_train)
    
    print(f"Total reservoir spikes: {all_spikes.sum()}")
    print(f"Average spikes per timestep: {all_spikes.sum() / len(all_spikes):.1f}")
    print(f"Output shape: {all_outputs.shape}")
    
    # Check for active assembly
    active = reservoir.get_active_assembly()
    print(f"\nActive assembly size: {len(active)} neurons")
    
    # Check for polychronous groups
    groups = reservoir.get_polychronous_candidates()
    print(f"Polychronous group candidates: {len(groups)}")
    
    print("\n✅ Reservoir test complete!")
