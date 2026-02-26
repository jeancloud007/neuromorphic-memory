"""
Leaky Integrate-and-Fire (LIF) Neuron Model

The fundamental building block for the SNN reservoir.
Based on the Miruvor paper's use of adaptive LIF neurons.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LIFParams:
    """Parameters for LIF neuron dynamics."""
    tau_m: float = 20.0        # Membrane time constant (ms)
    v_rest: float = -65.0      # Resting potential (mV)
    v_thresh: float = -50.0    # Spike threshold (mV)
    v_reset: float = -70.0     # Reset potential (mV)
    r_m: float = 10.0          # Membrane resistance (MΩ)
    t_refract: float = 2.0     # Refractory period (ms)


class LIFNeuron:
    """
    Single Leaky Integrate-and-Fire neuron.
    
    Membrane dynamics:
        τ_m * dV/dt = -(V - V_rest) + R_m * I
        
    When V > V_thresh:
        - Emit spike
        - V = V_reset
        - Enter refractory period
    """
    
    def __init__(self, params: Optional[LIFParams] = None, neuron_id: int = 0):
        self.params = params or LIFParams()
        self.neuron_id = neuron_id
        self.reset_state()
    
    def reset_state(self):
        """Reset neuron to initial state."""
        self.v = self.params.v_rest
        self.t_last_spike: Optional[float] = None
        self.spike_history: List[float] = []
    
    def is_refractory(self, t: float) -> bool:
        """Check if neuron is in refractory period."""
        if self.t_last_spike is None:
            return False
        return (t - self.t_last_spike) < self.params.t_refract
    
    def step(self, I: float, t: float, dt: float = 1.0) -> bool:
        """
        Advance neuron by one timestep.
        
        Args:
            I: Input current (sum of synaptic inputs)
            t: Current time (ms)
            dt: Timestep (ms)
            
        Returns:
            True if neuron spiked, False otherwise
        """
        # Check refractory
        if self.is_refractory(t):
            return False
        
        # Integrate membrane equation (Euler method)
        p = self.params
        dv = (-(self.v - p.v_rest) + p.r_m * I) / p.tau_m * dt
        self.v += dv
        
        # Check threshold
        if self.v >= p.v_thresh:
            self.v = p.v_reset
            self.t_last_spike = t
            self.spike_history.append(t)
            return True
        
        return False


class ALIFNeuron(LIFNeuron):
    """
    Adaptive LIF neuron with spike-frequency adaptation.
    
    Adds an adaptation current that increases with each spike,
    making the neuron less likely to fire in rapid succession.
    Used in Miruvor for more biologically plausible dynamics.
    """
    
    def __init__(self, params: Optional[LIFParams] = None, 
                 neuron_id: int = 0,
                 tau_adapt: float = 100.0,
                 delta_adapt: float = 0.5):
        super().__init__(params, neuron_id)
        self.tau_adapt = tau_adapt  # Adaptation time constant
        self.delta_adapt = delta_adapt  # Adaptation increment per spike
        self.w_adapt = 0.0  # Adaptation variable
    
    def reset_state(self):
        """Reset neuron including adaptation."""
        super().reset_state()
        self.w_adapt = 0.0
    
    def step(self, I: float, t: float, dt: float = 1.0) -> bool:
        """
        Advance adaptive neuron by one timestep.
        
        The adaptation current opposes spiking, decaying exponentially
        between spikes and jumping up after each spike.
        """
        # Check refractory
        if self.is_refractory(t):
            # Still decay adaptation during refractory
            self.w_adapt *= np.exp(-dt / self.tau_adapt)
            return False
        
        # Decay adaptation
        self.w_adapt *= np.exp(-dt / self.tau_adapt)
        
        # Integrate with adaptation current subtracted
        p = self.params
        effective_I = I - self.w_adapt
        dv = (-(self.v - p.v_rest) + p.r_m * effective_I) / p.tau_m * dt
        self.v += dv
        
        # Check threshold
        if self.v >= p.v_thresh:
            self.v = p.v_reset
            self.t_last_spike = t
            self.spike_history.append(t)
            self.w_adapt += self.delta_adapt  # Increase adaptation
            return True
        
        return False


class LIFPopulation:
    """
    Population of LIF neurons for efficient batch simulation.
    
    Uses vectorized operations for performance when simulating
    large reservoirs (10k-500k neurons as in Miruvor).
    """
    
    def __init__(self, n_neurons: int, params: Optional[LIFParams] = None,
                 adaptive: bool = True):
        self.n = n_neurons
        self.params = params or LIFParams()
        self.adaptive = adaptive
        self.reset_state()
    
    def reset_state(self):
        """Reset all neurons."""
        p = self.params
        self.v = np.full(self.n, p.v_rest)
        self.t_last_spike = np.full(self.n, -np.inf)
        self.spikes = np.zeros(self.n, dtype=bool)
        
        if self.adaptive:
            self.w_adapt = np.zeros(self.n)
    
    def step(self, I: np.ndarray, t: float, dt: float = 1.0) -> np.ndarray:
        """
        Advance population by one timestep.
        
        Args:
            I: Input currents for each neuron (shape: n_neurons)
            t: Current time
            dt: Timestep
            
        Returns:
            Boolean array of which neurons spiked
        """
        p = self.params
        
        # Find neurons not in refractory
        not_refract = (t - self.t_last_spike) >= p.t_refract
        
        # Decay adaptation (if adaptive)
        if self.adaptive:
            self.w_adapt *= np.exp(-dt / 100.0)  # tau_adapt = 100
            effective_I = I - self.w_adapt
        else:
            effective_I = I
        
        # Integrate membrane equation (vectorized)
        dv = (-(self.v - p.v_rest) + p.r_m * effective_I) / p.tau_m * dt
        self.v = np.where(not_refract, self.v + dv, self.v)
        
        # Check threshold
        self.spikes = self.v >= p.v_thresh
        
        # Reset spiking neurons
        self.v = np.where(self.spikes, p.v_reset, self.v)
        self.t_last_spike = np.where(self.spikes, t, self.t_last_spike)
        
        # Increase adaptation for spiking neurons
        if self.adaptive:
            self.w_adapt = np.where(self.spikes, self.w_adapt + 0.5, self.w_adapt)
        
        return self.spikes


if __name__ == "__main__":
    # Quick test
    print("Testing LIF neuron...")
    neuron = LIFNeuron()
    spikes = []
    for t in range(100):
        I = 2.0 if 20 <= t < 80 else 0.0  # Current injection
        if neuron.step(I, t):
            spikes.append(t)
    print(f"Spike times: {spikes}")
    
    print("\nTesting ALIF neuron...")
    alif = ALIFNeuron()
    spikes = []
    for t in range(100):
        I = 2.0 if 20 <= t < 80 else 0.0
        if alif.step(I, t):
            spikes.append(t)
    print(f"ALIF spike times (fewer due to adaptation): {spikes}")
    
    print("\nTesting population...")
    pop = LIFPopulation(1000)
    I = np.random.randn(1000) * 0.5 + 1.5
    total_spikes = 0
    for t in range(100):
        spikes = pop.step(I, t)
        total_spikes += spikes.sum()
    print(f"Total spikes from 1000 neurons over 100 timesteps: {total_spikes}")
