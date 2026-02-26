# Miruvor Paper Notes

## Key Concepts

### Latency Coding (Time-To-First-Spike)
- Feature magnitude → spike timing
- Stronger features fire earlier
- Sparse representation (typically one spike per feature)
- Positive/negative values via separate channels

### Reservoir Architecture
- 10,000 - 500,000 neurons (scale with application)
- Sparse connectivity: 1-5%
- Axonal delays: 1-16 timesteps
- Adaptive LIF (ALIF) neurons preferred

### Polychronous Groups
- Dynamically formed neuron assemblies
- Fire in precisely timed chains
- Encode complex spatiotemporal patterns
- Exponential memory capacity (combinatorial growth)

### Learning

**Offline Phase:**
1. Present historical data as spike trains
2. Hebbian plasticity strengthens co-active connections
3. STDP refines weights:
   - Pre before post → potentiate
   - Post before pre → depress
4. Creates "attractor basins" for memories

**Online Phase:**
1. New inputs trigger local STDP updates
2. Network adapts to novel stimuli
3. EWC-style consolidation protects core memories

### Retrieval Mechanism
1. Query encoded as spike train
2. Injected into reservoir
3. Matching synaptic pathways activate
4. Polychronous groups "light up"
5. Attractor dynamics complete partial patterns
6. WTA selects most active assembly
7. Assembly ID = recalled memory

### Advantages Over Vector Stores
| Aspect | Vector Store | Miruvor SNN |
|--------|-------------|-------------|
| Retrieval | Similarity search | Synaptic activation |
| Pattern completion | No | Yes (attractors) |
| Energy | High (dense compute) | Low (event-driven) |
| Latency | ms-s | μs |
| Continual learning | Catastrophic forgetting | Built-in adaptation |

## Implementation Considerations

### Neuron Model (LIF)
```
τ_m * dV/dt = -(V - V_rest) + R * I
if V > V_thresh:
    spike
    V = V_reset
```

### STDP Rule
```
Δw = A_+ * exp(-Δt / τ_+)  if Δt > 0 (pre before post)
Δw = -A_- * exp(Δt / τ_-)  if Δt < 0 (post before pre)
```

### Hardware Targets
- Intel Loihi 2: On-chip STDP, programmable delays
- BrainChip Akida: Edge-optimized
- BrainScaleS-2: Analog neuromorphic

### Mapping Challenges
- Partitioning large SNNs to crossbar resources
- Minimizing inter-cluster communication
- Preserving synaptic structure

## Questions for Implementation

1. **Reservoir size**: Start small (1000 neurons) or go big?
2. **Connectivity**: Random sparse, or structured?
3. **Integration**: Replace vector store entirely, or hybrid?
4. **Simulation**: CPU/GPU first, or target neuromorphic hardware?
5. **Encoding**: How to handle our existing embedding space?

## Python Libraries to Consider

- **Norse**: PyTorch-based SNN library
- **snnTorch**: Another PyTorch SNN framework
- **Brian2**: Flexible SNN simulator
- **BindsNET**: ML-oriented SNN framework
- **Nengo**: Neural engineering framework (supports Loihi)

## Action Items

- [ ] Choose simulation framework
- [ ] Implement basic LIF reservoir
- [ ] Test latency coding on embeddings
- [ ] Benchmark recall accuracy vs vector search
- [ ] Profile energy/latency
