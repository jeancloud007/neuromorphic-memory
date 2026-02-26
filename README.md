# Neuromorphic Memory System

A collaborative implementation of the Miruvor-inspired spiking neural network memory architecture for AI agents.

## Team
- **Jean** - Architecture & coordination
- **Jared** - TBD
- **Samantha** - TBD

## Overview

This project implements a neuromorphic memory system based on the Miruvor research paper. The goal is to replace or augment traditional vector-based memory (embeddings + similarity search) with a spiking neural network approach that provides:

- **Content-addressable recall** via synaptic activation (not similarity search)
- **Pattern completion** through attractor dynamics
- **Continual learning** without catastrophic forgetting
- **Ultra-low latency** retrieval
- **Energy efficiency** for edge deployment

## Architecture

### 1. Spike Encoding (Latency Coding)
Convert input features to spike trains using time-to-first-spike encoding:
- Stronger features → earlier spikes
- Weaker features → later spikes or no spike
- Produces sparse, temporally precise representations

### 2. Reservoir Network
Sparse recurrent SNN with:
- Leaky integrate-and-fire (LIF) neurons
- 1-5% sparse connectivity
- Axonal delays (1-16 timesteps)
- Support for polychronous group formation

### 3. Learning Phases
**Offline (Imprinting):**
- Hebbian plasticity for structure formation
- STDP refinement for weight tuning

**Online (Adaptation):**
- Continual STDP for new memories
- EWC-style consolidation to protect core memories

### 4. Retrieval
- Inject query as spike train
- Network evolves dynamically
- Polychronous groups compete
- Winner-take-all selection returns memory ID

## Implementation Plan

### Phase 1: Core SNN Engine
- [ ] LIF neuron model
- [ ] Sparse connectivity with delays
- [ ] Basic spike propagation

### Phase 2: Encoding/Decoding
- [ ] Latency coding for embeddings
- [ ] Text → spike train conversion
- [ ] Spike train → retrieval ID

### Phase 3: Learning Rules
- [ ] Hebbian plasticity
- [ ] STDP implementation
- [ ] Weight consolidation

### Phase 4: Memory Interface
- [ ] Store/recall API
- [ ] Integration with existing agent memory
- [ ] Benchmarking vs vector stores

## Directory Structure

```
neuromorphic-memory/
├── README.md
├── docs/
│   └── miruvor-paper-notes.md
├── src/
│   ├── neurons/       # LIF, ALIF neuron models
│   ├── network/       # Reservoir architecture
│   ├── encoding/      # Latency coding
│   ├── learning/      # STDP, Hebbian
│   └── memory/        # High-level API
├── tests/
└── examples/
```

## References

- Miruvor paper (attached to project)
- Izhikevich (2006) - Polychronization
- Hardware: Intel Loihi 2, BrainChip Akida, BrainScaleS-2

## Status

🚧 **Project initialized** - Awaiting team coordination
