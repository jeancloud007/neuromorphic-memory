"""
High-Level Neuromorphic Memory API

The main interface for storing and recalling memories.
Ties together: encoding, reservoir, learning, WTA, recall.

Usage:
    memory = NeuromorphicMemory()
    
    # Store memories
    memory.store("The capital of France is Paris", memory_id=0)
    memory.store("Python is a programming language", memory_id=1)
    
    # Recall
    result = memory.recall("What is the capital of France?")
    print(result.content)  # "The capital of France is Paris"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .wta import WinnerTakeAll, WTAParams
    from .recall import PatternRecall, RecallResult, RecallParams
except ImportError:
    from wta import WinnerTakeAll, WTAParams
    from recall import PatternRecall, RecallResult, RecallParams


@dataclass
class MemoryEntry:
    """A single memory entry."""
    memory_id: int
    content: str                      # Original text/content
    embedding: np.ndarray             # Vector embedding
    created_at: float                 # Unix timestamp
    access_count: int = 0             # How many times recalled
    last_accessed: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for neuromorphic memory system."""
    # Reservoir parameters
    n_reservoir: int = 10000
    connectivity: float = 0.02
    delay_range: tuple = (1, 16)
    
    # Memory parameters
    max_memories: int = 1000
    neurons_per_memory: int = 100
    
    # Encoding parameters
    encoding_window: float = 100.0
    
    # Learning parameters
    imprint_repetitions: int = 10
    learning_rate: float = 0.01
    
    # Retrieval parameters
    recall_timesteps: int = 100
    confidence_threshold: float = 0.5


class NeuromorphicMemory:
    """
    High-level neuromorphic memory system.
    
    Provides a simple API for storing and recalling memories
    using the SNN-based architecture.
    
    Key features:
    - Content-addressable recall (not similarity search)
    - Pattern completion from partial queries
    - Continual learning without catastrophic forgetting
    - Energy-efficient (event-driven computation)
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None,
                 embedding_fn: Optional[callable] = None):
        """
        Args:
            config: Memory system configuration
            embedding_fn: Function to convert text → embedding vector
                         If None, uses random embeddings (for testing)
        """
        self.config = config or MemoryConfig()
        self.embedding_fn = embedding_fn or self._random_embedding
        
        # Initialize components
        self._init_components()
        
        # Memory storage
        self.memories: Dict[int, MemoryEntry] = {}
        self.next_memory_id = 0
        
        # Statistics
        self.stats = {
            'stores': 0,
            'recalls': 0,
            'successful_recalls': 0,
            'total_store_time_ms': 0,
            'total_recall_time_ms': 0,
        }
    
    def _init_components(self):
        """Initialize SNN components."""
        cfg = self.config
        
        # Import here to avoid circular imports
        from network.reservoir import SparseReservoir, ReservoirParams
        from encoding.latency_coding import LatencyEncoder, LatencyParams
        from learning.stdp import STDPLearning, HebbianLearning
        
        # Reservoir
        reservoir_params = ReservoirParams(
            n_neurons=cfg.n_reservoir,
            n_input=512,  # Embedding dim (will be set properly on first store)
            connectivity=cfg.connectivity,
            delay_range=cfg.delay_range
        )
        self.reservoir = SparseReservoir(reservoir_params)
        
        # Encoder
        encoder_params = LatencyParams(t_window=cfg.encoding_window)
        self.encoder = LatencyEncoder(n_features=512, params=encoder_params)
        
        # WTA
        wta_params = WTAParams(
            n_memories=cfg.max_memories,
            n_neurons_per_memory=cfg.neurons_per_memory,
            min_confidence=cfg.confidence_threshold
        )
        self.wta = WinnerTakeAll(cfg.n_reservoir, wta_params)
        
        # Recall system
        recall_params = RecallParams(max_timesteps=cfg.recall_timesteps)
        self.recall_system = PatternRecall(
            self.reservoir, self.encoder, self.wta, recall_params
        )
        
        # Learning (for imprinting)
        self.stdp = STDPLearning(
            n_pre=cfg.n_reservoir,
            n_post=cfg.n_reservoir,
            connectivity=cfg.connectivity
        )
        self.hebbian = HebbianLearning(
            n_neurons=cfg.n_reservoir,
            learning_rate=cfg.learning_rate
        )
        
        self._initialized = True
    
    def _random_embedding(self, text: str) -> np.ndarray:
        """Default embedding function (random, for testing)."""
        # Use hash for reproducibility
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(512)
    
    def store(self, content: str, 
              embedding: Optional[np.ndarray] = None,
              memory_id: Optional[int] = None,
              metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store a memory.
        
        Args:
            content: Text content to store
            embedding: Pre-computed embedding (optional)
            memory_id: Specific ID to use (optional, auto-assigns if None)
            metadata: Additional metadata
            
        Returns:
            memory_id: Assigned memory ID
        """
        start_time = time.time()
        
        # Get embedding
        if embedding is None:
            embedding = self.embedding_fn(content)
        
        # Assign memory ID
        if memory_id is None:
            memory_id = self.next_memory_id
            self.next_memory_id += 1
        else:
            self.next_memory_id = max(self.next_memory_id, memory_id + 1)
        
        # Create memory entry
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
            created_at=time.time(),
            metadata=metadata or {}
        )
        self.memories[memory_id] = entry
        
        # Imprint into reservoir (simplified version)
        # Full version would run STDP over multiple presentations
        self._imprint(memory_id, embedding)
        
        # Store in recall system
        self.recall_system.store_memory(memory_id, {
            'content': content,
            'embedding': embedding,
            'metadata': metadata or {}
        })
        
        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['stores'] += 1
        self.stats['total_store_time_ms'] += elapsed_ms
        
        return memory_id
    
    def _imprint(self, memory_id: int, embedding: np.ndarray):
        """
        Imprint a memory into the reservoir via STDP.
        
        This is the "offline learning" phase where we present
        the pattern repeatedly to form stable attractors.
        """
        cfg = self.config
        
        # Encode to spike train
        spike_train = self.encoder.to_spike_train(embedding)
        
        # Track which neurons are most active for this memory
        activity_counts = np.zeros(cfg.n_reservoir)
        
        # Present pattern multiple times
        for rep in range(cfg.imprint_repetitions):
            self.reservoir.reset()
            
            for t in range(len(spike_train)):
                input_spikes = spike_train[t]
                reservoir_spikes, _ = self.reservoir.step(input_spikes)
                
                # Update activity counts
                activity_counts += reservoir_spikes.astype(float)
                
                # Apply Hebbian learning
                self.hebbian.step(reservoir_spikes)
        
        # Assign most active neurons to this memory's assembly
        top_neurons = np.argsort(activity_counts)[-cfg.neurons_per_memory:]
        self.wta.assign_assembly(memory_id, top_neurons)
    
    def recall(self, query: Union[str, np.ndarray],
               top_k: int = 1) -> Union[RecallResult, List[RecallResult]]:
        """
        Recall memory(ies) matching a query.
        
        Args:
            query: Query text or embedding
            top_k: Number of results to return
            
        Returns:
            RecallResult(s) with matched memory/memories
        """
        start_time = time.time()
        
        # Get query embedding
        if isinstance(query, str):
            query_embedding = self.embedding_fn(query)
        else:
            query_embedding = query
        
        # Run recall
        result = self.recall_system.recall(query_embedding)
        
        # Enhance result with content
        if result.memory_id in self.memories:
            entry = self.memories[result.memory_id]
            entry.access_count += 1
            entry.last_accessed = time.time()
            result.metadata['content'] = entry.content
        
        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['recalls'] += 1
        self.stats['total_recall_time_ms'] += elapsed_ms
        if result.memory_id >= 0:
            self.stats['successful_recalls'] += 1
        
        if top_k == 1:
            return result
        else:
            # Return top-k as list
            results = [result]
            for mem_id, score in result.candidates[1:top_k]:
                if mem_id in self.memories:
                    entry = self.memories[mem_id]
                    r = RecallResult(
                        memory_id=mem_id,
                        confidence=score,
                        candidates=result.candidates,
                        pattern_completion=0,  # Only computed for winner
                        retrieval_time_ms=0,
                        spike_count=0,
                        metadata={'content': entry.content}
                    )
                    results.append(r)
            return results
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories (compatibility API for Clawdbot).
        
        Returns list of dicts with 'content', 'score', 'memory_id'.
        """
        results = self.recall(query, top_k=limit)
        
        if not isinstance(results, list):
            results = [results]
        
        return [
            {
                'content': r.metadata.get('content', ''),
                'score': r.confidence,
                'memory_id': r.memory_id,
                'pattern_completion': r.pattern_completion
            }
            for r in results
            if r.memory_id >= 0
        ]
    
    def get(self, memory_id: int) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        return self.memories.get(memory_id)
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            # Note: assembly remains in WTA (could implement pruning)
            return True
        return False
    
    def list_memories(self, limit: int = 100) -> List[MemoryEntry]:
        """List all stored memories."""
        entries = list(self.memories.values())
        return sorted(entries, key=lambda e: e.created_at, reverse=True)[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = self.stats.copy()
        stats['memory_count'] = len(self.memories)
        stats['avg_store_time_ms'] = (
            stats['total_store_time_ms'] / stats['stores']
            if stats['stores'] > 0 else 0
        )
        stats['avg_recall_time_ms'] = (
            stats['total_recall_time_ms'] / stats['recalls']
            if stats['recalls'] > 0 else 0
        )
        stats['recall_success_rate'] = (
            stats['successful_recalls'] / stats['recalls']
            if stats['recalls'] > 0 else 0
        )
        return stats
    
    def save(self, path: str):
        """Save memory system to disk."""
        import pickle
        
        state = {
            'config': self.config,
            'memories': self.memories,
            'next_memory_id': self.next_memory_id,
            'stats': self.stats,
            # Would also save reservoir weights, WTA assemblies, etc.
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str, embedding_fn: Optional[callable] = None):
        """Load memory system from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        instance = cls(config=state['config'], embedding_fn=embedding_fn)
        instance.memories = state['memories']
        instance.next_memory_id = state['next_memory_id']
        instance.stats = state['stats']
        
        return instance


if __name__ == "__main__":
    print("Testing NeuromorphicMemory API...")
    print("(Full test requires all components initialized)")
    
    # Test MemoryEntry
    entry = MemoryEntry(
        memory_id=0,
        content="Test memory content",
        embedding=np.random.randn(512),
        created_at=time.time()
    )
    print(f"\nMemoryEntry: ID={entry.memory_id}, content='{entry.content[:20]}...'")
    
    # Test MemoryConfig
    config = MemoryConfig(
        n_reservoir=1000,
        max_memories=100
    )
    print(f"MemoryConfig: reservoir={config.n_reservoir}, max_memories={config.max_memories}")
    
    print("\n✅ Memory store module test complete!")
    print("\nFull integration test requires:")
    print("  - SparseReservoir from network/")
    print("  - LatencyEncoder from encoding/")
    print("  - STDPLearning from learning/")
