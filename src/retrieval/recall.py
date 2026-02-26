"""
Pattern Recall & Content-Addressable Memory

Implements the recall mechanism for neuromorphic memory:
1. Accept partial/noisy queries
2. Run through reservoir (attractor dynamics)
3. Pattern completion via polychronous groups
4. WTA selection of winning memory
5. Return recalled content with confidence

This is where the "magic" happens - recall via synaptic
activation rather than similarity search.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .wta import WinnerTakeAll, WTAParams
except ImportError:
    from wta import WinnerTakeAll, WTAParams


@dataclass
class RecallResult:
    """Result of a memory recall operation."""
    memory_id: int                    # Retrieved memory ID (-1 if no match)
    confidence: float                 # Recall confidence (0-1)
    candidates: List[Tuple[int, float]]  # Top-k candidates with scores
    pattern_completion: float         # How much of pattern was completed
    retrieval_time_ms: float          # Time taken for recall
    spike_count: int                  # Total spikes during recall
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class RecallParams:
    """Parameters for pattern recall."""
    max_timesteps: int = 100        # Max timesteps for recall
    early_stop_confidence: float = 0.9  # Stop if confidence exceeds this
    settling_window: int = 10       # Timesteps to check for settling
    settling_threshold: float = 0.05  # Change threshold for "settled"
    noise_tolerance: float = 0.3    # Fraction of query that can be noisy


class PatternRecall:
    """
    Content-addressable memory recall via SNN dynamics.
    
    The recall process:
    1. Encode query as spike train (partial/noisy OK)
    2. Inject into reservoir
    3. Let attractor dynamics evolve
    4. WTA selects winning assembly
    5. Decode and return memory
    
    Key insight: recall emerges from network structure,
    not from explicit search. The reservoir "knows" the
    answer through its synaptic weights.
    """
    
    def __init__(self, reservoir, encoder, wta: WinnerTakeAll,
                 params: Optional[RecallParams] = None):
        """
        Args:
            reservoir: SparseReservoir instance
            encoder: LatencyEncoder instance
            wta: WinnerTakeAll layer
            params: Recall parameters
        """
        self.reservoir = reservoir
        self.encoder = encoder
        self.wta = wta
        self.params = params or RecallParams()
        
        # Memory content storage
        # Maps memory_id -> stored content (embedding, metadata, etc.)
        self.memory_store: Dict[int, Dict[str, Any]] = {}
        
        # Statistics
        self.recall_count = 0
        self.total_spikes = 0
    
    def recall(self, query: np.ndarray, 
               return_embedding: bool = False) -> RecallResult:
        """
        Recall a memory given a (potentially partial) query.
        
        Args:
            query: Query vector (same dim as stored memories)
            return_embedding: Whether to include decoded embedding
            
        Returns:
            RecallResult with memory_id, confidence, etc.
        """
        import time
        start_time = time.time()
        
        p = self.params
        
        # Reset state for fresh recall
        self.reservoir.reset()
        self.wta.reset()
        
        # Encode query as spike train
        spike_train = self.encoder.to_spike_train(query)
        
        # Run reservoir with query input
        total_spikes = 0
        prev_confidence = 0.0
        settling_count = 0
        
        for t in range(min(p.max_timesteps, len(spike_train))):
            # Step reservoir
            input_spikes = spike_train[t] if t < len(spike_train) else None
            reservoir_spikes, _ = self.reservoir.step(input_spikes)
            
            total_spikes += reservoir_spikes.sum()
            
            # Update WTA
            self.wta.update(reservoir_spikes)
            
            # Check for early stopping
            _, confidence = self.wta.select_winner()
            
            if confidence >= p.early_stop_confidence:
                break
            
            # Check if settled
            if abs(confidence - prev_confidence) < p.settling_threshold:
                settling_count += 1
                if settling_count >= p.settling_window:
                    break
            else:
                settling_count = 0
            
            prev_confidence = confidence
        
        # Continue running (no input) to let attractor dynamics settle
        for t in range(p.settling_window):
            reservoir_spikes, _ = self.reservoir.step(None)
            total_spikes += reservoir_spikes.sum()
            self.wta.update(reservoir_spikes)
        
        # Get final winner
        memory_id, confidence = self.wta.select_winner()
        candidates = self.wta.get_top_k(5)
        
        # Compute pattern completion (how active was the winning assembly?)
        pattern_completion = 0.0
        if memory_id >= 0 and memory_id < len(self.wta.assemblies):
            assembly = self.wta.assemblies[memory_id]
            if len(assembly) > 0:
                # What fraction of assembly neurons fired?
                recent_spikes = np.array(self.wta.activity_buffer)
                if len(recent_spikes) > 0:
                    assembly_activity = recent_spikes[:, assembly].sum()
                    max_possible = len(assembly) * len(recent_spikes)
                    pattern_completion = assembly_activity / max_possible
        
        # Timing
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Build result
        result = RecallResult(
            memory_id=memory_id,
            confidence=confidence,
            candidates=candidates,
            pattern_completion=pattern_completion,
            retrieval_time_ms=elapsed_ms,
            spike_count=total_spikes
        )
        
        # Add stored metadata if available
        if memory_id in self.memory_store:
            result.metadata = self.memory_store[memory_id].copy()
            if return_embedding and 'embedding' in result.metadata:
                result.metadata['retrieved_embedding'] = result.metadata['embedding']
        
        # Update stats
        self.recall_count += 1
        self.total_spikes += total_spikes
        
        return result
    
    def recall_with_noise(self, query: np.ndarray, 
                          noise_level: float = 0.1) -> RecallResult:
        """
        Test recall with added noise.
        
        Useful for evaluating pattern completion capability.
        
        Args:
            query: Original query
            noise_level: Fraction of elements to corrupt (0-1)
        """
        # Add noise
        noisy_query = query.copy()
        n_corrupt = int(len(query) * noise_level)
        corrupt_indices = np.random.choice(len(query), n_corrupt, replace=False)
        noisy_query[corrupt_indices] = np.random.randn(n_corrupt)
        
        return self.recall(noisy_query)
    
    def recall_partial(self, query: np.ndarray, 
                       mask: np.ndarray) -> RecallResult:
        """
        Recall with partial query (some dimensions missing).
        
        Args:
            query: Query vector
            mask: Boolean array (True = known, False = missing)
        """
        # Zero out unknown dimensions
        partial_query = query.copy()
        partial_query[~mask] = 0.0
        
        return self.recall(partial_query)
    
    def store_memory(self, memory_id: int, 
                     content: Dict[str, Any]):
        """
        Store metadata/content for a memory.
        
        Called during imprinting to associate content with memory ID.
        
        Args:
            memory_id: Memory index
            content: Dict with 'embedding', 'text', 'metadata', etc.
        """
        self.memory_store[memory_id] = content
    
    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve stored content for a memory."""
        return self.memory_store.get(memory_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recall statistics."""
        return {
            'recall_count': self.recall_count,
            'total_spikes': self.total_spikes,
            'avg_spikes_per_recall': (
                self.total_spikes / self.recall_count 
                if self.recall_count > 0 else 0
            ),
            'memories_stored': len(self.memory_store),
            'assemblies_defined': len(self.wta.assemblies)
        }


class BatchRecall:
    """
    Batch recall for multiple queries.
    
    Useful for benchmarking and evaluation.
    """
    
    def __init__(self, pattern_recall: PatternRecall):
        self.recall = pattern_recall
    
    def recall_batch(self, queries: np.ndarray,
                     ground_truth: Optional[np.ndarray] = None
                     ) -> Tuple[List[RecallResult], Dict[str, float]]:
        """
        Recall multiple queries and compute metrics.
        
        Args:
            queries: Query matrix (n_queries x embedding_dim)
            ground_truth: True memory IDs for each query (optional)
            
        Returns:
            results: List of RecallResults
            metrics: Accuracy, avg confidence, avg time, etc.
        """
        results = []
        
        for query in queries:
            result = self.recall.recall(query)
            results.append(result)
        
        # Compute metrics
        metrics = {
            'avg_confidence': np.mean([r.confidence for r in results]),
            'avg_time_ms': np.mean([r.retrieval_time_ms for r in results]),
            'avg_spikes': np.mean([r.spike_count for r in results]),
            'avg_completion': np.mean([r.pattern_completion for r in results]),
        }
        
        if ground_truth is not None:
            correct = sum(
                1 for r, gt in zip(results, ground_truth) 
                if r.memory_id == gt
            )
            metrics['accuracy'] = correct / len(queries)
            metrics['top5_accuracy'] = sum(
                1 for r, gt in zip(results, ground_truth)
                if gt in [c[0] for c in r.candidates]
            ) / len(queries)
        
        return results, metrics


if __name__ == "__main__":
    print("Testing Pattern Recall...")
    print("(Full test requires reservoir + encoder integration)")
    
    # Test RecallResult dataclass
    result = RecallResult(
        memory_id=5,
        confidence=0.87,
        candidates=[(5, 1.0), (2, 0.3), (7, 0.1)],
        pattern_completion=0.65,
        retrieval_time_ms=12.5,
        spike_count=1500,
        metadata={'text': 'Test memory content'}
    )
    
    print(f"\nSample RecallResult:")
    print(f"  Memory ID: {result.memory_id}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Candidates: {result.candidates[:3]}")
    print(f"  Completion: {result.pattern_completion:.2f}")
    print(f"  Time: {result.retrieval_time_ms:.1f}ms")
    print(f"  Spikes: {result.spike_count}")
    
    print("\n✅ Recall module test complete!")
