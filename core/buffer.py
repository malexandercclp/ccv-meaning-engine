"""
CCV Recurrence Buffer

The recurrence buffer is the structured reservoir of survived experience.
Objects have valence-tagged interrogative axes enabling the same object
to hold different representations depending on emotional context.

Structure:
    - Each object has 5 axes: what, when, where, how, why
    - Each axis has 3 valence slots: positive (+), negative (-), neutral (0)
    - Each slot tracks: content, entropy, crystallization

Example:
    "father" might have:
        why+: "provides for family" (crystallization: 0.4)
        why-: "emotionally unavailable / critical" (crystallization: 0.85)
"""

import networkx as nx
import pickle
import os
import shutil
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Set, Tuple
from pathlib import Path

from .config import get_config
from .exceptions import (
    BufferError, BufferLoadError, BufferSaveError, BufferCorruptionError,
    ObjectNotFoundError, validate_object_name, validate_axis, validate_valence,
    validate_sentiment
)

logger = logging.getLogger("ccv.buffer")

# ============================================================================
# VALENCE CONSTANTS AND UTILITIES
# ============================================================================

VALENCE_POSITIVE = '+'
VALENCE_NEUTRAL = '0'
VALENCE_NEGATIVE = '-'
VALENCES = [VALENCE_POSITIVE, VALENCE_NEUTRAL, VALENCE_NEGATIVE]
AXES = ['what', 'when', 'where', 'how', 'why']


def sentiment_to_valence(sentiment: float) -> str:
    """
    Convert sentiment score to valence category.
    
    Args:
        sentiment: Value from -1.0 to 1.0
        
    Returns:
        Valence character: '+', '-', or '0'
    """
    config = get_config()
    sentiment = validate_sentiment(sentiment)
    
    if sentiment > config.valence.positive_threshold:
        return VALENCE_POSITIVE
    elif sentiment < config.valence.negative_threshold:
        return VALENCE_NEGATIVE
    else:
        return VALENCE_NEUTRAL


def get_valence_tagged_axis(axis: str, valence: str) -> str:
    """Create a valence-tagged axis key like 'why+' or 'what-'."""
    return f"{axis}{valence}"


def parse_valence_tagged_axis(tagged_axis: str) -> Tuple[str, str]:
    """
    Parse 'why+' into ('why', '+').
    
    Args:
        tagged_axis: Valence-tagged axis string
        
    Returns:
        Tuple of (base_axis, valence)
    """
    if tagged_axis and tagged_axis[-1] in VALENCES:
        return tagged_axis[:-1], tagged_axis[-1]
    return tagged_axis, VALENCE_NEUTRAL


# ============================================================================
# RECURRENCE BUFFER
# ============================================================================

class RecurrenceBuffer:
    """
    The recurrence buffer: structured reservoir of survived experience.
    
    Objects have valence-tagged interrogative axes:
    - what+, what-, what0
    - when+, when-, when0
    - where+, where-, where0
    - how+, how-, how0
    - why+, why-, why0
    
    This allows the same object to hold different representations
    depending on emotional context, without contradiction.
    """
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the recurrence buffer.
        
        Args:
            file_path: Path to buffer persistence file. If None, uses config default.
        """
        self.config = get_config()
        
        if file_path:
            self.file_path = Path(file_path)
        else:
            self.file_path = self.config.paths.buffer_path
        
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._modified = False
        self._load_buffer()
        
        logger.info(f"Buffer initialized: {len(self.graph.nodes())} objects, {len(self.graph.edges())} edges")
    
    # ========================================================================
    # NORMALIZATION
    # ========================================================================
    
    def _normalize(self, obj_name: str) -> str:
        """Case-insensitive normalization with validation."""
        return validate_object_name(obj_name)
    
    def _init_valence_tagged_structure(self) -> Dict[str, Any]:
        """Create empty structure for all valence-tagged axes."""
        config = self.config
        structure = {
            'interrogatives': {},
            'axis_entropy': {},
            'crystallization': {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'access_count': 0,
        }
        
        for axis in AXES:
            for valence in VALENCES:
                tagged = get_valence_tagged_axis(axis, valence)
                structure['interrogatives'][tagged] = ""
                structure['axis_entropy'][tagged] = config.entropy.initial
                structure['crystallization'][tagged] = config.crystallization.initial
        
        return structure
    
    # ========================================================================
    # OBJECT MANAGEMENT
    # ========================================================================
    
    def add_object(
        self, 
        obj_name: str, 
        interrogatives: Dict[str, str], 
        sentiment: float = 0.0
    ) -> str:
        """
        Add or update an object in the buffer.
        
        Interrogatives are placed in the appropriate valence-tagged slot
        based on the sentiment of the experience.
        
        Content is deduplicated - repeated identical inputs reinforce 
        crystallization but don't create duplicate entries.
        
        Args:
            obj_name: Name of the object/concept
            interrogatives: Dict mapping axis names to content
            sentiment: Emotional valence of this experience (-1 to 1)
            
        Returns:
            Normalized object name
        """
        obj_name = self._normalize(obj_name)
        sentiment = validate_sentiment(sentiment)
        valence = sentiment_to_valence(sentiment)
        
        if obj_name not in self.graph:
            structure = self._init_valence_tagged_structure()
            self.graph.add_node(obj_name, sentiment=sentiment, **structure)
            logger.debug(f"Created new object: {obj_name}")
        
        node = self.graph.nodes[obj_name]
        node['updated_at'] = datetime.now().isoformat()
        node['access_count'] = node.get('access_count', 0) + 1
        
        # Place each interrogative in its valence-tagged slot
        for axis, content in interrogatives.items():
            if not content:
                continue
            
            # Validate axis
            try:
                base_axis = axis.rstrip('+-0')
                if base_axis not in AXES:
                    logger.warning(f"Skipping invalid axis: {axis}")
                    continue
            except Exception:
                continue
            
            tagged_axis = get_valence_tagged_axis(base_axis, valence)
            existing = node['interrogatives'].get(tagged_axis, "")
            
            if existing:
                # Deduplicate - check if content already exists
                existing_entries = {e.strip().lower() for e in existing.split(" / ")}
                if content.strip().lower() not in existing_entries:
                    # New unique content - accumulate
                    node['interrogatives'][tagged_axis] = existing + " / " + content.strip()
                    logger.debug(f"Accumulated content on {obj_name}[{tagged_axis}]")
            else:
                node['interrogatives'][tagged_axis] = content.strip()
                logger.debug(f"Set content on {obj_name}[{tagged_axis}]")
        
        # Update overall sentiment (exponential moving average)
        alpha = 0.3  # Weight for new sentiment
        node['sentiment'] = alpha * sentiment + (1 - alpha) * node.get('sentiment', 0.0)
        
        self._modified = True
        return obj_name
    
    def get_object(self, obj_name: str) -> Optional[Dict[str, Any]]:
        """
        Get object from buffer.
        
        Args:
            obj_name: Object name to retrieve
            
        Returns:
            Object data dict or None if not found
        """
        try:
            obj_name = self._normalize(obj_name)
        except Exception:
            return None
        
        return self.graph.nodes.get(obj_name)
    
    def has_object(self, obj_name: str) -> bool:
        """Check if object exists in buffer."""
        try:
            obj_name = self._normalize(obj_name)
            return obj_name in self.graph
        except Exception:
            return False
    
    def remove_object(self, obj_name: str) -> bool:
        """
        Remove an object from the buffer.
        
        Args:
            obj_name: Object to remove
            
        Returns:
            True if removed, False if not found
        """
        try:
            obj_name = self._normalize(obj_name)
            if obj_name in self.graph:
                self.graph.remove_node(obj_name)
                self._modified = True
                logger.info(f"Removed object: {obj_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing object {obj_name}: {e}")
            return False
    
    def list_objects(self) -> List[str]:
        """Get list of all object names in buffer."""
        return sorted(list(self.graph.nodes()))
    
    # ========================================================================
    # AXIS CONTENT ACCESS
    # ========================================================================
    
    def get_axis_content(
        self, 
        obj_name: str, 
        axis: str, 
        valence: Optional[str] = None
    ) -> Optional[Dict[str, str] | str]:
        """
        Get content for an axis, optionally filtered by valence.
        
        Args:
            obj_name: Object name
            axis: Base axis name (what/when/where/how/why)
            valence: Optional valence filter (+/-/0)
            
        Returns:
            If valence specified: content string or empty string
            If no valence: dict mapping valences to content
        """
        node = self.get_object(obj_name)
        if not node:
            return None
        
        if valence:
            tagged = get_valence_tagged_axis(axis, valence)
            return node['interrogatives'].get(tagged, "")
        else:
            result = {}
            for v in VALENCES:
                tagged = get_valence_tagged_axis(axis, v)
                content = node['interrogatives'].get(tagged, "")
                if content:
                    result[v] = content
            return result
    
    # ========================================================================
    # ENTROPY MANAGEMENT
    # ========================================================================
    
    def get_axis_entropy(self, obj_name: str, axis: str, valence: str) -> float:
        """Get entropy for a specific valence-tagged axis."""
        node = self.get_object(obj_name)
        if not node:
            return 0.0
        tagged = get_valence_tagged_axis(axis, valence)
        return node['axis_entropy'].get(tagged, self.config.entropy.initial)
    
    def set_axis_entropy(self, obj_name: str, axis: str, valence: str, entropy: float):
        """Set entropy for a specific valence-tagged axis."""
        node = self.get_object(obj_name)
        if node:
            tagged = get_valence_tagged_axis(axis, valence)
            node['axis_entropy'][tagged] = max(0.0, min(1.0, entropy))
            self._modified = True
    
    def increase_entropy(self, obj_name: str, axis: str, valence: str, amount: float):
        """Increase entropy on an axis (from contradiction)."""
        current = self.get_axis_entropy(obj_name, axis, valence)
        new_val = min(1.0, current + amount * self.config.entropy.multiplier)
        self.set_axis_entropy(obj_name, axis, valence, new_val)
        logger.debug(f"Entropy increased on {obj_name}[{axis}{valence}]: {current:.3f} → {new_val:.3f}")
    
    def decay_entropy(self, obj_name: str, axis: str, valence: str):
        """Decay entropy on an axis (from successful traversal)."""
        current = self.get_axis_entropy(obj_name, axis, valence)
        new_val = current * self.config.entropy.decay
        self.set_axis_entropy(obj_name, axis, valence, new_val)
    
    def is_axis_blocked(self, obj_name: str, axis: str, valence: str) -> bool:
        """Check if an axis is blocked due to high entropy."""
        entropy = self.get_axis_entropy(obj_name, axis, valence)
        return entropy >= self.config.entropy.threshold
    
    # ========================================================================
    # CRYSTALLIZATION MANAGEMENT
    # ========================================================================
    
    def get_crystallization(self, obj_name: str, axis: str, valence: str) -> float:
        """Get crystallization for a specific valence-tagged axis."""
        node = self.get_object(obj_name)
        if not node:
            return self.config.crystallization.initial
        tagged = get_valence_tagged_axis(axis, valence)
        return node['crystallization'].get(tagged, self.config.crystallization.initial)
    
    def set_crystallization(self, obj_name: str, axis: str, valence: str, cryst: float):
        """Set crystallization for a specific valence-tagged axis."""
        node = self.get_object(obj_name)
        if node:
            tagged = get_valence_tagged_axis(axis, valence)
            clamped = max(
                self.config.crystallization.minimum,
                min(self.config.crystallization.maximum, cryst)
            )
            node['crystallization'][tagged] = clamped
            self._modified = True
    
    def increase_crystallization(
        self, 
        obj_name: str, 
        axis: str, 
        valence: str, 
        amount: Optional[float] = None
    ):
        """Increase crystallization through reinforcement."""
        if amount is None:
            amount = self.config.crystallization.boost
        current = self.get_crystallization(obj_name, axis, valence)
        self.set_crystallization(obj_name, axis, valence, current + amount)
        logger.debug(f"Crystallization increased on {obj_name}[{axis}{valence}]")
    
    def decrease_crystallization(
        self,
        obj_name: str,
        axis: str,
        valence: str,
        amount: Optional[float] = None
    ):
        """Decrease crystallization (from competition penalty)."""
        if amount is None:
            amount = self.config.crystallization.boost * self.config.valence.competition_penalty
        current = self.get_crystallization(obj_name, axis, valence)
        self.set_crystallization(obj_name, axis, valence, current - amount)
    
    # ========================================================================
    # EDGE MANAGEMENT
    # ========================================================================
    
    def add_dependency(
        self, 
        from_obj: str, 
        to_obj: str, 
        axis: Optional[str] = None, 
        valence: Optional[str] = None, 
        weight: float = 1.0
    ):
        """Add a directional dependency edge between objects."""
        from_obj = self._normalize(from_obj)
        to_obj = self._normalize(to_obj)
        
        if from_obj not in self.graph or to_obj not in self.graph:
            logger.warning(f"Cannot add edge: one or both objects not in buffer")
            return
        
        if from_obj == to_obj:
            return  # No self-loops
        
        if axis and valence:
            key = get_valence_tagged_axis(axis, valence)
        elif axis:
            key = axis
        else:
            key = "dependency"
        
        self.graph.add_edge(
            from_obj, to_obj, 
            key=key, 
            weight=weight,
            axis=axis, 
            valence=valence,
            created_at=datetime.now().isoformat()
        )
        self._modified = True
    
    def add_cooccurrence_edges(
        self, 
        objects: List[str], 
        axis: Optional[str] = None, 
        valence: Optional[str] = None
    ):
        """
        Create edges from co-occurrence in same input.
        
        Objects that appear together in the same experience become linked.
        """
        if len(objects) < 2:
            return
        
        normalized = [self._normalize(obj) for obj in objects]
        # Filter to only existing objects
        normalized = [obj for obj in normalized if obj in self.graph]
        
        if len(normalized) < 2:
            return
        
        primary = normalized[0]
        
        for other in normalized[1:]:
            if primary == other:
                continue
            
            # Build edge key
            if axis and valence:
                key = get_valence_tagged_axis(axis, valence)
            elif axis:
                key = axis
            elif valence:
                key = f"cooccur_{valence}"
            else:
                key = "cooccur"
            
            if self.graph.has_edge(primary, other, key=key):
                # Edge exists - increase weight
                edge_data = self.graph[primary][other][key]
                edge_data['weight'] = edge_data.get('weight', 1.0) + 0.5
            else:
                # Create new edge
                self.graph.add_edge(
                    primary, other, 
                    key=key, 
                    weight=1.0,
                    axis=axis, 
                    valence=valence,
                    created_at=datetime.now().isoformat()
                )
        
        self._modified = True
    
    def get_connected_objects(self, obj_name: str) -> Set[str]:
        """Get all objects connected to this one (predecessors and successors)."""
        obj_name = self._normalize(obj_name)
        if obj_name not in self.graph:
            return {obj_name}
        
        connected = {obj_name}
        connected.update(self.graph.predecessors(obj_name))
        connected.update(self.graph.successors(obj_name))
        return connected
    
    def get_descendants(self, obj_name: str, max_depth: int = 3) -> Set[str]:
        """Get all descendants up to max_depth."""
        obj_name = self._normalize(obj_name)
        if obj_name not in self.graph:
            return set()
        
        descendants = set()
        current_level = {obj_name}
        
        for _ in range(max_depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.successors(node))
            descendants.update(next_level)
            current_level = next_level - descendants
            if not current_level:
                break
        
        return descendants
    
    # ========================================================================
    # SUMMARY AND INSPECTION
    # ========================================================================
    
    def get_node_summary(self, obj_name: str) -> Optional[Dict[str, Any]]:
        """Get a human-readable summary of an object's state."""
        node = self.get_object(obj_name)
        if not node:
            return None
        
        obj_name = self._normalize(obj_name)
        summary = {
            'object': obj_name,
            'sentiment': round(node.get('sentiment', 0.0), 3),
            'created_at': node.get('created_at', 'unknown'),
            'updated_at': node.get('updated_at', 'unknown'),
            'access_count': node.get('access_count', 0),
            'axes': {},
            'connections': {
                'incoming': list(self.graph.predecessors(obj_name)),
                'outgoing': list(self.graph.successors(obj_name)),
            }
        }
        
        interrogatives = node.get('interrogatives', {})
        crystallization = node.get('crystallization', {})
        axis_entropy = node.get('axis_entropy', {})
        
        for axis in AXES:
            axis_data = {}
            for valence in VALENCES:
                tagged = get_valence_tagged_axis(axis, valence)
                content = interrogatives.get(tagged, "")
                entropy = axis_entropy.get(tagged, self.config.entropy.initial)
                cryst = crystallization.get(tagged, self.config.crystallization.initial)
                
                if content or entropy > 0.15 or cryst != 0.5:
                    blocked = entropy >= self.config.entropy.threshold
                    axis_data[valence] = {
                        'content': content,
                        'entropy': round(entropy, 3),
                        'crystallization': round(cryst, 3),
                        'blocked': blocked,
                    }
            
            if axis_data:
                summary['axes'][axis] = axis_data
        
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer-wide statistics."""
        total_content = 0
        total_blocked = 0
        total_slots = 0
        entropy_sum = 0.0
        cryst_sum = 0.0
        
        for node_name, node_data in self.graph.nodes(data=True):
            interrogatives = node_data.get('interrogatives', {})
            axis_entropy = node_data.get('axis_entropy', {})
            crystallization = node_data.get('crystallization', {})
            
            for axis in AXES:
                for valence in VALENCES:
                    tagged = get_valence_tagged_axis(axis, valence)
                    total_slots += 1
                    
                    if interrogatives.get(tagged):
                        total_content += 1
                    
                    entropy = axis_entropy.get(tagged, self.config.entropy.initial)
                    entropy_sum += entropy
                    if entropy >= self.config.entropy.threshold:
                        total_blocked += 1
                    
                    cryst_sum += crystallization.get(tagged, self.config.crystallization.initial)
        
        return {
            'num_objects': len(self.graph.nodes()),
            'num_edges': len(self.graph.edges()),
            'total_slots': total_slots,
            'filled_slots': total_content,
            'blocked_slots': total_blocked,
            'avg_entropy': entropy_sum / max(1, total_slots),
            'avg_crystallization': cryst_sum / max(1, total_slots),
            'fill_rate': total_content / max(1, total_slots),
            'block_rate': total_blocked / max(1, total_slots),
        }
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def _load_buffer(self):
        """Load buffer from disk."""
        if not self.file_path.exists():
            logger.info(f"No existing buffer at {self.file_path}, starting fresh")
            self.graph = nx.MultiDiGraph()
            return
        
        try:
            with open(self.file_path, 'rb') as f:
                loaded_graph = pickle.load(f)
            
            # Handle legacy DiGraph format
            if isinstance(loaded_graph, nx.DiGraph) and not isinstance(loaded_graph, nx.MultiDiGraph):
                logger.info("Converting legacy DiGraph to MultiDiGraph")
                self.graph = nx.MultiDiGraph()
                for node, attrs in loaded_graph.nodes(data=True):
                    self.graph.add_node(node, **attrs)
                for u, v, attrs in loaded_graph.edges(data=True):
                    self.graph.add_edge(u, v, key=attrs.get('axis'), **attrs)
                self._modified = True  # Save converted format
            else:
                self.graph = loaded_graph
            
            logger.info(f"Loaded buffer: {len(self.graph.nodes())} objects")
            
        except Exception as e:
            logger.error(f"Failed to load buffer: {e}")
            raise BufferLoadError(f"Failed to load buffer from {self.file_path}: {e}")
    
    def save_buffer(self, force: bool = False):
        """
        Persist buffer to disk.
        
        Args:
            force: Save even if no modifications detected
        """
        if not self._modified and not force:
            return
        
        try:
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write: write to temp file, then rename
            temp_path = self.file_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Rename temp to actual (atomic on most systems)
            temp_path.replace(self.file_path)
            
            self._modified = False
            logger.debug(f"Buffer saved to {self.file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save buffer: {e}")
            raise BufferSaveError(f"Failed to save buffer to {self.file_path}: {e}")
    
    def create_backup(self, name: Optional[str] = None) -> Path:
        """
        Create a backup of the current buffer.
        
        Args:
            name: Optional backup name. If None, uses timestamp.
            
        Returns:
            Path to backup file
        """
        backup_dir = self.config.paths.backup_path
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_path = backup_dir / f"buffer_backup_{name}.pkl"
        
        try:
            with open(backup_path, 'wb') as f:
                pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise BufferSaveError(f"Failed to create backup: {e}")
    
    def restore_backup(self, backup_path: str):
        """
        Restore buffer from a backup file.
        
        Args:
            backup_path: Path to backup file
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise BufferLoadError(f"Backup file not found: {backup_path}")
        
        try:
            with open(backup_path, 'rb') as f:
                self.graph = pickle.load(f)
            self._modified = True
            self.save_buffer()
            logger.info(f"Restored from backup: {backup_path}")
        except Exception as e:
            raise BufferLoadError(f"Failed to restore backup: {e}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backup_dir = self.config.paths.backup_path
        if not backup_dir.exists():
            return []
        
        backups = []
        for path in sorted(backup_dir.glob("buffer_backup_*.pkl")):
            stat = path.stat()
            backups.append({
                'path': str(path),
                'name': path.stem.replace('buffer_backup_', ''),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        
        return backups
    
    def clear(self, create_backup: bool = True):
        """
        Clear all data from the buffer.
        
        Args:
            create_backup: Whether to create a backup before clearing
        """
        if create_backup and len(self.graph.nodes()) > 0:
            self.create_backup("pre_clear")
        
        # Explicitly clear the existing graph first
        # This ensures all internal state is cleaned up
        self.graph.clear()
        
        # Now create a completely fresh MultiDiGraph
        self.graph = nx.MultiDiGraph()
        self._modified = True
        self.save_buffer(force=True)  # Force save to ensure pickle is written
        logger.info("Buffer cleared")
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Export buffer to dictionary format."""
        export = {
            'version': self.config.version,
            'exported_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'nodes': {},
            'edges': [],
        }
        
        for node, data in self.graph.nodes(data=True):
            export['nodes'][node] = {
                'sentiment': data.get('sentiment', 0),
                'interrogatives': data.get('interrogatives', {}),
                'crystallization': data.get('crystallization', {}),
                'axis_entropy': data.get('axis_entropy', {}),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'access_count': data.get('access_count', 0),
            }
        
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            export['edges'].append({
                'source': u,
                'target': v,
                'key': key,
                'weight': data.get('weight', 1.0),
                'valence': data.get('valence'),
                'axis': data.get('axis'),
                'created_at': data.get('created_at'),
            })
        
        return export
