"""
CCV (Curiosity, Clarity, Valence) Processor

Core processing engine for the CCV framework. Implements the computational model
of meaning-making based on pre-linguistic interrogative structures with
valence-tagged axes.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .config import get_config, CCVConfig
from .exceptions import ProcessingError, InvalidInputError, validate_input_text
from .buffer import (
    RecurrenceBuffer, AXES, VALENCES,
    VALENCE_POSITIVE, VALENCE_NEGATIVE, VALENCE_NEUTRAL,
    sentiment_to_valence, get_valence_tagged_axis, parse_valence_tagged_axis
)
from .parser import parse_input, ParsedObject, analyze_semantic_similarity, detect_contradiction, llm_detect_contradiction

logger = logging.getLogger("ccv.processor")


class SystemState(Enum):
    """System engagement states based on CCV dynamics."""
    ENGAGED = "engaged"
    TENTATIVE = "tentative"
    DISENGAGED = "disengaged"
    WITHDRAWN = "withdrawn"


@dataclass
class AxisUpdate:
    """Record of what happened to a specific axis."""
    axis: str
    valence: str
    action: str
    mismatch: float
    entropy_before: float
    entropy_after: float
    crystallization_before: float
    crystallization_after: float
    blocked: bool = False


@dataclass
class ObjectUpdate:
    """Record of what happened to an object during processing."""
    object_name: str
    is_new: bool
    valence: str
    sentiment: float
    axis_updates: List[AxisUpdate] = field(default_factory=list)
    mismatch: float = 0.0


@dataclass
class CompetitionEvent:
    """Record of a valence competition event."""
    object_name: str
    axis: str
    dominant_valence: str
    subordinate_valence: str
    dominant_crystallization: float
    subordinate_before: float
    subordinate_after: float


@dataclass
class CrossValenceFriction:
    """Record of cross-valence contradiction friction."""
    object_name: str
    axis: str
    new_valence: str
    opposing_valence: str
    new_content: str
    opposing_content: str
    contradiction_confidence: float
    entropy_added_to_new: float
    entropy_added_to_opposing: float


@dataclass
class TraversalResult:
    """Result of attempting to traverse an axis."""
    object_name: str
    tagged_axis: str
    success: bool
    entropy: float
    reason: str = ""


@dataclass
class ProcessingResult:
    """Complete result of processing an experience."""
    input_text: str
    timestamp: str
    clarity: float
    arousal: float
    valence_intensity: float
    curiosity: float
    system_state: SystemState
    object_updates: List[ObjectUpdate]
    competition_events: List[CompetitionEvent]
    friction_events: List[CrossValenceFriction]
    traversal_results: List[TraversalResult]
    affected_objects: List[str]
    new_objects: List[str]
    blocked_axes: List[str]
    spillover_opportunities: List[str]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'input_text': self.input_text,
            'timestamp': self.timestamp,
            'clarity': round(self.clarity, 3),
            'arousal': round(self.arousal, 3),
            'valence_intensity': round(self.valence_intensity, 3),
            'curiosity': round(self.curiosity, 3),
            'system_state': self.system_state.value,
            'affected_objects': self.affected_objects,
            'new_objects': self.new_objects,
            'blocked_traversals': self.blocked_axes,
            'spillover_effects': self.spillover_opportunities,
            'valence_competitions': [
                f"{e.object_name}[{e.axis}{e.dominant_valence}] dominates [{e.axis}{e.subordinate_valence}]"
                for e in self.competition_events
            ],
            'cross_valence_friction': [
                f"{e.object_name}[{e.axis}]: '{e.new_content}' vs '{e.opposing_content}' (conf: {e.contradiction_confidence:.2f})"
                for e in self.friction_events
            ],
            'error': self.error,
        }


class CCVProcessor:
    """
    CCV (Curiosity, Clarity, Valence) Processor
    
    Implements a computational model of meaning-making based on pre-linguistic
    interrogative structures (what/when/where/how/why) with valence-tagged axes.
    """
    
    def __init__(
        self, 
        buffer: Optional[RecurrenceBuffer] = None,
        config: Optional[CCVConfig] = None
    ):
        self.config = config or get_config()
        self.buffer = buffer or RecurrenceBuffer()
        self.history: List[ProcessingResult] = []
        logger.info("CCV Processor initialized")
    
    def calculate_clarity(self) -> float:
        """Calculate global clarity: overall navigability of the buffer."""
        if len(self.buffer.graph.nodes()) == 0:
            return 1.0
        
        total_axes = 0
        total_entropy = 0.0
        
        for node, attrs in self.buffer.graph.nodes(data=True):
            axis_entropy = attrs.get('axis_entropy', {})
            interrogatives = attrs.get('interrogatives', {})
            
            for axis in AXES:
                for valence in VALENCES:
                    tagged = get_valence_tagged_axis(axis, valence)
                    content = interrogatives.get(tagged, "")
                    if content:
                        total_axes += 1
                        total_entropy += axis_entropy.get(tagged, self.config.entropy.initial)
        
        if total_axes == 0:
            return 1.0
        
        average_entropy = total_entropy / total_axes
        return max(0.0, min(1.0, 1.0 - average_entropy))
    
    def calculate_arousal(self, total_mismatch: float, new_objects_count: int) -> float:
        """Calculate arousal: the initial attention signal."""
        novelty_component = 0.3 * new_objects_count
        mismatch_component = total_mismatch
        arousal = novelty_component + mismatch_component
        return max(0.0, min(1.0, arousal))
    
    def calculate_valence_intensity(self, arousal: float, clarity: float) -> float:
        """Calculate valence intensity: how intensely significant something feels."""
        if clarity > 0.7:
            intensity = arousal * clarity
        elif clarity > 0.4:
            intensity = arousal * (1.0 + (0.7 - clarity))
        else:
            intensity = arousal * (1.5 + (0.4 - clarity))
        return max(0.0, min(1.0, intensity))
    
    def calculate_curiosity(self, arousal: float, total_mismatch: float, clarity: float) -> float:
        """Calculate curiosity: conditional engagement with novelty."""
        something_happening = arousal > self.config.curiosity.arousal_threshold
        something_to_explore = total_mismatch > self.config.curiosity.mismatch_threshold
        can_tolerate = clarity >= self.config.curiosity.clarity_floor
        
        if not (something_happening and something_to_explore and can_tolerate):
            return 0.0
        
        floor = self.config.curiosity.clarity_floor
        capacity = (clarity - floor) / (1.0 - floor)
        curiosity = total_mismatch * capacity
        return max(0.0, min(1.0, curiosity))
    
    def compute_valence_aware_mismatch(
        self,
        existing: Optional[Dict[str, Any]],
        interrogatives: Dict[str, str],
        sentiment: float
    ) -> Tuple[float, Dict[str, float]]:
        """Compute mismatch considering valence-tagged slots."""
        if existing is None:
            return 0.1, {}
        
        valence = sentiment_to_valence(sentiment)
        axis_mismatches = {}
        total_diff = 0.0
        axis_count = 0
        existing_inter = existing.get('interrogatives', {})
        
        for axis in AXES:
            input_content = interrogatives.get(axis, "")
            if not input_content:
                continue
            
            axis_count += 1
            tagged_axis = get_valence_tagged_axis(axis, valence)
            existing_content = existing_inter.get(tagged_axis, "")
            
            if existing_content:
                input_lower = input_content.lower().strip()
                existing_lower = existing_content.lower().strip()
                
                if input_lower == existing_lower:
                    axis_diff = 0.0
                elif input_lower in existing_lower or existing_lower in input_lower:
                    axis_diff = 0.0
                else:
                    similarity = analyze_semantic_similarity(input_content, existing_content)
                    is_contradiction, conf = detect_contradiction(input_content, existing_content)
                    
                    if is_contradiction and conf > 0.5:
                        axis_diff = 0.3 * conf
                    elif similarity > 0.5:
                        axis_diff = 0.02
                    else:
                        axis_diff = 0.05
            else:
                axis_diff = 0.05
            
            axis_mismatches[tagged_axis] = axis_diff
            total_diff += axis_diff
        
        if axis_count > 0:
            total_diff /= axis_count
        
        existing_sentiment = existing.get('sentiment', 0.0)
        sent_diff = abs(sentiment - existing_sentiment) * 0.2
        total_mismatch = total_diff + sent_diff
        
        return total_mismatch, axis_mismatches
    
    def apply_valence_competition(
        self,
        obj_name: str,
        axis: str,
        reinforced_valence: str
    ) -> Optional[CompetitionEvent]:
        """Apply valence competition dynamics."""
        if reinforced_valence == VALENCE_NEUTRAL:
            return None
        
        competing_valence = (
            VALENCE_NEGATIVE if reinforced_valence == VALENCE_POSITIVE 
            else VALENCE_POSITIVE
        )
        
        current_cryst = self.buffer.get_crystallization(obj_name, axis, reinforced_valence)
        competing_cryst = self.buffer.get_crystallization(obj_name, axis, competing_valence)
        gap = current_cryst - competing_cryst
        
        if gap > self.config.valence.dominance_threshold:
            penalty = self.config.crystallization.boost * self.config.valence.competition_penalty
            new_competing = max(self.config.crystallization.minimum, competing_cryst - penalty)
            self.buffer.set_crystallization(obj_name, axis, competing_valence, new_competing)
            
            return CompetitionEvent(
                object_name=obj_name,
                axis=axis,
                dominant_valence=reinforced_valence,
                subordinate_valence=competing_valence,
                dominant_crystallization=current_cryst,
                subordinate_before=competing_cryst,
                subordinate_after=new_competing
            )
        
        return None
    
    def check_cross_valence_friction(
        self, 
        obj_name: str, 
        axis: str, 
        new_valence: str, 
        new_content: str
    ) -> Optional[CrossValenceFriction]:
        """
        Check for contradiction with opposing valence slots and apply friction.
        
        When new positive content contradicts existing negative content (or vice versa),
        entropy is added to BOTH slots, representing cognitive dissonance.
        
        Args:
            obj_name: Object being updated
            axis: Base axis (what/when/where/how/why)
            new_valence: Valence of incoming content
            new_content: The new content being added
            
        Returns:
            CrossValenceFriction event if friction detected, None otherwise
        """
        # Determine opposing valence(s)
        if new_valence == VALENCE_POSITIVE:
            opposing_valences = [VALENCE_NEGATIVE]
        elif new_valence == VALENCE_NEGATIVE:
            opposing_valences = [VALENCE_POSITIVE]
        else:
            # Neutral - check both positive and negative
            opposing_valences = [VALENCE_POSITIVE, VALENCE_NEGATIVE]
        
        for opposing_valence in opposing_valences:
            opposing_content = self.buffer.get_axis_content(obj_name, axis, opposing_valence)
            
            if not opposing_content:
                continue
            
            # Check for semantic contradiction using LLM
            is_contradiction, confidence = llm_detect_contradiction(opposing_content, new_content)
            
            if is_contradiction and confidence > 0.4:
                # Calculate friction amount based on confidence and opposing crystallization
                opposing_cryst = self.buffer.get_crystallization(obj_name, axis, opposing_valence)
                
                # More crystallized opposing beliefs create more friction
                friction_amount = confidence * 0.15 * (1 + opposing_cryst)
                
                # Apply entropy to BOTH slots
                # New slot: the incoming content is destabilized by existing contrary evidence
                self.buffer.increase_entropy(obj_name, axis, new_valence, friction_amount)
                
                # Opposing slot: existing content is challenged by new contrary evidence
                self.buffer.increase_entropy(obj_name, axis, opposing_valence, friction_amount * 0.5)
                
                logger.info(
                    f"Cross-valence friction detected: {obj_name}[{axis}] "
                    f"'{new_content}' vs '{opposing_content}' (conf: {confidence:.2f})"
                )
                
                return CrossValenceFriction(
                    object_name=obj_name,
                    axis=axis,
                    new_valence=new_valence,
                    opposing_valence=opposing_valence,
                    new_content=new_content[:50],
                    opposing_content=opposing_content[:50],
                    contradiction_confidence=confidence,
                    entropy_added_to_new=friction_amount,
                    entropy_added_to_opposing=friction_amount * 0.5
                )
        
        return None
    
    def determine_system_state(self, curiosity: float, clarity: float) -> SystemState:
        """Determine system state from curiosity and clarity."""
        if curiosity == 0.0:
            if clarity < self.config.curiosity.clarity_floor:
                return SystemState.WITHDRAWN
            return SystemState.DISENGAGED
        elif curiosity > 0.5:
            return SystemState.ENGAGED
        else:
            return SystemState.TENTATIVE
    
    def process_input(self, text: str) -> ProcessingResult:
        """
        Process an experience against the recurrence buffer.
        
        This is the main entry point for the CCV system.
        """
        timestamp = datetime.now().isoformat()
        
        # Validate input
        try:
            text = validate_input_text(text)
        except InvalidInputError as e:
            return ProcessingResult(
                input_text=text if text else "",
                timestamp=timestamp,
                clarity=self.calculate_clarity(),
                arousal=0.0,
                valence_intensity=0.0,
                curiosity=0.0,
                system_state=SystemState.DISENGAGED,
                object_updates=[],
                competition_events=[],
                friction_events=[],
                traversal_results=[],
                affected_objects=[],
                new_objects=[],
                blocked_axes=[],
                spillover_opportunities=[],
                error=str(e)
            )
        
        # Parse input
        try:
            parsed_objects = parse_input(text)
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return ProcessingResult(
                input_text=text,
                timestamp=timestamp,
                clarity=self.calculate_clarity(),
                arousal=0.0,
                valence_intensity=0.0,
                curiosity=0.0,
                system_state=SystemState.DISENGAGED,
                object_updates=[],
                competition_events=[],
                friction_events=[],
                traversal_results=[],
                affected_objects=[],
                new_objects=[],
                blocked_axes=[],
                spillover_opportunities=[],
                error=f"Parse error: {e}"
            )
        
        if not parsed_objects:
            return ProcessingResult(
                input_text=text,
                timestamp=timestamp,
                clarity=self.calculate_clarity(),
                arousal=0.0,
                valence_intensity=0.0,
                curiosity=0.0,
                system_state=SystemState.DISENGAGED,
                object_updates=[],
                competition_events=[],
                friction_events=[],
                traversal_results=[],
                affected_objects=[],
                new_objects=[],
                blocked_axes=[],
                spillover_opportunities=[],
                error="No objects parsed"
            )
        
        # Phase 1: Process each object
        object_updates = []
        competition_events = []
        friction_events = []
        total_mismatch = 0.0
        new_objects_count = 0
        affected_objects = []
        new_objects = []
        
        for parsed in parsed_objects:
            obj_name = parsed.object
            interrogatives = parsed.interrogatives
            sentiment = parsed.sentiment
            valence = sentiment_to_valence(sentiment)
            
            existing = self.buffer.get_object(obj_name)
            is_new = existing is None
            
            if is_new:
                new_objects_count += 1
                new_objects.append(obj_name)
                mismatch = 0.1
                axis_mismatches = {}
            else:
                mismatch, axis_mismatches = self.compute_valence_aware_mismatch(
                    existing, interrogatives, sentiment
                )
            
            # Create object update record
            obj_update = ObjectUpdate(
                object_name=obj_name,
                is_new=is_new,
                valence=valence,
                sentiment=sentiment,
                mismatch=mismatch
            )
            
            # Process each axis
            for axis in AXES:
                content = interrogatives.get(axis, "")
                if not content:
                    continue
                
                tagged_axis = get_valence_tagged_axis(axis, valence)
                
                # Get before values
                entropy_before = self.buffer.get_axis_entropy(obj_name, axis, valence) if not is_new else self.config.entropy.initial
                cryst_before = self.buffer.get_crystallization(obj_name, axis, valence) if not is_new else self.config.crystallization.initial
                
                axis_diff = axis_mismatches.get(tagged_axis, 0.05 if is_new else 0.0)
                
                # === CROSS-VALENCE FRICTION CHECK ===
                # Check for contradiction with opposing valence slots (only for existing objects)
                if not is_new:
                    friction = self.check_cross_valence_friction(obj_name, axis, valence, content)
                    if friction:
                        friction_events.append(friction)
                        # Update mismatch to reflect the friction
                        axis_diff = max(axis_diff, friction.contradiction_confidence * 0.3)
                
                # Determine action and apply changes
                if is_new:
                    action = "created"
                elif axis_diff > 0.15:
                    action = "contradicted"
                    # Increase entropy
                    if existing:
                        self.buffer.increase_entropy(obj_name, axis, valence, axis_diff)
                elif axis_diff <= 0.05:
                    action = "reinforced"
                    # Boost crystallization
                    self.buffer.increase_crystallization(obj_name, axis, valence)
                    # Decay entropy
                    self.buffer.decay_entropy(obj_name, axis, valence)
                    # Check for valence competition
                    competition = self.apply_valence_competition(obj_name, axis, valence)
                    if competition:
                        competition_events.append(competition)
                else:
                    action = "accumulated"
                
                # Get after values
                entropy_after = self.buffer.get_axis_entropy(obj_name, axis, valence)
                cryst_after = self.buffer.get_crystallization(obj_name, axis, valence)
                blocked = entropy_after >= self.config.entropy.threshold
                
                obj_update.axis_updates.append(AxisUpdate(
                    axis=axis,
                    valence=valence,
                    action=action,
                    mismatch=axis_diff,
                    entropy_before=entropy_before,
                    entropy_after=entropy_after,
                    crystallization_before=cryst_before,
                    crystallization_after=cryst_after,
                    blocked=blocked
                ))
            
            # Add/update object in buffer
            self.buffer.add_object(obj_name, interrogatives, sentiment)
            
            object_updates.append(obj_update)
            total_mismatch += mismatch
            affected_objects.append(obj_name)
        
        if parsed_objects:
            total_mismatch /= len(parsed_objects)
        
        # Phase 2: Calculate CCV dynamics
        clarity = self.calculate_clarity()
        arousal = self.calculate_arousal(total_mismatch, new_objects_count)
        valence_intensity = self.calculate_valence_intensity(arousal, clarity)
        curiosity = self.calculate_curiosity(arousal, total_mismatch, clarity)
        system_state = self.determine_system_state(curiosity, clarity)
        
        # Phase 3: Traversal analysis
        traversal_results = []
        blocked_axes = []
        spillover_opportunities = []
        
        for obj_update in object_updates:
            obj_name = obj_update.object_name
            existing = self.buffer.get_object(obj_name)
            
            if not existing:
                continue
            
            for axis_update in obj_update.axis_updates:
                tagged_axis = get_valence_tagged_axis(axis_update.axis, axis_update.valence)
                
                if axis_update.blocked:
                    traversal_results.append(TraversalResult(
                        object_name=obj_name,
                        tagged_axis=tagged_axis,
                        success=False,
                        entropy=axis_update.entropy_after,
                        reason="Entropy threshold exceeded"
                    ))
                    blocked_axes.append(f"{obj_name}[{tagged_axis}]")
                    
                    # Find spillover opportunities
                    for v in VALENCES:
                        if v != axis_update.valence:
                            other_tagged = get_valence_tagged_axis(axis_update.axis, v)
                            other_entropy = self.buffer.get_axis_entropy(obj_name, axis_update.axis, v)
                            other_content = existing['interrogatives'].get(other_tagged, "")
                            if other_entropy < self.config.entropy.threshold and other_content:
                                spillover_opportunities.append(
                                    f"{obj_name}[{other_tagged}]: '{other_content[:50]}'"
                                )
                else:
                    traversal_results.append(TraversalResult(
                        object_name=obj_name,
                        tagged_axis=tagged_axis,
                        success=True,
                        entropy=axis_update.entropy_after
                    ))
        
        # Create co-occurrence edges
        if len(affected_objects) >= 2:
            primary_valence = object_updates[0].valence if object_updates else VALENCE_NEUTRAL
            self.buffer.add_cooccurrence_edges(affected_objects, valence=primary_valence)
        
        # Save buffer
        self.buffer.save_buffer()
        
        # Create result
        result = ProcessingResult(
            input_text=text,
            timestamp=timestamp,
            clarity=clarity,
            arousal=arousal,
            valence_intensity=valence_intensity,
            curiosity=curiosity,
            system_state=system_state,
            object_updates=object_updates,
            competition_events=competition_events,
            friction_events=friction_events,
            traversal_results=traversal_results,
            affected_objects=affected_objects,
            new_objects=new_objects,
            blocked_axes=blocked_axes,
            spillover_opportunities=spillover_opportunities
        )
        
        # Add to history
        self.history.append(result)
        
        logger.info(
            f"Processed: clarity={clarity:.2f}, arousal={arousal:.2f}, "
            f"curiosity={curiosity:.2f}, state={system_state.value}"
        )
        
        return result
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Return summary of buffer state."""
        stats = self.buffer.get_statistics()
        clarity = self.calculate_clarity()
        
        return {
            'clarity': round(clarity, 3),
            'num_objects': stats['num_objects'],
            'num_edges': stats['num_edges'],
            'active_axes': stats['filled_slots'],
            'blocked_axes': stats['blocked_slots'],
            'avg_entropy': round(stats['avg_entropy'], 3),
            'avg_crystallization': round(stats['avg_crystallization'], 3),
        }
    
    def inspect_object(self, obj_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed view of an object's valence-tagged structure."""
        return self.buffer.get_node_summary(obj_name)
    
    def clear_history(self):
        """Clear processing history."""
        self.history = []
    
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent processing history."""
        return [r.to_dict() for r in self.history[-limit:]]
