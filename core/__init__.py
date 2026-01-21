"""
CCV Core Module

Core components for the CCV (Curiosity, Clarity, Valence) system.
"""

from .config import (
    CCVConfig,
    get_config,
    set_config,
    reset_config,
    get_preset,
    setup_logging,
    PRESETS,
)

from .exceptions import (
    CCVError,
    BufferError,
    BufferLoadError,
    BufferSaveError,
    ParserError,
    ProcessingError,
    InvalidInputError,
)

from .buffer import (
    RecurrenceBuffer,
    AXES,
    VALENCES,
    VALENCE_POSITIVE,
    VALENCE_NEGATIVE,
    VALENCE_NEUTRAL,
    sentiment_to_valence,
    get_valence_tagged_axis,
    parse_valence_tagged_axis,
)

from .parser import (
    parse_input,
    ParsedObject,
)

from .processor import (
    CCVProcessor,
    ProcessingResult,
    SystemState,
    ObjectUpdate,
    AxisUpdate,
    CompetitionEvent,
    TraversalResult,
)

__all__ = [
    # Config
    'CCVConfig',
    'get_config',
    'set_config',
    'reset_config',
    'get_preset',
    'setup_logging',
    'PRESETS',
    
    # Exceptions
    'CCVError',
    'BufferError',
    'BufferLoadError',
    'BufferSaveError',
    'ParserError',
    'ProcessingError',
    'InvalidInputError',
    
    # Buffer
    'RecurrenceBuffer',
    'AXES',
    'VALENCES',
    'VALENCE_POSITIVE',
    'VALENCE_NEGATIVE',
    'VALENCE_NEUTRAL',
    'sentiment_to_valence',
    'get_valence_tagged_axis',
    'parse_valence_tagged_axis',
    
    # Parser
    'parse_input',
    'ParsedObject',
    
    # Processor
    'CCVProcessor',
    'ProcessingResult',
    'SystemState',
    'ObjectUpdate',
    'AxisUpdate',
    'CompetitionEvent',
    'TraversalResult',
]
