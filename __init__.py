"""
CCV (Curiosity, Clarity, Valence) System

A computational framework for modeling meaning-making through pre-linguistic
interrogative structures.

The CCV system implements:
- Recurrence buffer with valence-tagged interrogative axes
- Entropy and crystallization dynamics
- Curiosity as conditional engagement
- Clarity as global navigability

Usage:
    from ccv_v1.core import CCVProcessor
    
    processor = CCVProcessor()
    result = processor.process_input("I saw my father today")
    print(result.system_state)  # engaged/tentative/disengaged/withdrawn
"""

__version__ = "1.0.0"
__author__ = "Maxwell"

from .core import (
    CCVProcessor,
    RecurrenceBuffer,
    CCVConfig,
    get_config,
    SystemState,
)

__all__ = [
    '__version__',
    'CCVProcessor',
    'RecurrenceBuffer', 
    'CCVConfig',
    'get_config',
    'SystemState',
]
