"""
CCV System Configuration

Centralized configuration for all system parameters, paths, and settings.
Supports environment variable overrides and configuration file loading.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging for the CCV system."""
    logger = logging.getLogger("ccv")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class EntropyConfig:
    """Parameters controlling entropy dynamics."""
    threshold: float = 0.65          # Axis blocks when exceeded
    multiplier: float = 3.0          # Contradiction accumulation rate
    decay: float = 0.98              # Decay on successful traversal
    initial: float = 0.1             # Starting entropy for new axes
    
    def validate(self) -> bool:
        """Validate entropy parameters are within acceptable ranges."""
        return (
            0.0 < self.threshold <= 1.0 and
            0.0 < self.multiplier <= 10.0 and
            0.0 < self.decay < 1.0 and
            0.0 <= self.initial < self.threshold
        )


@dataclass
class CrystallizationConfig:
    """Parameters controlling crystallization (belief strength) dynamics."""
    boost: float = 0.05              # Reinforcement amount per confirmation
    initial: float = 0.5             # Starting crystallization
    minimum: float = 0.1             # Floor value (never goes below)
    maximum: float = 1.0             # Ceiling value
    
    def validate(self) -> bool:
        return (
            0.0 < self.boost <= 0.5 and
            self.minimum < self.initial < self.maximum and
            0.0 <= self.minimum < self.maximum <= 1.0
        )


@dataclass
class ValenceConfig:
    """Parameters controlling valence dynamics."""
    positive_threshold: float = 0.3   # Sentiment > this → positive valence
    negative_threshold: float = -0.3  # Sentiment < this → negative valence
    dominance_threshold: float = 0.2  # Gap needed for competition
    competition_penalty: float = 0.1  # Penalty ratio when dominated
    
    def validate(self) -> bool:
        return (
            -1.0 < self.negative_threshold < 0.0 < self.positive_threshold < 1.0 and
            0.0 < self.dominance_threshold < 1.0 and
            0.0 < self.competition_penalty < 1.0
        )


@dataclass
class CuriosityConfig:
    """Parameters controlling curiosity dynamics."""
    clarity_floor: float = 0.3       # Minimum clarity for curiosity to engage
    arousal_threshold: float = 0.05  # Minimum arousal for "something happening"
    mismatch_threshold: float = 0.05 # Minimum mismatch for curiosity trigger
    
    def validate(self) -> bool:
        return (
            0.0 < self.clarity_floor < 1.0 and
            0.0 <= self.arousal_threshold < 0.5 and
            0.0 <= self.mismatch_threshold < 0.5
        )


@dataclass 
class LLMConfig:
    """Configuration for LLM integration."""
    provider: str = "ollama"
    model: str = "gemma2:9b-instruct-q5_K_M"
    base_url: str = "http://localhost:11434/api/chat"
    timeout: int = 60
    temperature: float = 0.1
    context_length: int = 4096
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def validate(self) -> bool:
        return (
            self.timeout > 0 and
            0.0 <= self.temperature <= 2.0 and
            self.context_length > 0 and
            self.max_retries >= 0
        )


@dataclass
class PathConfig:
    """File and directory paths."""
    data_dir: str = "data"
    buffer_file: str = "buffer.pkl"
    backup_dir: str = "backups"
    log_file: Optional[str] = None
    export_dir: str = "exports"
    
    @property
    def buffer_path(self) -> Path:
        return Path(self.data_dir) / self.buffer_file
    
    @property
    def backup_path(self) -> Path:
        return Path(self.data_dir) / self.backup_dir
    
    def ensure_directories(self):
        """Create all necessary directories."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        Path(self.export_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class UIConfig:
    """User interface configuration."""
    theme: str = "dark"
    page_title: str = "CCV Meaning Engine"
    page_icon: str = "🧠"
    layout: str = "wide"
    graph_height: int = 500
    max_history_display: int = 50
    auto_refresh: bool = False
    refresh_interval: int = 5


@dataclass
class CCVConfig:
    """Master configuration containing all sub-configs."""
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    crystallization: CrystallizationConfig = field(default_factory=CrystallizationConfig)
    valence: ValenceConfig = field(default_factory=ValenceConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Versioning
    version: str = "1.0.0"
    
    def validate_all(self) -> Dict[str, bool]:
        """Validate all configuration sections."""
        return {
            "entropy": self.entropy.validate(),
            "crystallization": self.crystallization.validate(),
            "valence": self.valence.validate(),
            "curiosity": self.curiosity.validate(),
            "llm": self.llm.validate(),
        }
    
    def is_valid(self) -> bool:
        """Check if entire configuration is valid."""
        return all(self.validate_all().values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CCVConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            entropy=EntropyConfig(**data.get('entropy', {})),
            crystallization=CrystallizationConfig(**data.get('crystallization', {})),
            valence=ValenceConfig(**data.get('valence', {})),
            curiosity=CuriosityConfig(**data.get('curiosity', {})),
            llm=LLMConfig(**data.get('llm', {})),
            paths=PathConfig(**data.get('paths', {})),
            ui=UIConfig(**data.get('ui', {})),
            version=data.get('version', '1.0.0')
        )
    
    @classmethod
    def from_env(cls) -> 'CCVConfig':
        """Create configuration with environment variable overrides."""
        config = cls()
        
        # Override from environment variables
        if os.getenv('CCV_LLM_MODEL'):
            config.llm.model = os.getenv('CCV_LLM_MODEL')
        if os.getenv('CCV_LLM_URL'):
            config.llm.base_url = os.getenv('CCV_LLM_URL')
        if os.getenv('CCV_DATA_DIR'):
            config.paths.data_dir = os.getenv('CCV_DATA_DIR')
        if os.getenv('CCV_LOG_LEVEL'):
            # This would be used by the logging setup
            pass
        
        return config


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Default configuration instance
_config: Optional[CCVConfig] = None


def get_config() -> CCVConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = CCVConfig.from_env()
        _config.paths.ensure_directories()
    return _config


def set_config(config: CCVConfig):
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config():
    """Reset to default configuration."""
    global _config
    _config = None


# ============================================================================
# PARAMETER PRESETS
# ============================================================================

PRESETS = {
    "default": CCVConfig(),
    
    "sensitive": CCVConfig(
        entropy=EntropyConfig(threshold=0.5, multiplier=4.0),
        crystallization=CrystallizationConfig(boost=0.08),
        curiosity=CuriosityConfig(clarity_floor=0.4),
    ),
    
    "resilient": CCVConfig(
        entropy=EntropyConfig(threshold=0.8, multiplier=2.0, decay=0.95),
        crystallization=CrystallizationConfig(boost=0.03),
        curiosity=CuriosityConfig(clarity_floor=0.2),
    ),
    
    "rapid_learning": CCVConfig(
        entropy=EntropyConfig(multiplier=2.0, decay=0.99),
        crystallization=CrystallizationConfig(boost=0.1),
    ),
    
    "trauma_prone": CCVConfig(
        entropy=EntropyConfig(threshold=0.45, multiplier=5.0, decay=0.92),
        crystallization=CrystallizationConfig(boost=0.1),
        valence=ValenceConfig(dominance_threshold=0.15),
    ),
}


def get_preset(name: str) -> CCVConfig:
    """Get a configuration preset by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
