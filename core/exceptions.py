"""
CCV System Exceptions

Custom exception classes for clear error handling and debugging.
"""


class CCVError(Exception):
    """Base exception for all CCV system errors."""
    pass


# ============================================================================
# BUFFER ERRORS
# ============================================================================

class BufferError(CCVError):
    """Errors related to the recurrence buffer."""
    pass


class BufferLoadError(BufferError):
    """Failed to load buffer from disk."""
    pass


class BufferSaveError(BufferError):
    """Failed to save buffer to disk."""
    pass


class BufferCorruptionError(BufferError):
    """Buffer data is corrupted or invalid."""
    pass


class ObjectNotFoundError(BufferError):
    """Requested object does not exist in buffer."""
    def __init__(self, object_name: str):
        self.object_name = object_name
        super().__init__(f"Object not found in buffer: '{object_name}'")


# ============================================================================
# PARSER ERRORS  
# ============================================================================

class ParserError(CCVError):
    """Errors related to input parsing."""
    pass


class LLMConnectionError(ParserError):
    """Failed to connect to LLM service."""
    def __init__(self, url: str, original_error: Exception = None):
        self.url = url
        self.original_error = original_error
        super().__init__(f"Failed to connect to LLM at {url}: {original_error}")


class LLMTimeoutError(ParserError):
    """LLM request timed out."""
    def __init__(self, timeout: int):
        self.timeout = timeout
        super().__init__(f"LLM request timed out after {timeout} seconds")


class LLMResponseError(ParserError):
    """LLM returned an invalid or unparseable response."""
    def __init__(self, response: str, reason: str = ""):
        self.response = response
        self.reason = reason
        super().__init__(f"Invalid LLM response: {reason}")


class ParseFailedError(ParserError):
    """Failed to parse input text into structured objects."""
    def __init__(self, input_text: str, reason: str = ""):
        self.input_text = input_text[:100]  # Truncate for safety
        self.reason = reason
        super().__init__(f"Failed to parse input: {reason}")


# ============================================================================
# PROCESSING ERRORS
# ============================================================================

class ProcessingError(CCVError):
    """Errors during experience processing."""
    pass


class InvalidInputError(ProcessingError):
    """Input text is invalid or empty."""
    def __init__(self, reason: str = ""):
        super().__init__(f"Invalid input: {reason}")


class TraversalError(ProcessingError):
    """Error during axis traversal."""
    pass


# ============================================================================
# CONFIGURATION ERRORS
# ============================================================================

class ConfigurationError(CCVError):
    """Errors related to configuration."""
    pass


class InvalidConfigError(ConfigurationError):
    """Configuration values are invalid."""
    def __init__(self, section: str, details: str = ""):
        self.section = section
        super().__init__(f"Invalid configuration in [{section}]: {details}")


class ConfigFileError(ConfigurationError):
    """Error loading or saving configuration file."""
    pass


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_input_text(text: str) -> str:
    """
    Validate and sanitize input text.
    
    Args:
        text: Raw input text
        
    Returns:
        Sanitized text
        
    Raises:
        InvalidInputError: If text is invalid
    """
    if text is None:
        raise InvalidInputError("Input cannot be None")
    
    if not isinstance(text, str):
        raise InvalidInputError(f"Input must be string, got {type(text).__name__}")
    
    text = text.strip()
    
    if not text:
        raise InvalidInputError("Input cannot be empty")
    
    if len(text) > 10000:
        raise InvalidInputError("Input exceeds maximum length of 10000 characters")
    
    # Basic sanitization - remove control characters except newlines/tabs
    sanitized = ''.join(
        char for char in text 
        if char.isprintable() or char in '\n\t'
    )
    
    return sanitized


def validate_object_name(name: str) -> str:
    """
    Validate and normalize object name.
    
    Args:
        name: Raw object name
        
    Returns:
        Normalized name (lowercase, stripped)
        
    Raises:
        InvalidInputError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise InvalidInputError("Object name must be a non-empty string")
    
    name = name.strip().lower()
    
    if not name:
        raise InvalidInputError("Object name cannot be empty after normalization")
    
    if len(name) > 200:
        raise InvalidInputError("Object name exceeds maximum length of 200 characters")
    
    return name


def validate_sentiment(sentiment: float) -> float:
    """
    Validate sentiment value.
    
    Args:
        sentiment: Raw sentiment value
        
    Returns:
        Clamped sentiment value in [-1, 1]
    """
    try:
        sentiment = float(sentiment)
    except (TypeError, ValueError):
        return 0.0
    
    return max(-1.0, min(1.0, sentiment))


def validate_axis(axis: str) -> str:
    """
    Validate interrogative axis name.
    
    Args:
        axis: Axis name
        
    Returns:
        Validated axis name
        
    Raises:
        InvalidInputError: If axis is invalid
    """
    valid_axes = {'what', 'when', 'where', 'how', 'why'}
    
    if not axis or not isinstance(axis, str):
        raise InvalidInputError("Axis must be a non-empty string")
    
    axis = axis.strip().lower()
    
    # Handle valence-tagged axes like 'why+', 'what-'
    base_axis = axis.rstrip('+-0')
    
    if base_axis not in valid_axes:
        raise InvalidInputError(f"Invalid axis: {axis}. Must be one of {valid_axes}")
    
    return axis


def validate_valence(valence: str) -> str:
    """
    Validate valence tag.
    
    Args:
        valence: Valence character
        
    Returns:
        Validated valence
        
    Raises:
        InvalidInputError: If valence is invalid
    """
    valid_valences = {'+', '-', '0'}
    
    if valence not in valid_valences:
        raise InvalidInputError(f"Invalid valence: {valence}. Must be one of {valid_valences}")
    
    return valence
