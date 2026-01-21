"""
CCV LLM Parser

Parses natural language input into structured objects with interrogative relationships.
Uses local LLM (Ollama) for semantic extraction.

Handles:
    - Object extraction from sentences
    - Interrogative axis identification (what/when/where/how/why)
    - Sentiment analysis
    - Self-reference normalization (I/me/my → "me")
"""

import requests
import json
import re
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from .config import get_config
from .exceptions import (
    LLMConnectionError, LLMTimeoutError, LLMResponseError, ParseFailedError,
    validate_input_text, validate_sentiment
)

logger = logging.getLogger("ccv.parser")

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ParsedObject:
    """A parsed object with its interrogative structure."""
    object: str
    interrogatives: Dict[str, str]
    sentiment: float
    
    def __post_init__(self):
        """Validate and normalize on creation."""
        self.object = self.object.strip().lower()
        self.sentiment = validate_sentiment(self.sentiment)
        # Clean interrogatives
        self.interrogatives = {
            k.strip().lower(): v.strip() 
            for k, v in self.interrogatives.items() 
            if v and v.strip()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'object': self.object,
            'interrogatives': self.interrogatives,
            'sentiment': self.sentiment
        }


# ============================================================================
# LLM COMMUNICATION
# ============================================================================

def _call_ollama(
    prompt: str, 
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Call Ollama API with retry logic.
    
    Args:
        prompt: The prompt to send
        model: Model name (uses config default if None)
        timeout: Request timeout (uses config default if None)
        temperature: Generation temperature (uses config default if None)
        
    Returns:
        Response text from LLM
        
    Raises:
        LLMConnectionError: If connection fails
        LLMTimeoutError: If request times out
        LLMResponseError: If response is invalid
    """
    config = get_config()
    model = model or config.llm.model
    timeout = timeout or config.llm.timeout
    temperature = temperature if temperature is not None else config.llm.temperature
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": config.llm.context_length,
        }
    }
    
    last_error = None
    
    for attempt in range(config.llm.max_retries + 1):
        try:
            logger.debug(f"Ollama request attempt {attempt + 1}/{config.llm.max_retries + 1}")
            
            response = requests.post(
                config.llm.base_url,
                json=payload,
                timeout=(8, timeout)  # (connect timeout, read timeout)
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("message", {}).get("content", "")
            
            if not content:
                raise LLMResponseError("", "Empty response from LLM")
            
            logger.debug(f"Ollama response received ({len(content)} chars)")
            return content
            
        except requests.exceptions.Timeout as e:
            last_error = LLMTimeoutError(timeout)
            logger.warning(f"Ollama timeout (attempt {attempt + 1})")
            
        except requests.exceptions.ConnectionError as e:
            last_error = LLMConnectionError(config.llm.base_url, e)
            logger.warning(f"Ollama connection error (attempt {attempt + 1}): {e}")
            
        except requests.exceptions.RequestException as e:
            last_error = LLMConnectionError(config.llm.base_url, e)
            logger.warning(f"Ollama request error (attempt {attempt + 1}): {e}")
        
        # Wait before retry
        if attempt < config.llm.max_retries:
            time.sleep(config.llm.retry_delay * (attempt + 1))
    
    # All retries exhausted
    raise last_error


def _extract_json_from_response(text: str) -> Optional[List[Dict]]:
    """
    Extract JSON array from LLM response text.
    
    Handles:
        - Markdown code fences
        - Extra text before/after JSON
        - Malformed JSON with recovery attempts
    """
    if not text:
        return None
    
    # Strip markdown/code fences
    text = re.sub(r'^```json?\s*', '', text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
    
    # Find JSON array
    start = text.find('[')
    if start == -1:
        return None
    
    # Find matching closing bracket
    count = 0
    for i in range(start, len(text)):
        if text[i] == '[':
            count += 1
        elif text[i] == ']':
            count -= 1
        
        if count == 0:
            try:
                return json.loads(text[start:i+1])
            except json.JSONDecodeError:
                # Try to fix common issues
                json_str = text[start:i+1]
                # Fix trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
            break
    
    return None


# ============================================================================
# PARSING PROMPTS
# ============================================================================

PARSE_PROMPT_TEMPLATE = '''You are a semantic parser that extracts objects and their interrogative relationships from sentences. Return ONLY a valid JSON array.

Input: "{text}"

For each object/concept in the sentence, identify which INTERROGATIVE AXES apply:

INTERROGATIVE DEFINITIONS (use these precisely):
- "what": Category, type, or essential nature. What IS it? (e.g., "beverage", "emotion", "tool", "person")
- "when": Temporal context. When does it occur/apply? (e.g., "morning", "daily", "during winter", "at night")  
- "where": Spatial/locational context. Where is it? (e.g., "at home", "in the office", "kitchen")
- "how": Method, process, or manner. How does it work/happen? (e.g., "brewed", "slowly", "with effort")
- "why": Cause, reason, purpose, or effect. Why does it matter? What does it cause? (e.g., "for energy", "causes anxiety", "because tired", "gives focus")

CRITICAL: Causal language maps to "why":
- "X gives me Y" → X has why: "gives Y" or "causes Y"
- "X makes me Y" → X has why: "makes Y" or "causes Y"  
- "X because Y" → X has why: "Y"
- "X causes Y" → X has why: "causes Y"
- "X helps with Y" → X has why: "helps with Y"

Output format:
[
  {{"object": "name", "interrogatives": {{"what": "...", "when": "...", "where": "...", "how": "...", "why": "..."}}, "sentiment": 0.0}}
]

Rules:
- Extract 1-4 main objects/concepts
- ONLY include interrogative keys that have actual content from the sentence
- Omit keys entirely if not mentioned (don't use empty strings)
- sentiment: -1.0 (very negative) to +1.0 (very positive)
- IMPORTANT: Always normalize first-person references (I, me, my, myself, mine) to the object "me"

EXAMPLES:

Input: "Coffee gives me energy in the morning"
Output: [
  {{"object": "coffee", "interrogatives": {{"what": "beverage", "when": "morning", "why": "gives energy"}}, "sentiment": 0.7}},
  {{"object": "me", "interrogatives": {{"what": "person", "why": "energized by coffee"}}, "sentiment": 0.8}}
]

Input: "I love my job"
Output: [
  {{"object": "me", "interrogatives": {{"what": "person", "why": "loves job"}}, "sentiment": 0.8}},
  {{"object": "job", "interrogatives": {{"what": "work", "why": "loved by me"}}, "sentiment": 0.8}}
]

Input: "I hate running because it hurts my knees"
Output: [
  {{"object": "me", "interrogatives": {{"what": "person", "why": "hates running"}}, "sentiment": -0.5}},
  {{"object": "running", "interrogatives": {{"what": "exercise", "why": "hurts knees"}}, "sentiment": -0.7}},
  {{"object": "knees", "interrogatives": {{"what": "body part", "why": "hurt by running"}}, "sentiment": -0.5}}
]

Input: "My dog sleeps on the couch every afternoon"
Output: [
  {{"object": "dog", "interrogatives": {{"what": "pet", "when": "every afternoon", "where": "on the couch", "how": "sleeps"}}, "sentiment": 0.3}}
]

Return ONLY the JSON array for the input sentence. No explanation.'''


# ============================================================================
# FALLBACK PARSING
# ============================================================================

# Self-reference patterns
SELF_REFERENCES = {'i', 'me', 'my', 'myself', 'mine', "i'm", "im", "i've", "ive"}

def _normalize_self_references(obj_name: str) -> str:
    """Normalize first-person references to 'me'."""
    if obj_name.lower().strip() in SELF_REFERENCES:
        return 'me'
    return obj_name


def _simple_sentiment(text: str) -> float:
    """
    Simple rule-based sentiment analysis as fallback.
    
    This is a backup for when LLM parsing fails.
    """
    text = text.lower()
    
    positive_words = {
        'love', 'happy', 'joy', 'excited', 'great', 'wonderful', 'amazing',
        'good', 'fantastic', 'excellent', 'beautiful', 'peaceful', 'calm',
        'grateful', 'thankful', 'blessed', 'proud', 'confident', 'hopeful',
        'success', 'win', 'achieve', 'accomplish', 'enjoy', 'fun', 'delight'
    }
    
    negative_words = {
        'hate', 'sad', 'angry', 'frustrated', 'anxious', 'worried', 'scared',
        'terrible', 'awful', 'horrible', 'bad', 'painful', 'hurt', 'suffer',
        'fail', 'failure', 'lose', 'lost', 'fear', 'stress', 'overwhelm',
        'depressed', 'lonely', 'alone', 'worthless', 'hopeless', 'ashamed'
    }
    
    words = set(re.findall(r'\b\w+\b', text))
    
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    
    if pos_count == 0 and neg_count == 0:
        return 0.0
    
    total = pos_count + neg_count
    sentiment = (pos_count - neg_count) / total
    
    return max(-1.0, min(1.0, sentiment * 0.8))  # Scale to avoid extremes


def _fallback_parse(text: str) -> List[ParsedObject]:
    """
    Simple fallback parsing when LLM is unavailable.
    
    Extracts basic noun phrases and applies simple heuristics.
    """
    logger.info("Using fallback parser")
    
    text = validate_input_text(text)
    sentiment = _simple_sentiment(text)
    
    # Extract potential objects (simple noun extraction)
    # This is very basic but provides something rather than nothing
    words = text.split()
    objects = []
    
    # Look for capitalized words (potential proper nouns)
    for word in words:
        cleaned = re.sub(r'[^\w]', '', word)
        if cleaned and (cleaned[0].isupper() or cleaned.lower() in SELF_REFERENCES):
            obj_name = _normalize_self_references(cleaned)
            if obj_name not in [o.object for o in objects]:
                objects.append(ParsedObject(
                    object=obj_name,
                    interrogatives={'what': 'concept'},
                    sentiment=sentiment
                ))
    
    # If no objects found, create a generic one
    if not objects:
        # Take first few meaningful words as the object
        meaningful = [w for w in words if len(w) > 2 and w.lower() not in 
                     {'the', 'and', 'but', 'for', 'with', 'was', 'were', 'are', 'been'}]
        if meaningful:
            obj_name = meaningful[0].lower()
            obj_name = _normalize_self_references(obj_name)
        else:
            obj_name = 'experience'
        
        objects.append(ParsedObject(
            object=obj_name,
            interrogatives={'what': text[:100]},
            sentiment=sentiment
        ))
    
    return objects[:4]  # Limit to 4 objects


# ============================================================================
# MAIN PARSING FUNCTIONS
# ============================================================================

def parse_input(text: str, use_fallback_on_error: bool = True) -> List[ParsedObject]:
    """
    Parse natural language input into structured objects.
    
    Args:
        text: Input text to parse
        use_fallback_on_error: If True, use simple fallback when LLM fails
        
    Returns:
        List of ParsedObject instances
        
    Raises:
        ParseFailedError: If parsing fails and fallback is disabled
    """
    # Validate input
    text = validate_input_text(text)
    logger.info(f"Parsing input: {text[:50]}...")
    
    try:
        # Build prompt
        prompt = PARSE_PROMPT_TEMPLATE.format(text=text)
        
        # Call LLM
        raw_response = _call_ollama(prompt)
        logger.debug(f"Raw LLM response: {raw_response[:200]}...")
        
        # Extract JSON
        parsed = _extract_json_from_response(raw_response)
        
        if not parsed or not isinstance(parsed, list):
            raise LLMResponseError(raw_response, "Could not extract valid JSON array")
        
        # Convert to ParsedObjects
        objects = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            
            obj_name = item.get('object', '')
            if not obj_name:
                continue
            
            # Normalize self-references
            obj_name = _normalize_self_references(obj_name)
            
            interrogatives = item.get('interrogatives', {})
            if not isinstance(interrogatives, dict):
                interrogatives = {}
            
            sentiment = item.get('sentiment', 0.0)
            
            try:
                parsed_obj = ParsedObject(
                    object=obj_name,
                    interrogatives=interrogatives,
                    sentiment=sentiment
                )
                objects.append(parsed_obj)
            except Exception as e:
                logger.warning(f"Error creating ParsedObject: {e}")
                continue
        
        if not objects:
            raise ParseFailedError(text, "No valid objects extracted")
        
        logger.info(f"Successfully parsed {len(objects)} objects")
        return objects
        
    except (LLMConnectionError, LLMTimeoutError, LLMResponseError, ParseFailedError) as e:
        logger.warning(f"LLM parsing failed: {e}")
        
        if use_fallback_on_error:
            return _fallback_parse(text)
        else:
            raise ParseFailedError(text, str(e))


def compute_mismatch(
    existing: Optional[Dict[str, Any]], 
    new_interrogatives: Dict[str, str],
    new_sentiment: float
) -> Tuple[float, Dict[str, float]]:
    """
    Compute mismatch between existing buffer content and new input.
    
    This is used for non-valence-aware comparison (legacy compatibility).
    For valence-aware comparison, use the CCV processor's method.
    
    Args:
        existing: Existing object data from buffer (or None if new)
        new_interrogatives: New interrogative content
        new_sentiment: Sentiment of new input
        
    Returns:
        Tuple of (total_mismatch, axis_mismatches_dict)
    """
    if existing is None:
        # New object - moderate mismatch (novelty)
        return 0.68, {}
    
    # Sentiment difference component
    existing_sentiment = existing.get('sentiment', 0.0)
    sent_diff = abs(new_sentiment - existing_sentiment)
    
    # Interrogative differences
    axis_mismatches = {}
    inter_diff = 0.0
    axis_count = 0
    
    existing_inter = existing.get('interrogatives', {})
    
    for axis in ['what', 'when', 'where', 'how', 'why']:
        new_content = new_interrogatives.get(axis, '')
        if not new_content:
            continue
        
        axis_count += 1
        
        # Check all valence slots for this axis
        existing_content = ''
        for valence in ['+', '-', '0']:
            tagged = f"{axis}{valence}"
            content = existing_inter.get(tagged, '')
            if content:
                existing_content = content
                break
        
        if existing_content:
            # Compare content
            if new_content.lower() == existing_content.lower():
                axis_diff = 0.0  # Exact match
            elif new_content.lower() in existing_content.lower():
                axis_diff = 0.05  # Subset
            else:
                axis_diff = 0.25  # Different content
        else:
            axis_diff = 0.05  # New content for this axis
        
        axis_mismatches[axis] = axis_diff
        inter_diff += axis_diff
    
    if axis_count > 0:
        inter_diff /= axis_count
    
    # Weighted combination
    total_mismatch = 0.55 * sent_diff + 0.45 * inter_diff
    
    return total_mismatch, axis_mismatches


# ============================================================================
# SEMANTIC ANALYSIS (for future enhancement)
# ============================================================================

def analyze_semantic_similarity(text1: str, text2: str) -> float:
    """
    Analyze semantic similarity between two texts.
    
    Currently uses simple heuristics. Could be enhanced with embeddings.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score 0.0 to 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'between', 'under', 'again', 'further', 'then', 'once'}
    
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def detect_contradiction(text1: str, text2: str) -> Tuple[bool, float]:
    """
    Detect if two texts represent contradictory statements.
    
    Currently uses simple heuristics. Could be enhanced with NLI models.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Tuple of (is_contradiction, confidence)
    """
    if not text1 or not text2:
        return False, 0.0
    
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    # Check for negation patterns
    negation_pairs = [
        ('not ', ''),
        ('never ', 'always '),
        ("don't ", 'do '),
        ("doesn't ", 'does '),
        ("didn't ", 'did '),
        ("won't ", 'will '),
        ("can't ", 'can '),
        ("isn't ", 'is '),
        ("aren't ", 'are '),
        ("wasn't ", 'was '),
        ("weren't ", 'were '),
    ]
    
    # Check for opposite sentiment indicators
    positive_indicators = {'love', 'like', 'enjoy', 'happy', 'good', 'great', 'wonderful', 'gentle', 'kind', 'caring', 'supportive', 'protective', 'taught', 'helped', 'provided'}
    negative_indicators = {'hate', 'dislike', 'sad', 'bad', 'terrible', 'awful', 'horrible', 'beat', 'hit', 'hurt', 'abuse', 'neglect', 'abandon', 'scream', 'yell', 'drunk', 'nothing'}
    
    words1 = set(text1_lower.split())
    words2 = set(text2_lower.split())
    
    has_pos1 = bool(words1 & positive_indicators)
    has_neg1 = bool(words1 & negative_indicators)
    has_pos2 = bool(words2 & positive_indicators)
    has_neg2 = bool(words2 & negative_indicators)
    
    # Contradiction if one is positive and other is negative about similar content
    if (has_pos1 and has_neg2) or (has_neg1 and has_pos2):
        # Check if they're about the same thing
        content_similarity = analyze_semantic_similarity(text1, text2)
        if content_similarity > 0.3:
            return True, min(0.8, content_similarity + 0.3)
    
    return False, 0.0


def llm_detect_contradiction(text1: str, text2: str) -> Tuple[bool, float]:
    """
    Use LLM to detect semantic contradiction between two texts.
    
    More accurate than heuristic-based detection, especially for
    implicit contradictions like "beat me" vs "was gentle".
    
    Args:
        text1: First text (e.g., existing belief content)
        text2: Second text (e.g., new incoming content)
        
    Returns:
        Tuple of (is_contradiction, confidence 0.0-1.0)
    """
    if not text1 or not text2:
        return False, 0.0
    
    prompt = f'''Determine if these two statements about the same subject are contradictory.

Statement A: "{text1}"
Statement B: "{text2}"

Contradictory means they cannot both be true about the same person/thing.
Examples of contradictions:
- "beat me" vs "was gentle and kind" (contradictory)
- "never there for me" vs "always supported me" (contradictory)  
- "taught me to fish" vs "took me to games" (NOT contradictory - both can be true)

Respond with ONLY a JSON object:
{{"contradictory": true/false, "confidence": 0.0-1.0}}'''

    try:
        response = _call_ollama(prompt, temperature=0.1)
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            result = json.loads(json_match.group())
            is_contradiction = result.get('contradictory', False)
            confidence = float(result.get('confidence', 0.0))
            return is_contradiction, confidence
    except Exception as e:
        logger.warning(f"LLM contradiction detection failed: {e}")
    
    # Fall back to heuristic detection
    return detect_contradiction(text1, text2)
