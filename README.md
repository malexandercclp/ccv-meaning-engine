# CCV Meaning Engine

**A computational framework for modeling meaning-making through pre-linguistic interrogative structures.**

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

The CCV (Curiosity, Clarity, Valence) system implements a novel approach to modeling how meaning is organized and processed in conscious experience. Rather than treating consciousness as information processing, CCV proposes that meaningful experience emerges from **pre-linguistic interrogative structures**—the implicit WHAT, WHEN, WHERE, HOW, and WHY that structure all phenomenal content.

### Key Concepts

- **Interrogative Axes**: Every object/concept is represented through five fundamental questions
- **Valence-Tagged Slots**: Each axis has three emotional containers (+/-/0) allowing the same concept to hold different meanings depending on emotional context
- **Crystallization**: Belief strength increases through reinforcement
- **Entropy**: Axis-specific confusion accumulates through contradiction
- **Curiosity**: Conditional engagement that requires sufficient clarity

### Emergent Phenomena

The system demonstrates that complex psychological dynamics emerge naturally from these structural principles:

- **Cognitive dissonance** (entropy accumulation from contradictions)
- **Belief crystallization** (reinforcement strengthening)
- **Defensive patterns** (valence competition and blockage)
- **Trauma responses** (high entropy blocking traversal)
- **Growth trajectories** (entropy decay through integration)

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) with a local LLM model

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ccv-framework.git
cd ccv-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull an LLM model for Ollama
ollama pull gemma2:9b-instruct-q5_K_M
```

---

## Quick Start

### 1. Seed the System (Optional)

Pre-load a complex life history for testing:

```bash
python seed_history.py
```

### 2. Launch the Interface

```bash
streamlit run ui/app.py
```

### 3. Process Experiences

Enter experiences in natural language:

```
"I saw my father today and felt anxious"
"Luna greeted me at the door with such joy"
"I received praise at work but felt like an imposter"
```

---

## Architecture

```
ccv_v1/
├── core/
│   ├── __init__.py      # Package exports
│   ├── config.py        # Configuration management
│   ├── exceptions.py    # Custom exceptions
│   ├── buffer.py        # Recurrence buffer (NetworkX graph)
│   ├── parser.py        # LLM integration for parsing
│   └── processor.py     # CCV dynamics engine
├── ui/
│   ├── __init__.py
│   └── app.py           # Streamlit interface
├── analysis/            # (Future: analysis tools)
├── tests/               # (Future: test suite)
├── docs/                # (Future: documentation)
├── data/                # Buffer persistence
├── seed_history.py      # Complex history seeder
├── requirements.txt
└── README.md
```

---

## Core Concepts

### The Recurrence Buffer

The buffer is a directed graph where each node represents an object/concept with:

```python
{
    'interrogatives': {
        'what+': "parent / caregiver",     # Positive identity
        'what-': "distant figure",          # Negative identity
        'why+': "provides security",        # Positive meaning
        'why-': "source of criticism",      # Negative meaning
        # ... all 15 valence-tagged axes
    },
    'crystallization': {
        'why-': 0.85,  # Strong negative belief
        'why+': 0.40,  # Weaker positive belief
    },
    'axis_entropy': {
        'why-': 0.12,  # Low confusion
        'why+': 0.45,  # Higher confusion
    },
    'sentiment': -0.3,  # Overall emotional tone
}
```

### CCV Dynamics

| Metric | Description | Range |
|--------|-------------|-------|
| **Clarity** | Global buffer navigability (inverse of average entropy) | 0-1 |
| **Arousal** | Attention signal from novelty/mismatch | 0-1 |
| **Valence Intensity** | Emotional significance (modulated by clarity) | 0-1 |
| **Curiosity** | Conditional engagement (requires clarity floor) | 0-1 |

### System States

| State | Curiosity | Clarity | Meaning |
|-------|-----------|---------|---------|
| **Engaged** | > 0.5 | ≥ 0.3 | Actively exploring |
| **Tentative** | 0-0.5 | ≥ 0.3 | Cautiously engaging |
| **Disengaged** | 0 | ≥ 0.3 | No curiosity, adequate clarity |
| **Withdrawn** | 0 | < 0.3 | Overwhelmed, can't engage |

---

## Configuration

### Parameters

```python
from ccv_v1.core import get_config

config = get_config()

# Entropy dynamics
config.entropy.threshold = 0.65      # Axis blocks when exceeded
config.entropy.multiplier = 3.0      # Contradiction accumulation rate
config.entropy.decay = 0.98          # Decay on successful traversal

# Crystallization dynamics
config.crystallization.boost = 0.05  # Reinforcement amount

# Valence competition
config.valence.dominance_threshold = 0.2  # Gap needed for competition
config.valence.competition_penalty = 0.1  # Penalty ratio
```

### Presets

```python
from ccv_v1.core import get_preset

# For modeling trauma-prone individuals
config = get_preset("trauma_prone")

# For modeling resilient individuals
config = get_preset("resilient")

# Available: default, sensitive, resilient, rapid_learning, trauma_prone
```

---

## API Usage

### Basic Processing

```python
from ccv_v1.core import CCVProcessor

processor = CCVProcessor()

# Process an experience
result = processor.process_input("I saw my father today")

print(f"State: {result.system_state.value}")
print(f"Clarity: {result.clarity:.3f}")
print(f"Curiosity: {result.curiosity:.3f}")
print(f"Affected: {result.affected_objects}")
```

### Inspecting Objects

```python
# Get detailed object view
summary = processor.inspect_object("father")

for axis, data in summary['axes'].items():
    for valence, slot in data.items():
        print(f"{axis}[{valence}]: {slot['content']}")
        print(f"  Crystallization: {slot['crystallization']:.2f}")
        print(f"  Entropy: {slot['entropy']:.2f}")
```

### Buffer Operations

```python
buffer = processor.buffer

# Statistics
stats = buffer.get_statistics()
print(f"Objects: {stats['num_objects']}")
print(f"Fill rate: {stats['fill_rate']:.1%}")

# Backup
backup_path = buffer.create_backup("before_experiment")

# Export
data = buffer.to_dict()

# Clear (creates backup automatically)
buffer.clear()
```

---

## Theoretical Background

### The CCV Framework

The CCV framework proposes that consciousness is not primarily about information processing, but about **meaning organization**. Meaningful experience has intrinsic interrogative structure:

1. **WHAT** - Categorical identity ("What is this?")
2. **WHEN** - Temporal context ("When does this apply?")
3. **WHERE** - Spatial context ("Where is this located?")
4. **HOW** - Procedural knowledge ("How does this work?")
5. **WHY** - Causal/purposive meaning ("Why does this matter?")

These interrogatives are **pre-linguistic**—they structure experience before and independent of language. An infant experiences the world interrogatively before acquiring words.

### Valence as Structural Differentiation

The same object can mean different things in different emotional contexts. "Father" might simultaneously be:

- **why+**: "Provides for the family"
- **why-**: "Never emotionally available"

These aren't contradictions—they're different aspects held in different valence containers. True contradiction only occurs when incompatible content tries to occupy the **same** valence slot.

### Emergence Without Programming

The system demonstrates that complex psychological phenomena emerge from structural dynamics without explicit programming:

- We don't code "cognitive dissonance"—it emerges from entropy accumulation
- We don't code "defense mechanisms"—they emerge from blockage and spillover
- We don't code "trauma responses"—they emerge from high-entropy crystallization

---

## Research Applications

The CCV system is designed for research into:

- **Computational consciousness**: Testing structural theories of phenomenal experience
- **AI alignment**: Understanding how meaning shapes behavior
- **Therapeutic modeling**: Simulating therapeutic interventions
- **Cultural cognition**: Modeling how shared frameworks shape individual meaning

---

## Publications

Maxwell's work on the CCV framework has been published on PhilArchive:
- CCV: A Structural Framework for Computational Consciousness
- Emergent Psychological Dynamics in the CCV System

Links and citations available upon request.

---

## Contributing

This is an early-stage research project. If you're interested in collaborating or have questions:

- Open an issue on GitHub
- Reach out via email (malexandercclp@gmail.com)

### Development Setup

```bash
# Install dev dependencies (when available)
pip install -r requirements-dev.txt

# Run tests (when test suite is complete)
pytest

# Format code
black ccv_v1/
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

Free for personal and research use.

---

## Acknowledgments

- Built with [NetworkX](https://networkx.org/), [Streamlit](https://streamlit.io/), and [Ollama](https://ollama.ai/)
- Theoretical foundations inspired by phenomenological philosophy and consciousness studies

---

## Contact & Updates

- **Substack**: [https://substack.com/@maxwellalexander?]
- **Email**: [malexandercclp@gmail.com]
- **PhilArchive**: [https://philpeople.org/profiles/maxwell-alexander-1]

---

<p align="center">
  <strong>CCV Meaning Engine</strong><br>
  <em>Pre-linguistic interrogative architecture for computational consciousness</em>
</p>
