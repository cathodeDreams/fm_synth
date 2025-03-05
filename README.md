# FM Synthesis Python Library

A Python implementation of a frequency modulation synthesis engine with music theory algorithms and procedural music generation.

## Overview

This project provides a comprehensive framework for programmatic sound design, algorithmic composition, and procedural music generation using FM synthesis techniques. It combines a powerful FM synthesis engine with music theory algorithms to enable complex sound creation and musical composition.

## Features

- **FM Synthesis Engine**: Complete FM synthesis engine with operators, algorithms, and audio effects
- **Music Theory Library**: Implementation of notes, scales, chords, intervals, and rhythms
- **Procedural Music Generation**: Genre-specific algorithmic music generators
- **Audio Effects**: Delay, reverb, chorus, and other audio processing effects
- **Audio Export**: Save synthesized audio to WAV files

## Current Genres

- **City Pop**: Procedural generation of city pop and jazz fusion tracks
- **Ambient House**: Algorithmic ambient and electronic music generation

## Installation

```bash
# Clone the repository
git clone https://github.com/cathodeDreams/fm_synth.git
cd fm_synth

# Install dependencies
pip install numpy sounddevice soundfile psutil
```

## Usage

### Command Line Interface

Generate music with custom parameters:

```bash
# Generate city pop in F Lydian at 90 BPM
python main.py --style citypop --root F --octave 4 --tempo 90

# Generate ambient house in C minor at 110 BPM
python main.py --style ambient_house --root C --octave 3 --tempo 110
```

### API Usage

Generate city pop programmatically:

```python
from citypop_generator import generate_and_play_track
generate_and_play_track(root_note='F', octave=4, tempo=88)
```

Generate ambient house:

```python
from ambient_house_generator import generate_and_play_track
generate_and_play_track(root_note='C', octave=3, tempo=110)
```

Run FM synthesis examples:

```bash
python fm_synth.py
```

## Project Structure

### Core Components
- `fm_synth.py` - Core FM synthesis implementation
- `music_theory.py` - Music theory algorithms and data structures
- `modulation.py` - Envelope and LFO framework (planned)
- `spatialization.py` - Stereo audio processing (planned)

### Music Generation
- `core_music_generator.py` - Base framework for procedural music generation
- `citypop_generator.py` - City pop genre generator
- `ambient_house_generator.py` - Ambient house genre generator

### Application
- `main.py` - Command-line interface
- `todo.md` - Development roadmap and implementation plans

## Roadmap

We are working on several major enhancements to the synthesis engine:

1. **Stereophonic Output**: Full stereo synthesis with panning controls
2. **LFO System**: Flexible modulation system for dynamic sound design
3. **Enhanced Envelopes**: More sophisticated envelope system with multiple stages and curve shaping

See the [todo.md](todo.md) file for detailed implementation plans.

## License

MIT