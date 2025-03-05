# FM Synthesis Python Library

A study repository for exploring frequency modulation synthesis in Python.

## Overview

This project is a Python implementation of an FM synthesis engine with supporting music theory algorithms. It provides a framework for programmatic sound design, algorithmic composition, and procedural music generation through FM synthesis techniques.

## Features

- **FM Synthesis Engine**: Complete FM synthesis engine with operators, algorithms, and audio effects
- **Music Theory Library**: Implementation of notes, scales, chords, intervals, and rhythms
- **City Pop Generator**: Example procedural generation of city pop / jazz fusion tracks
- **Audio Export**: Save synthesized audio to WAV files

## Installation

```bash
# Clone the repository
git clone https://github.com/cathodeDreams/fm_synth.git
cd fm_synth

# Install dependencies
pip install numpy sounddevice soundfile psutil
```

## Usage

Run the main example:

```bash
python main.py
```

Try just the FM synthesis examples:

```bash
python fm_synth.py
```

Test the city pop generator:

```python
from citypop_generator import generate_and_play_track
generate_and_play_track(root_note='F', octave=4, tempo=88)
```

## Project Structure

- `fm_synth.py` - Core FM synthesis implementation
- `music_theory.py` - Music theory algorithms and data structures
- `citypop_generator.py` - Procedural music generator example
- `main.py` - Main entry point and demonstration

## License

MIT