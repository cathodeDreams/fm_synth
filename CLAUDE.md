# CLAUDE.md - FM Synth Project

## Commands

### Run Commands
- Main project: `python main.py` (supports style, key, octave, tempo options)
- FM synth examples: `python fm_synth.py`
- Generate city pop: `python -c "from citypop_generator import generate_and_play_track; generate_and_play_track(root_note='F', octave=4, tempo=88)"`

### Installation
- Install dependencies: `pip install numpy sounddevice soundfile psutil`

### Development
- Run tests: (None implemented yet - consider adding pytest)
- Linting: `python -m pylint *.py` (if pylint is installed)

## Code Style Guidelines
- **Imports**: Standard library → third-party packages → local modules
- **Formatting**: PEP 8 with descriptive variable names
- **Types**: Use type hints (List, Dict, Optional, Union, etc.)
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Error Handling**: Try/except blocks for expected exceptions
- **Documentation**: Google style docstrings for all functions/classes
- **Comments**: Explain complex algorithms but prefer self-explanatory code

## Project Structure
- `fm_synth.py` - Core FM synthesis engine with operators and effects
- `music_theory.py` - Music theory algorithms (notes, scales, chords)
- `citypop_generator.py` - City pop style music generator
- `main.py` - Entry point with command-line interface
- `output/` - Directory for generated audio files