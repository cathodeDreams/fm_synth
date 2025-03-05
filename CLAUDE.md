# CLAUDE.md - FM Synth Project

## Commands

### Run Commands
- Main project: `python main.py`
- Just FM synth example: `python fm_synth.py`
- Test single feature: `python -c "from citypop_generator import generate_and_play_track; generate_and_play_track()"`

## Code Style Guidelines

- **Imports**: Standard library first, then third-party, then local modules
- **Formatting**: Use descriptive variable names, follow PEP 8 conventions
- **Types**: Use type hints for function parameters and return values
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Error Handling**: Use try/except blocks for expected exceptions
- **Documentation**: Docstrings in Google style format for all functions/classes
- **Comments**: Explain complex algorithms but prefer self-explanatory code

The project consists of three main modules:
1. `fm_synth.py` - The core FM synthesis engine
2. `music_theory.py` - Music theory algorithms and data structures
3. `citypop_generator.py` - City pop style music generator using the other modules