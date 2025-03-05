import argparse
import os
import sys
import datetime
import io
from contextlib import redirect_stdout

# Import generators
from citypop_generator import generate_and_play_track as generate_citypop
from ambient_house_generator import generate_and_play_track as generate_ambient_house

def generate_unique_filename(base_path):
    """Generate a unique filename by adding a timestamp."""
    # Get base directory and filename
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    
    # Add timestamp to filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{name}_{timestamp}{ext}"
    
    return os.path.join(directory, unique_filename)

def main():
    # Capture all output
    captured_output = io.StringIO()
    
    with redirect_stdout(captured_output):
        try:
            # Set up command line argument parsing
            parser = argparse.ArgumentParser(description='Generate procedural music using FM synthesis')
            
            parser.add_argument('--style', type=str, choices=['citypop', 'ambient_house'], default='citypop',
                                help='Music style to generate (citypop or ambient_house)')
            
            parser.add_argument('--root', type=str, default='F',
                                help='Root note for the track (e.g., C, F#, Bb)')
            
            parser.add_argument('--octave', type=int, default=4,
                                help='Octave for the root note (3-5 recommended)')
            
            parser.add_argument('--tempo', type=int,
                                help='Tempo in BPM (if not specified, uses style default)')
            
            parser.add_argument('--output', type=str, 
                                help='Output file path (if not specified, uses style default)')
            
            # Parse arguments
            args = parser.parse_args()
            
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            
            # Set style-specific defaults
            if args.style == 'citypop':
                default_tempo = 88
                default_output = "output/citypop_track.wav"
            else:  # ambient_house
                default_tempo = 120
                default_output = "output/ambient_house_track.wav"
            
            # Use provided args or defaults
            tempo = args.tempo if args.tempo is not None else default_tempo
            base_output_file = args.output if args.output is not None else default_output
            
            # Generate unique filename
            output_file = generate_unique_filename(base_output_file)
            
            # Generate the track based on selected style
            if args.style == 'citypop':
                print(f"Generating city pop track in {args.root} with tempo {tempo} BPM...")
                track_path = generate_citypop(
                    root_note=args.root,
                    octave=args.octave,
                    tempo=tempo,
                    output_file=output_file
                )
            else:  # ambient_house
                print(f"Generating ambient house track in {args.root} with tempo {tempo} BPM...")
                track_path = generate_ambient_house(
                    root_note=args.root,
                    octave=args.octave,
                    tempo=tempo,
                    output_file=output_file
                )
            
            print(f"Track saved to: {track_path}")
            
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    # Mirror output to console
    output_text = captured_output.getvalue()
    print(output_text, end="")
    
    # Write output to log.txt
    with open("log.txt", "w") as log_file:
        log_file.write(output_text)

if __name__ == "__main__":
    main()

