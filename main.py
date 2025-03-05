from citypop_generator import generate_and_play_track
import os

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Generate and save a track to WAV file
output_path = generate_and_play_track(output_file="output/main_track.wav")
print(f"Track saved to: {output_path}")

