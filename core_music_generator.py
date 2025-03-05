"""
Core Music Generator

A base framework for procedural music generation, providing common functionality
that can be extended for specific musical genres.
"""
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

from music_theory import (
    Note, Scale, Chord, ChordType, ScaleType, 
    ChordProgression, Interval, Rhythm, RhythmPattern
)
from fm_synth import (
    Instrument, InstrumentPresets, Sequencer, 
    FMOperator, FMOperatorWithFeedback, FMVoice, 
    WaveformType, ADSREnvelope, OperatorRoutingType,
    EffectChain, DelayEffect, ReverbEffect, ChorusEffect
)


class MarkovMelodyGenerator:
    """Generate melodies using Markov chains with improved chord-tone alignment."""
    
    def __init__(self):
        """Initialize the enhanced Markov melody generator."""
        # Define transition probabilities between scale degrees for strong beats
        # Higher probability for chord tones (1, 3, 5, 7)
        self.strong_beat_transitions = {
            1: {1: 0.15, 2: 0.1, 3: 0.35, 5: 0.3, 7: 0.1},
            2: {1: 0.3, 3: 0.3, 5: 0.3, 7: 0.1},
            3: {1: 0.2, 3: 0.3, 5: 0.4, 7: 0.1},
            4: {3: 0.3, 5: 0.5, 7: 0.2},
            5: {1: 0.2, 3: 0.3, 5: 0.3, 7: 0.2},
            6: {5: 0.3, 7: 0.2, 1: 0.3, 3: 0.2},
            7: {1: 0.6, 3: 0.2, 5: 0.2}
        }
        
        # Transitions for weak beats - more freedom for passing tones
        self.weak_beat_transitions = {
            1: {1: 0.1, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1},
            2: {1: 0.2, 3: 0.3, 4: 0.2, 5: 0.2, 7: 0.1},
            3: {2: 0.2, 4: 0.3, 5: 0.3, 6: 0.1, 7: 0.1},
            4: {3: 0.3, 5: 0.3, 6: 0.2, 7: 0.2},
            5: {1: 0.1, 4: 0.2, 6: 0.3, 7: 0.2, 3: 0.2},
            6: {5: 0.3, 7: 0.3, 2: 0.2, 4: 0.2},
            7: {1: 0.4, 6: 0.3, 5: 0.2, 3: 0.1}
        }
        
        # Resolution patterns that create satisfying phrase endings
        self.resolution_patterns = [
            [2, 1],                          # Simple 2-1 resolution
            [7, 1],                          # Leading tone resolution 
            [6, 5],                          # 6-5 resolution
            [4, 3],                          # 4-3 resolution
            [3, 2, 1],                       # Stepwise descent to tonic
            [2, 7, 1],                       # Chromatic approach to tonic
        ]
    
    def get_chord_degrees(self, chord, scale):
        """
        Identify the scale degrees that match chord tones.
        
        Args:
            chord: The current chord
            scale: The current scale
            
        Returns:
            List of scale degrees that are chord tones
        """
        chord_tones = [note.note_name for note in chord.notes]
        scale_notes = scale.notes
        
        # Find scale degrees that match chord tones
        chord_degrees = []
        for i, scale_note in enumerate(scale_notes):
            if scale_note.note_name in chord_tones:
                chord_degrees.append(i + 1)  # +1 because scale degrees are 1-based
                
        return chord_degrees
    
    def generate_phrase(
        self, 
        scale, 
        chord,
        start_degree=None, 
        length=8, 
        rhythm_pattern=None,
        use_motif=False,
        end_on_chord_tone=True,
        octave=0
    ):
        """
        Generate a melodic phrase using enhanced Markov transitions.
        
        Args:
            scale: The scale to use
            chord: The current chord (for chord-tone alignment)
            start_degree: Starting scale degree (1-7)
            length: Number of notes in the phrase
            rhythm_pattern: List of rhythmic positions (0=weak beat, 1=strong beat)
            use_motif: Whether to use a predefined motif
            end_on_chord_tone: Whether to end the phrase on a chord tone
            octave: Octave adjustment (0 = no adjustment, 1 = up an octave)
            
        Returns:
            List of notes in the phrase
        """
        if rhythm_pattern is None:
            # Default rhythm pattern alternating strong/weak beats
            rhythm_pattern = [1 if i % 2 == 0 else 0 for i in range(length)]
            
        # Get chord tones as scale degrees
        chord_degrees = self.get_chord_degrees(chord, scale)
        
        # If no chord tones found, default to 1, 3, 5
        if not chord_degrees:
            if "minor" in chord.chord_type:
                chord_degrees = [1, 3, 5]  # Treat as minor triad
            else:
                chord_degrees = [1, 3, 5]  # Treat as major triad
        
        # If start_degree is None, choose a good starting note from chord tones
        if start_degree is None:
            start_degree = random.choice(chord_degrees)
        
        # Generate using Markov chain
        degrees = [start_degree]
        current = start_degree
        
        for i in range(length - 1):
            # Determine if this is a strong or weak beat
            is_strong_beat = rhythm_pattern[i+1] == 1
            
            # If we're on the last note and should end on chord tone
            is_last_note = i == length - 2
            
            if is_last_note and end_on_chord_tone:
                # Choose a chord tone for the ending
                if current in [2, 4, 6, 7]:  # If on a non-chord tone, resolve stepwise
                    # Find closest chord tone by step
                    if current == 2:
                        next_degree = 1 if 1 in chord_degrees else 3
                    elif current == 4:
                        next_degree = 3 if 3 in chord_degrees else 5
                    elif current == 6:
                        next_degree = 5 if 5 in chord_degrees else 7
                    else:  # 7
                        next_degree = 1
                else:
                    # Already on a potential chord tone, see if it's in the current chord
                    if current in chord_degrees:
                        next_degree = current  # Stay on current note
                    else:
                        # Move to closest chord tone
                        next_degree = min(chord_degrees, key=lambda x: abs(x - current))
            else:
                # Use appropriate transitions
                transitions = self.strong_beat_transitions if is_strong_beat else self.weak_beat_transitions
                
                # Use regular transitions
                next_candidates = list(transitions[current].keys())
                next_weights = list(transitions[current].values())
                
                # On strong beats, bias toward chord tones for better harmony
                if is_strong_beat:
                    for i, degree in enumerate(next_candidates):
                        if degree in chord_degrees:
                            next_weights[i] *= 1.5  # Increase weight for chord tones
                
                # Normalize weights
                total_weight = sum(next_weights)
                next_weights = [w/total_weight for w in next_weights]
                
                next_degree = random.choices(
                    next_candidates,
                    weights=next_weights,
                    k=1
                )[0]
            
            degrees.append(next_degree)
            current = next_degree
    
        # Convert degrees to actual notes, with octave adjustment
        notes = []
        for degree in degrees:
            note = scale.get_degree(degree)
            if octave > 0:
                note = note.transpose(12 * octave)
            notes.append(note)
                
        return notes
    
    def generate_phrase_for_chord_progression(
        self, 
        scale, 
        chords, 
        phrase_length, 
        rhythm_pattern=None,
        use_motif=True,
        octave=0
    ):
        """
        Generate a phrase that works well across a chord progression.
        
        Args:
            scale: The scale to use
            chords: List of chords in the progression
            phrase_length: Number of notes in the phrase
            rhythm_pattern: List of rhythmic positions (0=weak, 1=strong)
            use_motif: Whether to use a predefined motif
            octave: Octave adjustment
            
        Returns:
            List of notes in the phrase
        """
        # First, determine a good starting degree based on the first chord
        first_chord = chords[0]
        chord_degrees = self.get_chord_degrees(first_chord, scale)
        
        if not chord_degrees:
            # Default to 1, 3, 5 if no chord tones found
            if "minor" in first_chord.chord_type:
                chord_degrees = [1, 3, 5]
            else:
                chord_degrees = [1, 3, 5]
                
        # Prefer the root or third for a strong start
        start_weight = {1: 0.5, 3: 0.3, 5: 0.2}  # Root, third, fifth weights
        available_starts = [d for d in chord_degrees if d in start_weight]
        
        if available_starts:
            weights = [start_weight.get(d, 0.1) for d in available_starts]
            start_degree = random.choices(available_starts, weights=weights, k=1)[0]
        else:
            start_degree = random.choice(chord_degrees)
        
        # Create the phrase
        return self.generate_phrase(
            scale, 
            first_chord,
            start_degree, 
            phrase_length, 
            rhythm_pattern,
            use_motif=use_motif,
            end_on_chord_tone=True,
            octave=octave
        )


class CoreMusicGenerator:
    """Base class for procedural music generation."""
    
    # Common chord progressions as scale degrees
    CHORD_PROGRESSIONS = [
        # Common major key progressions
        [1, 6, 2, 5],     # I-vi-ii-V
        [1, 4, 2, 5],     # I-IV-ii-V  
        [2, 5, 1, 6],     # ii-V-I-vi
        [1, 4, 3, 6],     # I-IV-iii-vi
        [4, 5, 3, 6],     # IV-V-iii-vi
        [1, 6, 4, 5],     # I-vi-IV-V
        [6, 2, 5, 1]      # vi-ii-V-I
    ]
    
    # Common chord extensions
    CHORD_EXTENSIONS = {
        1: [ChordType.MAJOR7, ChordType.MAJOR6, ChordType.ADD9],       # I chord
        2: [ChordType.MINOR7, ChordType.MINOR6],                       # ii chord
        3: [ChordType.MINOR7],                                         # iii chord
        4: [ChordType.MAJOR7, ChordType.ADD9],                         # IV chord
        5: [ChordType.DOMINANT7, ChordType.MAJOR],                     # V chord
        6: [ChordType.MINOR7, ChordType.MINOR],                        # vi chord
        7: [ChordType.HALF_DIMINISHED7]                               # vii chord
    }
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize the base music generator."""
        self.sample_rate = sample_rate
        self.sequencer = Sequencer(sample_rate)
        
        # Track indices for effects
        self.track_indices = {
            'ep_piano': None,
            'synth_bass': None,
            'lead_synth': None,
            'bell': None,
            'saxophone': None,
            'strings': None,
            'drums': None
        }
        
        # Placeholder for instruments - to be defined by the genre-specific implementation
        self.instruments = {}
        
    def add_tasteful_extensions(self, chord, context='default'):
        """
        Add tasteful chord extensions based on context.
        
        Args:
            chord: The original chord
            context: The musical context ('verse', 'chorus', 'bridge', etc.)
            
        Returns:
            A new chord with extensions added
        """
        # Default implementation - subclasses should override
        return chord
    
    def _generate_song_form(self, chords: List[Chord]) -> Dict[str, List[int]]:
        """
        Generate the song form (intro, verse, chorus, etc.).
        
        Args:
            chords: List of chords in the progression
            
        Returns:
            Dictionary mapping section names to bar indices
        """
        print("Generating song form...")
        
        # Determine number of bars in the chord progression
        progression_length = len(chords)
        
        # Default structure:
        # Intro: 1 chord progression
        # Verse: 2 chord progressions
        # Chorus: 2 chord progressions
        # Verse 2: 2 chord progressions
        # Chorus: 2 chord progressions
        # Bridge: 1 chord progression
        # Chorus: 2 chord progressions
        # Outro: 1 chord progression
        
        # Calculate bar indices for each section
        current_bar = 0
        form = {}
        
        # Intro section
        intro_bars = progression_length
        form['intro'] = list(range(current_bar, current_bar + intro_bars))
        current_bar += intro_bars
        
        # Verse 1
        verse_bars = progression_length * 2
        form['verse_1'] = list(range(current_bar, current_bar + verse_bars))
        current_bar += verse_bars
        
        # Chorus 1
        chorus_bars = progression_length * 2
        form['chorus_1'] = list(range(current_bar, current_bar + chorus_bars))
        current_bar += chorus_bars
        
        # Verse 2
        form['verse_2'] = list(range(current_bar, current_bar + verse_bars))
        current_bar += verse_bars
        
        # Chorus 2
        form['chorus_2'] = list(range(current_bar, current_bar + chorus_bars))
        current_bar += chorus_bars
        
        # Bridge
        bridge_bars = progression_length
        form['bridge'] = list(range(current_bar, current_bar + bridge_bars))
        current_bar += bridge_bars
        
        # Final Chorus
        form['chorus_3'] = list(range(current_bar, current_bar + chorus_bars))
        current_bar += chorus_bars
        
        # Outro
        outro_bars = progression_length
        form['outro'] = list(range(current_bar, current_bar + outro_bars))
        
        # Print the form
        print(f"Song form: Intro ({len(form['intro'])} bars) → "
              f"Verse 1 ({len(form['verse_1'])} bars) → "
              f"Chorus 1 ({len(form['chorus_1'])} bars) → "
              f"Verse 2 ({len(form['verse_2'])} bars) → "
              f"Chorus 2 ({len(form['chorus_2'])} bars) → "
              f"Bridge ({len(form['bridge'])} bars) → "
              f"Chorus 3 ({len(form['chorus_3'])} bars) → "
              f"Outro ({len(form['outro'])} bars)")
        
        return form

    def _generate_melody(
        self, 
        scale, 
        chords, 
        tempo, 
        seconds_per_beat,
        form=None
    ):
        """
        Generate a melody that fits the chord progression with improved harmony.
        
        Args:
            scale: The scale to use for melody generation
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating melody with improved chord-tone alignment...")
        
        # Subclasses should override this
        pass
    
    def _apply_section_melody(
        self,
        section_melody,
        scale,
        chords,
        bar_indices,
        seconds_per_beat,
        bar_duration,
        variation=False
    ):
        """
        Apply a section melody to the specific bars.
        
        Args:
            section_melody: List of (note, start_time, duration, velocity) tuples
            scale: The scale to use
            chords: List of chords
            bar_indices: List of bar indices where this melody should be applied
            seconds_per_beat: Duration of one beat in seconds
            bar_duration: Duration of one bar in seconds
            variation: Whether to apply variation to the melody
        """
        # Calculate section offset
        section_start_bar = min(bar_indices)
        section_offset = section_start_bar * bar_duration
        
        # Get section duration
        section_duration = len(bar_indices) * bar_duration
        
        for note, rel_start_time, duration, velocity in section_melody:
            # Apply the note with timing relative to section start
            abs_start_time = section_offset + rel_start_time
            
            # Apply variation if requested
            if variation:
                # Slightly vary timing, duration, velocity
                abs_start_time += random.uniform(-0.05, 0.05) * seconds_per_beat
                duration *= random.uniform(0.9, 1.1)
                velocity *= random.uniform(0.9, 1.1)
                
                # Occasionally substitute with another scale note
                if random.random() < 0.2:  # 20% chance for note variation
                    try:
                        # Use a neighboring scale degree
                        scale_notes = scale.notes
                        
                        # Find the note within the scale (check if same note name exists)
                        # This handles the case where the note may be in a different octave
                        note_name = note.note_name
                        scale_note_names = [n.note_name for n in scale_notes]
                        
                        if note_name in scale_note_names:
                            # Find all matching notes by name
                            matching_indices = [i for i, n in enumerate(scale_notes) if n.note_name == note_name]
                            # Use the first match
                            current_idx = matching_indices[0]
                            
                            # Move up or down by one scale degree
                            variation_idx = current_idx + random.choice([-1, 1])
                            variation_idx = max(0, min(variation_idx, len(scale_notes) - 1))
                            
                            note = scale_notes[variation_idx]
                    except (ValueError, IndexError):
                        # If the note isn't in the scale or any other error occurs,
                        # we keep the original note without variation
                        pass
            
            # Ensure values are within reasonable bounds
            velocity = max(0.2, min(0.9, velocity))
            duration = max(0.1, duration)
            
            # Add note to sequencer - subclasses need to implement lead_synth instrument
            if hasattr(self, 'lead_synth'):
                self.sequencer.add_note(
                    self.lead_synth,
                    note,
                    abs_start_time,
                    duration,
                    velocity
                )
    
    def _apply_track_effects(self, tempo):
        """
        Apply effects to each instrument track for enhanced sound.
        
        Args:
            tempo: Tempo in BPM (for time-based effects)
        """
        print("Applying effects to tracks...")
        
        # Subclasses should override this with specific effect implementations
        pass
        
    def generate_track(
        self, 
        root_note='F', 
        octave=4, 
        tempo=90
    ):
        """
        Generate a complete music track.
        
        Args:
            root_note: Root note of the track
            octave: Octave of the root note
            tempo: Tempo in BPM
            
        Returns:
            A sequencer with the complete track
        """
        print(f"Generating track in {root_note} with tempo {tempo} BPM...")
        
        # Start with a clean sequencer
        self.sequencer.clear()
        
        # Subclasses should implement this method
        pass