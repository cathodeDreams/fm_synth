"""
Ambient House Generator

A procedural music generator that creates ambient house tracks
using the music_theory and fm_synth modules. Extends the core music generator
framework with ambient house specific elements.
"""
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import os

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
from core_music_generator import CoreMusicGenerator, MarkovMelodyGenerator


class AmbientHouseMarkovMelodyGenerator(MarkovMelodyGenerator):
    """Ambient house specific Markov melody generator with sparse patterns and sustained notes."""
    
    def __init__(self):
        """Initialize the ambient house Markov melody generator."""
        super().__init__()
        
        # Add ambient house specific melodic motifs
        self.ambient_motifs = AmbientHouseGenerator.AMBIENT_MOTIFS
        self.drone_patterns = AmbientHouseGenerator.DRONE_PATTERNS
    
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
        Generate a melodic phrase using enhanced Markov transitions with ambient house flavor.
        
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
        
        # Determine if we should use a motif
        if use_motif and length >= 5:
            # Higher chance to use ambient motifs for ambient house
            if random.random() < 0.7:  # 70% chance for ambient patterns
                motif = random.choice(self.ambient_motifs)
            else:
                motif = random.choice(self.motifs)
            
            # Adjust the length
            if len(motif) > length:
                # Truncate the motif
                degrees = motif[:length]
            else:
                # Extend the motif using Markov chain
                degrees = motif.copy()
                current = degrees[-1]
                
                while len(degrees) < length:
                    # Determine if this is a strong or weak beat
                    is_strong_beat = rhythm_pattern[len(degrees)] == 1
                    
                    # Use appropriate transitions
                    transitions = self.strong_beat_transitions if is_strong_beat else self.weak_beat_transitions
                    
                    # Ambient house uses more sustained notes and fewer transitions
                    # So we sometimes repeat the current note
                    if random.random() < 0.4:  # 40% chance to sustain
                        next_degree = current
                    else:
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
        else:
            # Use the parent class implementation for regular Markov generation
            return super().generate_phrase(
                scale, chord, start_degree, length, rhythm_pattern, 
                use_motif, end_on_chord_tone, octave
            )
        
        # Convert degrees to actual notes, with octave adjustment
        notes = []
        for degree in degrees:
            note = scale.get_degree(degree)
            if octave > 0:
                note = note.transpose(12 * octave)
            notes.append(note)
                
        return notes

class AmbientHouseGenerator(CoreMusicGenerator):
    """Generator for procedural ambient house tracks."""
    
    # Ambient house specific melodic patterns and motifs
    AMBIENT_MOTIFS = [
        [1, 1, 5, 5, 1],                # Sustained repeating notes
        [3, 3, 5, 5, 3, 3],             # Minimal movement pattern
        [1, 3, 5, 3, 1],                # Simple arpeggiated motif
        [5, 5, 6, 6, 5, 5],             # Oscillating higher motif
        [1, 1, 7, 7, 1, 1],             # Root-seventh oscillation
        [5, 3, 5, 3, 5, 3],             # Hypnotic alternating motif
    ]
    
    # Drone-like sustained patterns for pads
    DRONE_PATTERNS = [
        [1, 1, 1, 1, 5, 5, 5, 5],       # Root and fifth drones
        [3, 3, 3, 3, 7, 7, 7, 7],       # Third and seventh drones
        [1, 1, 1, 1, 1, 1, 1, 1],       # Sustained root
        [5, 5, 5, 5, 6, 6, 6, 6],       # Fifth to sixth movement
    ]
    
    # Custom FM synth instruments for ambient house
    @staticmethod
    def create_pad_synth() -> Instrument:
        """Create an atmospheric pad FM instrument."""
        # Main pad carrier with slow attack
        op1 = FMOperatorWithFeedback(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=1.5,
            amplitude=1.0,
            envelope=ADSREnvelope(1.5, 0.8, 0.7, 2.0),  # Very slow attack and release
            feedback_amount=0.05  # Subtle feedback for warmth
        )
        
        # Detuned component for width
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.003,  # Slightly detuned
            modulation_index=0.8,
            amplitude=0.6,
            envelope=ADSREnvelope(1.2, 0.9, 0.8, 1.8)  # Similarly slow envelope
        )
        
        # Higher harmonic component for texture
        op3 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.TRIANGLE,
            c_to_m_ratio=2.0,  # Octave up
            modulation_index=0.4,
            amplitude=0.4,
            envelope=ADSREnvelope(2.0, 1.0, 0.5, 2.5)  # Even slower attack for evolving texture
        )
        
        voice = FMVoice(
            [(op1, 0.5), (op2, 0.3), (op3, 0.2)],
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(voice)
    
    @staticmethod
    def create_bass_synth() -> Instrument:
        """Create a deep sub bass FM instrument for ambient house."""
        # Main sub bass
        op1 = FMOperatorWithFeedback(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=0.8,  # Subtle modulation for cleaner sub
            amplitude=1.0,
            envelope=ADSREnvelope(0.1, 0.2, 0.8, 0.5),  # Fairly short attack, good sustain
            feedback_amount=0.05  # Minimal feedback for clean sub
        )
        
        # Sub oscillator for thickness
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=0.5,  # Half frequency (octave down)
            modulation_index=0.3,
            amplitude=0.5,
            envelope=ADSREnvelope(0.15, 0.3, 0.7, 0.6)
        )
        
        # Minimal upper harmonics for definition
        op3 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=2.0,  # Octave up
            modulation_index=0.2,
            amplitude=0.15,  # Very subtle
            envelope=ADSREnvelope(0.2, 0.4, 0.3, 0.7)
        )
        
        voice = FMVoice(
            [(op1, 0.7), (op2, 0.25), (op3, 0.05)],
            routing_type=OperatorRoutingType.PARALLEL  # Parallel for cleaner sound
        )
        
        return Instrument(voice)
    
    @staticmethod
    def create_pluck_synth() -> Instrument:
        """Create a pluck synth FM instrument for ambient accents."""
        # Pluck body
        op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.TRIANGLE,
            c_to_m_ratio=1.0,
            modulation_index=3.0,  # Higher modulation for initial attack
            amplitude=1.0,
            envelope=ADSREnvelope(0.01, 0.3, 0.0, 0.4)  # Fast attack, no sustain
        )
        
        # Higher harmonics
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=2.0,  # Octave up
            modulation_index=1.5,
            amplitude=0.6,
            envelope=ADSREnvelope(0.01, 0.2, 0.0, 0.3)  # Fast attack, no sustain
        )
        
        voice = FMVoice(
            [(op1, 0.7), (op2, 0.3)],
            routing_type=OperatorRoutingType.STACKED
        )
        
        return Instrument(voice)
        
    @staticmethod
    def create_atmospheric_synth() -> Instrument:
        """Create an atmospheric texture synth for background ambience."""
        # Slow evolving main carrier
        op1 = FMOperatorWithFeedback(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=1.0,
            amplitude=1.0,
            envelope=ADSREnvelope(2.0, 1.5, 0.7, 3.0),  # Very slow evolution
            feedback_amount=0.1  # Some feedback for complexity
        )
        
        # Detuned layer
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.002,  # Micro-detuning
            modulation_index=0.7,
            amplitude=0.7,
            envelope=ADSREnvelope(2.5, 2.0, 0.6, 3.5)  # Even slower
        )
        
        # High frequency shimmer
        op3 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=4.0,  # 2 octaves up
            modulation_index=0.4,
            amplitude=0.3,
            envelope=ADSREnvelope(3.0, 2.0, 0.5, 4.0)  # Very slow attack
        )
        
        voice = FMVoice(
            [(op1, 0.5), (op2, 0.3), (op3, 0.2)],
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(voice)
    
    @staticmethod
    def _create_kick_drum() -> Instrument:
        """Create a kick drum using FM synthesis for house music."""
        # Kick drum body with longer decay for house music
        kick_op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=4.0,  # Higher modulation for initial click
            amplitude=1.0,
            envelope=ADSREnvelope(0.001, 0.2, 0.0, 0.3)  # Slower decay than typical
        )
        
        # Sub-bass for body
        kick_op2 = FMOperator(
            carrier_type=WaveformType.SINE, 
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=0.5,
            modulation_index=1.0,
            amplitude=0.9,
            envelope=ADSREnvelope(0.005, 0.3, 0.0, 0.4)  # Longer decay for deep house kick
        )
        
        # Create voice for kick
        kick_voice = FMVoice(
            [(kick_op1, 0.5), (kick_op2, 0.5)],  # Equal mix for deep kick
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(kick_voice)
    
    @staticmethod
    def _create_snare_drum() -> Instrument:
        """Create a snare drum using FM synthesis."""
        # Main snare tone - softer for ambient house
        snare_op1 = FMOperator(
            carrier_type=WaveformType.SQUARE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=1.0,
            modulation_index=2.0,  # Less aggressive modulation
            amplitude=0.8, 
            envelope=ADSREnvelope(0.001, 0.15, 0.0, 0.2)  # Slightly longer decay
        )
        
        # Noise component for snare rattle
        snare_op2 = FMOperator(
            carrier_type=WaveformType.NOISE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=1.0,
            modulation_index=0.4,
            amplitude=0.6,
            envelope=ADSREnvelope(0.001, 0.2, 0.1, 0.3)  # Longer decay for ambient feel
        )
        
        # Create voice for snare
        snare_voice = FMVoice(
            [(snare_op1, 0.4), (snare_op2, 0.6)],  # More noise component for washy sound
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(snare_voice)
    
    @staticmethod
    def _create_hihat() -> Instrument:
        """Create a hi-hat using FM synthesis."""
        # Hi-hat with softer character for ambient
        hihat_op1 = FMOperator(
            carrier_type=WaveformType.SQUARE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=5.0,  # High ratio for metallic character
            modulation_index=3.0,  # Less aggressive than typical
            amplitude=0.7,
            envelope=ADSREnvelope(0.001, 0.08, 0.0, 0.12)  # Slightly longer for softer attack
        )
        
        # Additional noise for texture
        hihat_op2 = FMOperator(
            carrier_type=WaveformType.NOISE,
            modulator_type=WaveformType.SINE, 
            c_to_m_ratio=8.0,
            modulation_index=1.5,
            amplitude=0.5,
            envelope=ADSREnvelope(0.001, 0.07, 0.0, 0.1)
        )
        
        # Create voice for hihat
        hihat_voice = FMVoice(
            [(hihat_op1, 0.6), (hihat_op2, 0.4)],  # More balanced mix for softer sound
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(hihat_voice)
    
    @staticmethod
    def _create_open_hihat() -> Instrument:
        """Create an open hi-hat using FM synthesis."""
        # Similar to closed hi-hat but with longer decay
        open_hihat_op1 = FMOperator(
            carrier_type=WaveformType.SQUARE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=5.0,
            modulation_index=3.0,
            amplitude=0.7,
            envelope=ADSREnvelope(0.001, 0.5, 0.1, 0.7)  # Much longer decay and release
        )
        
        # Additional noise for texture
        open_hihat_op2 = FMOperator(
            carrier_type=WaveformType.NOISE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=8.0,
            modulation_index=1.5,
            amplitude=0.5,
            envelope=ADSREnvelope(0.001, 0.4, 0.1, 0.6)
        )
        
        # Create voice for open hihat
        open_hihat_voice = FMVoice(
            [(open_hihat_op1, 0.6), (open_hihat_op2, 0.4)],
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(open_hihat_voice)
    
    @staticmethod
    def _create_clap() -> Instrument:
        """Create a clap sound using FM synthesis."""
        # Main clap body
        clap_op1 = FMOperator(
            carrier_type=WaveformType.NOISE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=1.0,
            modulation_index=1.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.001, 0.2, 0.0, 0.3)  # Fast attack, medium decay
        )
        
        # Secondary resonance
        clap_op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=3.0,  # Higher frequency component
            modulation_index=2.0,
            amplitude=0.5,
            envelope=ADSREnvelope(0.001, 0.15, 0.0, 0.2)  # Shorter decay than main
        )
        
        # Create voice for clap
        clap_voice = FMVoice(
            [(clap_op1, 0.7), (clap_op2, 0.3)],
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(clap_voice)
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize the ambient house generator."""
        # Initialize parent class
        super().__init__(sample_rate)
        
        # Create instruments
        self.pad_synth = self.create_pad_synth()
        self.bass_synth = self.create_bass_synth()
        self.pluck_synth = self.create_pluck_synth()
        self.atmospheric_synth = self.create_atmospheric_synth()
        
        # Create drum instruments
        self.kick = self._create_kick_drum()
        self.snare = self._create_snare_drum()
        self.hihat = self._create_hihat()
        self.open_hihat = self._create_open_hihat()
        self.clap = self._create_clap()
        
        # Initialize the ambient house specific MarkovMelodyGenerator
        self.markov_generator = AmbientHouseMarkovMelodyGenerator()
        
    def add_ambient_chord_extensions(self, chord, context='default'):
        """
        Add ambient house style extensions to a chord based on context.
        
        Args:
            chord: The original chord
            context: The musical context ('pad', 'bass', 'arp', etc.)
            
        Returns:
            A new chord with extensions added
        """
        # Ambient house often uses suspended chords, 9ths, and 11ths
        chord_type = chord.chord_type
        root = chord.root
        
        # Create a new list of notes based on the original chord
        notes = chord.notes.copy()
        
        # Different extensions for different contexts and chord types
        
        # For major chords - add 9ths, sus4, or maj7
        if chord_type == ChordType.MAJOR.value:
            if context == 'pad':
                # More colorful extensions for pads
                extension_options = [
                    'add9',      # Just add 9th
                    'sus4',      # Suspended 4th replacing 3rd
                    'maj7_9',    # Major 7th with 9th
                    'add11'      # Add 11th
                ]
                weights = [0.3, 0.2, 0.3, 0.2]
            else:
                # Simpler extensions in other contexts
                extension_options = [
                    'add9',      # Just add 9th
                    'maj7',      # Add major 7th
                    'sus4',      # Suspended 4th replacing 3rd
                    'none'       # No extensions
                ]
                weights = [0.3, 0.3, 0.2, 0.2]
            
            extension = random.choices(extension_options, weights=weights, k=1)[0]
            
            if extension == 'add9':
                ninth = root.transpose(14)  # 9th = root + 14 semitones
                notes.append(ninth)
            elif extension == 'sus4':
                # Find the third and replace it with a fourth
                third_idx = None
                for i, note in enumerate(notes):
                    if note.note_name == root.transpose(4).note_name:  # Major third is 4 semitones up
                        third_idx = i
                        break
                
                if third_idx is not None:
                    notes[third_idx] = root.transpose(5)  # Perfect 4th is 5 semitones up
                else:
                    # If no third found, just add the fourth
                    notes.append(root.transpose(5))
            elif extension == 'maj7':
                seventh = root.transpose(11)  # Major 7th
                notes.append(seventh)
            elif extension == 'maj7_9':
                seventh = root.transpose(11)  # Major 7th
                ninth = root.transpose(14)    # 9th
                notes.extend([seventh, ninth])
            elif extension == 'add11':
                eleventh = root.transpose(17)  # 11th
                notes.append(eleventh)
                
        # For minor chords - add 9ths, 11ths, or min7
        elif chord_type == ChordType.MINOR.value:
            if context == 'pad':
                # More extensions for pads
                extension_options = [
                    'min9',      # Minor 9
                    'min11',     # Minor 11
                    'min7_9',    # Minor 7/9
                    'none'       # No extension
                ]
                weights = [0.3, 0.3, 0.3, 0.1]
            else:
                # Simpler extensions elsewhere
                extension_options = [
                    'min9',      # Minor 9
                    'min7',      # Minor 7
                    'none'       # No extension
                ]
                weights = [0.3, 0.3, 0.4]
            
            extension = random.choices(extension_options, weights=weights, k=1)[0]
            
            if extension == 'min9':
                ninth = root.transpose(14)  # 9th
                notes.append(ninth)
            elif extension == 'min11':
                eleventh = root.transpose(17)  # 11th
                notes.append(eleventh)
            elif extension == 'min7':
                seventh = root.transpose(10)  # Minor 7th
                notes.append(seventh)
            elif extension == 'min7_9':
                seventh = root.transpose(10)  # Minor 7th
                ninth = root.transpose(14)    # 9th
                notes.extend([seventh, ninth])
        
        # For dominant7 chords, add 9ths or suspended variants
        elif chord_type == ChordType.DOMINANT7.value:
            extension_options = [
                '9',          # Dominant 9
                'sus4',       # Suspended 4th
                '13',         # Dominant 13
                'none'        # No extension
            ]
            weights = [0.4, 0.2, 0.2, 0.2]
            
            extension = random.choices(extension_options, weights=weights, k=1)[0]
            
            if extension == '9':
                ninth = root.transpose(14)  # 9th
                notes.append(ninth)
            elif extension == 'sus4':
                # Find the third and replace it with a fourth
                third_idx = None
                for i, note in enumerate(notes):
                    if note.note_name == root.transpose(4).note_name:  # Major third is 4 semitones up
                        third_idx = i
                        break
                
                if third_idx is not None:
                    notes[third_idx] = root.transpose(5)  # Perfect 4th is 5 semitones up
                else:
                    # If no third found, just add the fourth
                    notes.append(root.transpose(5))
            elif extension == '13':
                thirteenth = root.transpose(21)  # 13th
                notes.append(thirteenth)
        
        # Create a new chord object with the same properties as the original
        custom_chord = Chord(root, chord_type)
        
        # Store the extended notes for reference
        custom_chord.extended_notes = notes
        
        return custom_chord
        
    def generate_track(
        self, 
        root_note: str = 'C', 
        octave: int = 4, 
        tempo: int = 120
    ) -> Sequencer:
        """
        Generate a complete ambient house track with lush atmospheric textures.
        
        Args:
            root_note: Root note of the track
            octave: Octave of the root note
            tempo: Tempo in BPM
            
        Returns:
            A sequencer with the complete track
        """
        print(f"Generating ambient house track in {root_note} with tempo {tempo} BPM...")
        
        # Start with a clean sequencer
        self.sequencer.clear()
        
        # Convert tempo to seconds per beat
        seconds_per_beat = 60 / tempo
        
        # Initialize harmonic structure
        root = Note(root_note, octave)
        
        # Choose a scale - Ambient house often uses minor, dorian, or lydian
        scale_type = random.choice([
            ScaleType.NATURAL_MINOR, 
            ScaleType.DORIAN,
            ScaleType.LYDIAN
        ])
        
        # If we're using Dorian, we need to adjust the root down a step 
        # to maintain the same key signature
        if scale_type == ScaleType.DORIAN:
            root = root.transpose(-2)  # Down a whole step
        
        scale = Scale(root, scale_type)
        print(f"Using {root.note_name} {scale_type.value} scale")
        
        # Choose a chord progression - ambient house uses longer, sparser progressions
        # Often with 2 bars per chord
        progression_degrees = random.choice([
            [1, 6, 4, 5],      # Common house progression
            [1, 4, 1, 5],      # Simpler progression with repetition
            [6, 4, 1, 5],      # Minor-focused progression
            [1, 3, 4, 6],      # Moody progression
            [1, 1, 6, 5]       # Progression with sustained first chord
        ])
        
        # Apply chord extensions
        chords = []
        for degree in progression_degrees:
            # Choose a random chord extension from the available options for this degree
            chord_type = random.choice(self.CHORD_EXTENSIONS[degree])
            chord = Chord.from_scale_degree(scale, degree, chord_type)
            # Use ambient tasteful extensions
            chord = self.add_ambient_chord_extensions(chord, context='pad')
            chords.append(chord)
        
        # Print chord progression
        chord_names = [chord.symbol for chord in chords]
        print(f"Chord progression: {' | '.join(chord_names)}")
        
        # Calculate section durations
        bar_duration = 4 * seconds_per_beat  # 4 beats per bar
        
        # In ambient house, each chord often lasts 2 bars for a more spacious feel
        progression_duration = len(chords) * 2 * bar_duration
        
        # Generate form (song structure)
        form = self._generate_ambient_house_form(chords)
        
        # Generate and add each track with ambient house specific algorithms
        self.track_indices['pad_synth'] = 0  # Track index for pads (first track)
        self._generate_pad_layer(chords, tempo, seconds_per_beat, form)
        
        self.track_indices['bass_synth'] = 1  # Track index for bass
        self._generate_bass_line(chords, tempo, seconds_per_beat, form)
        
        self.track_indices['drums'] = 2  # Track index for drums
        self._generate_drum_pattern(tempo, seconds_per_beat, form)
        
        self.track_indices['pluck_synth'] = 3  # Track index for plucks
        self._generate_pluck_sequence(scale, chords, tempo, seconds_per_beat, form)
        
        self.track_indices['atmospheric_synth'] = 4  # Track index for atmospheric textures
        self._generate_atmospheric_textures(scale, chords, tempo, seconds_per_beat, form)
        
        # Apply effects to each track
        self._apply_track_effects(tempo)
        
        return self.sequencer
        
    def _generate_ambient_house_form(self, chords: List[Chord]) -> Dict[str, List[int]]:
        """
        Generate the song form for ambient house (intro, main, breakdown, etc.).
        
        Args:
            chords: List of chords in the progression
            
        Returns:
            Dictionary mapping section names to bar indices
        """
        print("Generating ambient house song structure...")
        
        # Ambient house typically has longer sections with gradual transitions
        # Each chord lasts 2 bars for more spacious feel
        progression_length = len(chords) * 2
        
        # Create an ambient house structure:
        # Intro: 2 progressions (atmospheric build)
        # Main 1: 2 progressions (with beat)
        # Breakdown: 1 progression (ambient, beatless)
        # Main 2: 2-3 progressions (fuller)
        # Outro: 1-2 progressions (gradual fade)
        
        # Calculate bar indices for each section
        current_bar = 0
        form = {}
        
        # Intro section
        intro_bars = progression_length * 2
        form['intro'] = list(range(current_bar, current_bar + intro_bars))
        current_bar += intro_bars
        
        # Main section 1
        main1_bars = progression_length * 2
        form['main_1'] = list(range(current_bar, current_bar + main1_bars))
        current_bar += main1_bars
        
        # Breakdown section
        breakdown_bars = progression_length
        form['breakdown'] = list(range(current_bar, current_bar + breakdown_bars))
        current_bar += breakdown_bars
        
        # Main section 2
        main2_bars = progression_length * random.choice([2, 3])  # Either 2 or 3 progressions
        form['main_2'] = list(range(current_bar, current_bar + main2_bars))
        current_bar += main2_bars
        
        # Outro
        outro_bars = progression_length * random.choice([1, 2])  # Either 1 or 2 progressions
        form['outro'] = list(range(current_bar, current_bar + outro_bars))
        
        # Print the form
        print(f"Song form: Intro ({len(form['intro'])} bars) → "
              f"Main 1 ({len(form['main_1'])} bars) → "
              f"Breakdown ({len(form['breakdown'])} bars) → "
              f"Main 2 ({len(form['main_2'])} bars) → "
              f"Outro ({len(form['outro'])} bars)")
        
        return form
    
    def _generate_drum_pattern(
        self, 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]]
    ) -> None:
        """
        Generate an ambient house drum pattern.
        
        Args:
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating ambient house drum pattern...")
        
        # Calculate durations
        bar_duration = 4 * seconds_per_beat
        
        # Total number of bars
        total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
        
        # Define drum patterns (16th notes per bar)
        # Ambient house has minimal, spacious drum patterns
        
        # Different patterns for different sections
        kick_patterns = {
            'intro': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Just a kick on beat 1 (sparse)
            'main_1': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Kick on 1 and 3
            'breakdown': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No drums in breakdown
            'main_2': [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # More active pattern
            'outro': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sparse again
        }
        
        # Clap/snare patterns on beats 2 and 4
        clap_patterns = {
            'intro': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No claps in intro
            'main_1': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Classic house pattern
            'breakdown': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No drums in breakdown
            'main_2': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Same clap pattern
            'outro': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Just one clap as fadeout
        }
        
        # Closed hihat patterns
        hihat_patterns = {
            'intro': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No hats in intro
            'main_1': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # Off-beat pattern
            'breakdown': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No drums in breakdown
            'main_2': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # Same pattern
            'outro': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Just one hat as fadeout
        }
        
        # Open hihat patterns
        open_hihat_patterns = {
            'intro': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No open hats in intro
            'main_1': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Just before 3
            'breakdown': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No drums in breakdown
            'main_2': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # Twice per bar
            'outro': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No open hats in outro
        }
        
        # Subtle fill patterns for transitions
        fill_kick = [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
        fill_clap = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
        
        # For each bar, determine which section it belongs to
        for bar in range(total_bars):
            # Determine which section this bar belongs to
            current_section = None
            for section, bar_indices in form.items():
                if bar in bar_indices:
                    current_section = section
                    break
            
            # Skip if we couldn't determine the section (shouldn't happen)
            if current_section is None:
                continue
            
            # Determine if this is a fill bar (last bar of a section except outro)
            is_fill_bar = False
            for section, bar_indices in form.items():
                if bar == bar_indices[-1] and section != 'outro' and section != 'breakdown':
                    is_fill_bar = True
                    break
                    
            # Skip certain bars for variety - ambient house doesn't have drums on every bar
            # Add randomization to make the pattern more human and interesting
            if current_section != 'breakdown':  # Always skip drums in breakdown
                # Determine a variation factor
                if current_section == 'intro':
                    # Sparse drums in intro - many bars have no drums
                    if random.random() < 0.7 and not is_fill_bar:  # 70% chance to skip drums
                        continue
                elif current_section == 'outro':
                    # Similarly sparse in outro
                    if random.random() < 0.6 and not is_fill_bar:  # 60% chance to skip drums
                        continue
                elif current_section == 'main_1' or current_section == 'main_2':
                    # Occasional variety in main sections
                    if random.random() < 0.1 and not is_fill_bar:  # 10% chance to skip drums
                        continue
            
            # Start time for this bar
            start_time = bar * bar_duration
            
            # Choose appropriate patterns based on section
            if current_section == 'breakdown':
                # No drums in breakdown
                continue
                
            kick_pattern = fill_kick if is_fill_bar else kick_patterns[current_section]
            clap_pattern = fill_clap if is_fill_bar else clap_patterns[current_section]
            hihat_pattern = hihat_patterns[current_section]
            open_hihat_pattern = open_hihat_patterns[current_section]
            
            # Add drum hits for this bar
            for i in range(16):  # 16 divisions per bar (16th notes)
                note_time = start_time + (i / 16) * bar_duration
                
                # Add kick drum notes
                if kick_pattern[i] > 0:
                    # Add slight humanization to timing
                    humanized_time = note_time + random.uniform(-0.01, 0.01)
                    self.sequencer.add_note(
                        self.kick,
                        "C1",  # Low note for deep kick
                        humanized_time,
                        0.2,  # Longer duration for ambient house kick
                        kick_pattern[i] * 0.6  # Softer volume
                    )
                
                # Add clap/snare notes
                if clap_pattern[i] > 0:
                    # Add subtle humanization
                    humanized_time = note_time + random.uniform(-0.01, 0.01)
                    self.sequencer.add_note(
                        self.clap,  # Use clap instead of snare for house
                        "D2",
                        humanized_time,
                        0.15,
                        clap_pattern[i] * 0.5  # Softer for ambient feel
                    )
                
                # Add hihat notes
                if hihat_pattern[i] > 0:
                    # Add subtle humanization
                    humanized_time = note_time + random.uniform(-0.01, 0.01)
                    self.sequencer.add_note(
                        self.hihat,
                        "F#3",
                        humanized_time,
                        0.08,  # Short duration
                        hihat_pattern[i] * 0.3  # Quite soft for ambient feel
                    )
                
                # Add open hihat notes
                if open_hihat_pattern[i] > 0:
                    # Add subtle humanization
                    humanized_time = note_time + random.uniform(-0.01, 0.01)
                    self.sequencer.add_note(
                        self.open_hihat,
                        "G#3",
                        humanized_time,
                        0.3,  # Longer duration for open hihat
                        open_hihat_pattern[i] * 0.3  # Soft
                    )
    
    def _generate_pad_layer(
        self, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate atmospheric pad chords.
        
        Args:
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating atmospheric pad layer...")
        
        # In ambient house, pads are a central element with very long sustain
        
        bar_duration = 4 * seconds_per_beat
        
        # Calculate total bars if form is provided
        if form:
            total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
        else:
            # Default to one chord per bar if no form is provided
            total_bars = len(chords) * 2  # Each chord lasts 2 bars
        
        # Each chord lasts 2 bars in ambient house for more space
        harmony_duration = 2 * bar_duration
        
        # Ambient house uses evolving pads with long attack and release
        # Determine section-specific settings
        for bar in range(0, total_bars, 2):  # Step by 2 bars
            # Determine which section this 2-bar group belongs to
            current_section = None
            if form:
                for section, bar_indices in form.items():
                    if bar in bar_indices and bar + 1 in bar_indices:  # Check both bars
                        current_section = section
                        break
            
            # Use default if no form provided or section not determined
            if current_section is None:
                current_section = 'main_1'
                
            # Get the chord for this 2-bar segment
            chord_idx = (bar // 2) % len(chords)
            chord = chords[chord_idx]
            
            # Start time for these 2 bars
            start_time = bar * bar_duration
            
            # Apply specific pad characteristics per section
            if current_section == 'intro':
                # Intro pads are sparse and introduce the atmosphere
                # Fade in gradually
                chord_duration = harmony_duration * 1.2  # Slight overlap
                attack_time = 2.0  # Very slow attack
                volume = 0.4  # Medium volume
            elif current_section == 'breakdown':
                # Breakdown pads are prominent and lush
                chord_duration = harmony_duration * 1.5  # More overlap
                attack_time = 1.5  # Medium-slow attack
                volume = 0.5  # Louder
            elif current_section == 'outro':
                # Outro pads sustain longer for fade-out
                chord_duration = harmony_duration * 1.7  # Even more overlap
                attack_time = 1.8  # Slow attack
                volume = 0.4  # Medium volume
            else:  # main_1 or main_2
                # Main section pads are full but don't overpower the beat
                chord_duration = harmony_duration * 1.3  # Standard overlap
                attack_time = 1.2  # Medium attack
                volume = 0.4  # Medium volume
                
            # Add tasteful extensions for pads
            extended_chord = self.add_ambient_chord_extensions(chord, context='pad')
            
            # Extract chord notes
            chord_notes = extended_chord.extended_notes if hasattr(extended_chord, 'extended_notes') else extended_chord.notes
            
            # Use different voicings for the pad
            # For ambient house, spread the chord across octaves for atmospheric effect
            voicing = []
            
            for i, note in enumerate(chord_notes):
                # Create wide, spread voicings
                if i == 0:  # Root note
                    voicing.append(note.transpose(-12))  # Root down an octave
                elif i == len(chord_notes) - 1:  # Highest note
                    voicing.append(note.transpose(12))  # Highest note up an octave
                else:
                    # Keep middle notes as is
                    voicing.append(note)
            
            # Add each note of the chord to create the pad
            for note in voicing:
                # Add slight timing variation for more natural feel
                note_start = start_time + random.uniform(-0.1, 0.1)
                
                # Create envelope automation
                # Adjust the instrument's envelope directly for this note
                envelope = ADSREnvelope(
                    attack=attack_time,
                    decay=0.8,
                    sustain=0.7,
                    release=3.0  # Long release for pad
                )
                
                # Override the envelope for this specific note
                self.pad_synth.voice.operators[0][0].envelope = envelope
                
                self.sequencer.add_note(
                    self.pad_synth,
                    note,
                    note_start,
                    chord_duration,
                    volume * random.uniform(0.9, 1.0)  # Slight volume variation
                )
    
    def _generate_bass_line(
        self, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate a deep sub bass line for ambient house.
        
        Args:
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating sub bass line...")
        
        bar_duration = 4 * seconds_per_beat
        
        # Calculate total bars if form is provided
        if form:
            total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
        else:
            # Default to one chord per bar if no form is provided
            total_bars = len(chords) * 2  # Each chord lasts 2 bars
            
        # Bass patterns for different sections - 16th note patterns (16 positions per bar)
        # Ambient house uses minimal, sustained bass
        basic_patterns = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Just on the 1
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # On 1 and 3
        ]
        
        # More movement patterns for main sections
        active_patterns = [
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # Pattern with 3 hits
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Syncopated pattern
        ]
        
        # For each bar, generate the bass line
        for bar in range(total_bars):
            # Determine which section this bar belongs to
            current_section = None
            if form:
                for section, bar_indices in form.items():
                    if bar in bar_indices:
                        current_section = section
                        break
            
            # Skip bass in some sections
            if current_section == 'breakdown':
                # Often no bass in breakdown
                if random.random() < 0.7:  # 70% chance to skip bass
                    continue
            elif current_section == 'intro':
                # Sparse bass in intro
                if bar < form['intro'][len(form['intro'])//2]:  # No bass in first half of intro
                    continue
                if random.random() < 0.5:  # 50% chance to skip bass in second half
                    continue
            elif current_section == 'outro':
                # Fading bass in outro
                if bar > form['outro'][len(form['outro'])//2]:  # No bass in second half of outro
                    continue
            
            # Get the chord for this bar
            chord_idx = (bar // 2) % len(chords)  # Each chord lasts 2 bars
            chord = chords[chord_idx]
            
            # Choose a pattern based on section
            if current_section in ['main_1', 'main_2']:
                pattern = random.choice(active_patterns)
            else:
                pattern = random.choice(basic_patterns)
                
            # Start time for this bar
            start_time = bar * bar_duration
            
            # Get the root note for the bass
            root_note = chord.root.transpose(-24)  # Two octaves down for deep bass
            
            # Occasionally use fifth
            fifth_note = root_note.transpose(7)  # Fifth up from root
            
            # For ambient house, we use long sustained bass notes
            for i, velocity in enumerate(pattern):
                if velocity > 0:
                    note_time = start_time + (i / 16) * bar_duration
                    
                    # Determine note duration - sustained for ambient house
                    if i == 0:  # First beat
                        # Longer note on the downbeat
                        duration = 2.0 * seconds_per_beat  # Half bar duration
                    else:
                        # Shorter for other hits
                        duration = 1.0 * seconds_per_beat
                    
                    # Occasionally use fifth instead of root
                    if random.random() < 0.1 and i > 0:  # 10% chance, but not on first beat
                        bass_note = fifth_note
                    else:
                        bass_note = root_note
                    
                    # Add slight timing and velocity variations for humanization
                    human_time = note_time + random.uniform(-0.01, 0.01)
                    human_velocity = velocity * random.uniform(0.95, 1.05)
                    
                    # Add the note to the sequencer
                    self.sequencer.add_note(
                        self.bass_synth,
                        bass_note,
                        human_time,
                        duration,
                        human_velocity * 0.6  # Reduced volume for sub bass
                    )
    
    def _generate_pluck_sequence(
        self, 
        scale: Scale, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate pluck synth sequences for melodic elements.
        
        Args:
            scale: The scale to use
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating pluck sequence...")
        
        bar_duration = 4 * seconds_per_beat
        
        # Calculate total bars if form is provided
        if form:
            total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
        else:
            total_bars = len(chords) * 2  # Each chord lasts 2 bars
        
        # Ambient house pluck patterns are often repetitive with subtle variations
        # We'll generate different patterns for different sections
        
        # Helper function to generate a pattern for a section
        def generate_section_pattern(section_type, scale, chord):
            # Get chord tones for use in patterns
            chord_tones = chord.notes
            
            # Different pattern characteristics for different sections
            if section_type in ['main_1', 'main_2']:
                # More active patterns in main sections
                pattern_length = random.choice([4, 8, 16])  # In 16th notes
                note_density = 0.3  # Percentage of possible positions that have notes
                use_scale_tones = True  # Use scale tones, not just chord tones
            elif section_type == 'breakdown':
                # More sparse in breakdown
                pattern_length = random.choice([8, 16])
                note_density = 0.2  # Fewer notes
                use_scale_tones = True  # More melodic variety
            else:  # intro, outro
                # Simpler patterns in intro/outro
                pattern_length = random.choice([4, 8])
                note_density = 0.15  # Very sparse
                use_scale_tones = False  # Stick to chord tones for stability
            
            # Generate the actual pattern
            pattern = []
            
            # Determine which positions will have notes
            for i in range(pattern_length):
                # Higher chance of notes on strong beats
                if i % 4 == 0:  # 16th notes, so every 4th is a quarter note
                    note_chance = note_density * 2.0  # Double chance on main beats
                elif i % 2 == 0:  # 8th notes
                    note_chance = note_density * 1.5  # 1.5x chance on 8th notes
                else:
                    note_chance = note_density  # Normal chance on off-beats
                
                # Decide if this position gets a note
                if random.random() < note_chance:
                    # Choose a note
                    if use_scale_tones and random.random() < 0.4:  # 40% chance for scale tone
                        # Use a scale tone
                        scale_note = random.choice(scale.notes)
                        # Move to appropriate octave
                        octave_adjust = random.choice([0, 12])  # Same octave or up one
                        note = scale_note.transpose(octave_adjust)
                    else:
                        # Use a chord tone
                        chord_note = random.choice(chord_tones)
                        # Move to appropriate octave
                        octave_adjust = random.choice([0, 12])  # Same octave or up one
                        note = chord_note.transpose(octave_adjust)
                    
                    # Add to pattern with random velocity for dynamics
                    velocity = random.uniform(0.3, 0.7)
                    pattern.append((i, note, velocity))
                
            return pattern
        
        # Dictionary to store patterns by section
        section_patterns = {}
        
        # Process each bar
        for bar in range(total_bars):
            # Determine which section this bar belongs to
            current_section = None
            if form:
                for section, bar_indices in form.items():
                    if bar in bar_indices:
                        current_section = section
                        break
            
            # Skip if we couldn't determine the section or if it's a no-pluck section
            if current_section is None:
                continue
            
            # Skip plucks in some sections/bars for variety
            if current_section == 'intro':
                # Very sparse in intro
                if bar < form['intro'][len(form['intro'])//2]:  # No plucks in first half of intro
                    continue
                if random.random() < 0.7:  # 70% chance to skip plucks in second half
                    continue
            elif current_section == 'breakdown':
                # Some plucks in breakdown for texture
                if random.random() < 0.5:  # 50% chance to skip
                    continue
            elif current_section == 'outro':
                # Fading plucks in outro
                if bar > form['outro'][len(form['outro'])//2]:  # No plucks in second half of outro
                    continue
                if random.random() < 0.6:  # 60% chance to skip
                    continue
            elif current_section in ['main_1', 'main_2']:
                # More consistent in main sections but still with some variation
                if random.random() < 0.2:  # 20% chance to skip for variety
                    continue
            
            # Get the chord for this bar
            chord_idx = (bar // 2) % len(chords)  # Each chord lasts 2 bars
            chord = chords[chord_idx]
            
            # Start time for this bar
            start_time = bar * bar_duration
            
            # Get or generate a pattern for this section
            if current_section not in section_patterns:
                # First time we've seen this section, generate a new pattern
                section_patterns[current_section] = generate_section_pattern(current_section, scale, chord)
            
            # Get the pattern
            pattern = section_patterns[current_section]
            
            # Apply the pattern to this bar
            for position, note, velocity in pattern:
                # Calculate the actual time
                note_time = start_time + (position / 16) * bar_duration
                
                # Determine duration - short for plucks
                duration = random.uniform(0.1, 0.3) * seconds_per_beat
                
                # Add some humanization
                human_time = note_time + random.uniform(-0.01, 0.01)
                human_velocity = velocity * random.uniform(0.95, 1.05)
                
                # Add the note
                self.sequencer.add_note(
                    self.pluck_synth,
                    note,
                    human_time,
                    duration,
                    human_velocity * 0.4  # Quieter for ambience
                )
            
            # Occasionally add a variation to the pattern
            if random.random() < 0.2:  # 20% chance for variation
                # Simple variation - change one note
                if pattern:  # Make sure pattern is not empty
                    variation_idx = random.randint(0, len(pattern) - 1)
                    pos, old_note, vel = pattern[variation_idx]
                    
                    # Choose a new note nearby
                    chord_note = random.choice(chord.notes)
                    new_note = chord_note.transpose(random.choice([0, 12]))
                    
                    # Replace in pattern
                    pattern[variation_idx] = (pos, new_note, vel)
    
    def _generate_atmospheric_textures(
        self, 
        scale: Scale, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate atmospheric texture layers.
        
        Args:
            scale: The scale to use
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating atmospheric textures...")
        
        bar_duration = 4 * seconds_per_beat
        
        # Calculate total bars if form is provided
        if form:
            total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
        else:
            total_bars = len(chords) * 2  # Each chord lasts 2 bars
        
        # Ambient textures are sparse, evolving sounds
        # We'll add long sustained notes at key points
        
        # Find the beginning of each section for texture transitions
        section_starts = {}
        if form:
            for section, bar_indices in form.items():
                section_starts[section] = min(bar_indices)
        
        # Generate textures at specific points
        texture_points = []
        
        # Add texture at the start of each section
        for section, start_bar in section_starts.items():
            texture_points.append(start_bar)
            
            # For longer sections, add textures at midpoints too
            if section in ['main_1', 'main_2', 'breakdown']:
                # Get section length
                section_length = len(form[section])
                if section_length > 8:  # If long enough
                    midpoint = start_bar + section_length // 2
                    texture_points.append(midpoint)
        
        # Add some random texture points for variety
        num_random_textures = total_bars // 8  # Approximately one per 8 bars
        for _ in range(num_random_textures):
            random_bar = random.randint(0, total_bars - 1)
            texture_points.append(random_bar)
        
        # Remove duplicates and sort
        texture_points = sorted(list(set(texture_points)))
        
        # Generate texture for each point
        for bar in texture_points:
            # Determine which section this bar belongs to
            current_section = None
            if form:
                for section, bar_indices in form.items():
                    if bar in bar_indices:
                        current_section = section
                        break
            
            # Skip if we couldn't determine the section
            if current_section is None:
                continue
            
            # Get the chord for this bar
            chord_idx = (bar // 2) % len(chords)  # Each chord lasts 2 bars
            chord = chords[chord_idx]
            
            # Start time for this bar
            start_time = bar * bar_duration
            
            # Section-specific texture characteristics
            if current_section == 'intro':
                # Evolving textures that introduce the track
                num_notes = random.randint(1, 2)  # Very sparse
                duration_factor = random.uniform(2.0, 3.0)  # Very long
                volume = 0.3  # Medium-low volume
            elif current_section == 'breakdown':
                # More prominent textures during breakdown
                num_notes = random.randint(2, 3)  # More notes
                duration_factor = random.uniform(2.5, 4.0)  # Extremely long
                volume = 0.4  # Medium volume
            elif current_section == 'outro':
                # Fading textures
                num_notes = random.randint(1, 3)  # Variable
                duration_factor = random.uniform(3.0, 4.0)  # Very long for fadeout
                volume = 0.3  # Medium-low volume
            else:  # main_1 or main_2
                # Supportive textures during main sections
                num_notes = random.randint(1, 2)  # Sparse
                duration_factor = random.uniform(1.5, 2.5)  # Long but not overwhelming
                volume = 0.25  # Lower volume to not conflict with other elements
            
            # Generate the notes
            for _ in range(num_notes):
                # Choose a note - prioritize chord tones but sometimes use scale tones
                if random.random() < 0.7:  # 70% chance for chord tone
                    note = random.choice(chord.notes)
                else:
                    note = random.choice(scale.notes)
                
                # Apply octave adjustment - textures often use high or low registers
                octave_adjust = random.choice([-12, 0, 12, 24])  # Two octaves in either direction
                note = note.transpose(octave_adjust)
                
                # Calculate duration - very long for textures
                duration = duration_factor * bar_duration
                
                # Add slight variation
                human_time = start_time + random.uniform(-0.2, 0.2)
                human_velocity = volume * random.uniform(0.9, 1.1)
                
                # Create envelope automation for this specific note
                attack_time = random.uniform(1.5, 3.0)  # Very slow attack
                release_time = random.uniform(2.0, 4.0)  # Very long release
                
                # Override the envelope for this note
                envelope = ADSREnvelope(
                    attack=attack_time,
                    decay=1.0,
                    sustain=0.7,
                    release=release_time
                )
                
                # Apply the envelope
                self.atmospheric_synth.voice.operators[0][0].envelope = envelope
                
                # Add the note
                self.sequencer.add_note(
                    self.atmospheric_synth,
                    note,
                    human_time,
                    duration,
                    human_velocity
                )
    
    def _apply_track_effects(self, tempo: int) -> None:
        """
        Apply ambient house specific effects to each track.
        
        Args:
            tempo: Tempo in BPM (for time-based effects)
        """
        print("Applying ambient house effects to tracks...")
        
        # Calculate delay time based on tempo (usually 1/8th or 1/16th note)
        eighth_note_delay = 60 / tempo / 2
        quarter_note_delay = 60 / tempo
        
        # Apply effects to pad synth
        if self.track_indices['pad_synth'] is not None:
            pad_effects = EffectChain()
            # Add chorus for width - subtle
            pad_effects.add_effect(ChorusEffect(rate=0.3, depth=0.5, mix=0.3))
            # Add reverb for space - large
            pad_effects.add_effect(ReverbEffect(room_size=0.8, damping=0.3, mix=0.7))
            self.sequencer.apply_effects_to_track(self.track_indices['pad_synth'], pad_effects)
        
        # Apply effects to bass synth
        if self.track_indices['bass_synth'] is not None:
            bass_effects = EffectChain()
            # Very subtle chorus and minimal reverb
            bass_effects.add_effect(ChorusEffect(rate=0.2, depth=0.3, mix=0.1))
            bass_effects.add_effect(ReverbEffect(room_size=0.3, damping=0.8, mix=0.1))
            self.sequencer.apply_effects_to_track(self.track_indices['bass_synth'], bass_effects)
        
        # Apply effects to pluck synth
        if self.track_indices['pluck_synth'] is not None:
            pluck_effects = EffectChain()
            # Add delay for rhythmic interest
            pluck_effects.add_effect(DelayEffect(delay_time=eighth_note_delay, feedback=0.5, mix=0.4))
            # Add reverb for space
            pluck_effects.add_effect(ReverbEffect(room_size=0.7, damping=0.4, mix=0.6))
            self.sequencer.apply_effects_to_track(self.track_indices['pluck_synth'], pluck_effects)
        
        # Apply effects to atmospheric textures
        if self.track_indices['atmospheric_synth'] is not None:
            atmos_effects = EffectChain()
            # Heavy chorus for movement
            atmos_effects.add_effect(ChorusEffect(rate=0.4, depth=0.7, mix=0.6))
            # Long delay for evolving textures
            atmos_effects.add_effect(DelayEffect(delay_time=quarter_note_delay*2, feedback=0.4, mix=0.3))
            # Large reverb for space
            atmos_effects.add_effect(ReverbEffect(room_size=0.9, damping=0.2, mix=0.8))
            self.sequencer.apply_effects_to_track(self.track_indices['atmospheric_synth'], atmos_effects)
        
        # Apply effects to drums
        if self.track_indices['drums'] is not None:
            drums_effects = EffectChain()
            # Add reverb for ambient house drum sound
            drums_effects.add_effect(ReverbEffect(room_size=0.6, damping=0.5, mix=0.3))
            self.sequencer.apply_effects_to_track(self.track_indices['drums'], drums_effects)

def generate_and_play_track(
    root_note: str = 'C', 
    octave: int = 4, 
    tempo: int = 120,
    output_file: str = "output/ambient_house_track.wav"
) -> str:
    """
    Generate and save an ambient house track to a WAV file.
    
    Args:
        root_note: Root note of the track
        octave: Octave of the root note
        tempo: Tempo in BPM
        output_file: Path to the output WAV file
        
    Returns:
        The absolute path to the saved WAV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    generator = AmbientHouseGenerator()
    sequencer = generator.generate_track(root_note, octave, tempo)
    
    print(f"Saving the generated track to {output_file}...")
    return sequencer.save_to_wav(output_file)

if __name__ == "__main__":
    # Generate a track in C minor at 120 BPM
    generate_and_play_track('C', 4, 120)