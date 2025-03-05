"""
City Pop Generator

A procedural music generator that creates city pop / jazz fusion tracks
using the music_theory and fm_synth modules.
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

class CityPopGenerator:
    """Generator for procedural city pop / jazz fusion tracks."""
    
    # Common city pop chord progressions as scale degrees
    CHORD_PROGRESSIONS = [
        # Common major key progressions with jazz flavor
        [1, 6, 2, 5],     # I-vi-ii-V
        [1, 4, 2, 5],     # I-IV-ii-V  
        [2, 5, 1, 6],     # ii-V-I-vi
        [1, 4, 3, 6],     # I-IV-iii-vi
        [4, 5, 3, 6],     # IV-V-iii-vi
        [1, 6, 4, 5],     # I-vi-IV-V
        [6, 2, 5, 1]      # vi-ii-V-I
    ]
    
    # Common chord extensions for city pop
    CHORD_EXTENSIONS = {
        1: [ChordType.MAJOR7, ChordType.MAJOR6, ChordType.ADD9],       # I chord
        2: [ChordType.MINOR7, ChordType.MINOR6],                       # ii chord
        3: [ChordType.MINOR7],                                         # iii chord
        4: [ChordType.MAJOR7, ChordType.ADD9],                         # IV chord
        5: [ChordType.DOMINANT7, ChordType.MAJOR],                     # V chord
        6: [ChordType.MINOR7, ChordType.MINOR],                        # vi chord
        7: [ChordType.HALF_DIMINISHED7]                               # vii chord
    }
    
    # Custom FM synth instruments for city pop
    @staticmethod
    def create_ep_piano() -> Instrument:
        """Create an electric piano FM instrument with feedback."""
        # EP-like settings with three operators and feedback
        op1 = FMOperatorWithFeedback(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=2.5,
            amplitude=1.0,
            envelope=ADSREnvelope(0.01, 0.2, 0.8, 0.3),
            feedback_amount=0.15  # Add feedback for more complex harmonics
        )
        
        # Tine component
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.TRIANGLE,
            c_to_m_ratio=14.0,  # High ratio for metallic character
            modulation_index=0.4,
            amplitude=0.5,
            envelope=ADSREnvelope(0.005, 0.4, 0.2, 0.6)
        )
        
        # Attack transient
        op3 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=5.0,
            modulation_index=1.0,
            amplitude=0.3,
            envelope=ADSREnvelope(0.001, 0.15, 0.0, 0.2)  # Quick decay for attack only
        )
        
        # Use stacked routing for more complex timbre
        voice = FMVoice(
            [(op1, 0.6), (op2, 0.25), (op3, 0.15)],
            routing_type=OperatorRoutingType.STACKED
        )
        
        return Instrument(voice)
    
    @staticmethod
    def create_synth_bass() -> Instrument:
        """Create a synth bass FM instrument with complex routing."""
        # Main bass tone
        op1 = FMOperatorWithFeedback(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SAWTOOTH,
            c_to_m_ratio=1.0,
            modulation_index=3.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.01, 0.2, 0.8, 0.2),
            feedback_amount=0.2  # Add feedback for punchier bass
        )
        
        # Sub oscillator for thickness
        op2 = FMOperator(
            carrier_type=WaveformType.TRIANGLE,
            modulator_type=WaveformType.SQUARE,
            c_to_m_ratio=0.5,  # Half frequency (octave down)
            modulation_index=1.5,
            amplitude=0.4,
            envelope=ADSREnvelope(0.01, 0.1, 0.7, 0.3)
        )
        
        # Attack transient
        op3 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.TRIANGLE,
            c_to_m_ratio=4.0,
            modulation_index=2.0,
            amplitude=0.3,
            envelope=ADSREnvelope(0.001, 0.08, 0.0, 0.1)  # Very short for attack click
        )
        
        # Series routing creates more harmonic interaction
        voice = FMVoice(
            [(op1, 0.7), (op2, 0.25), (op3, 0.05)],
            routing_type=OperatorRoutingType.SERIES
        )
        
        return Instrument(voice)
    
    @staticmethod
    def create_lead_synth() -> Instrument:
        """Create a lead synth FM instrument using advanced techniques."""
        # Main lead tone
        op1 = FMOperatorWithFeedback(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=2.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.05, 0.2, 0.7, 0.4),
            feedback_amount=0.1  # Subtle feedback for warmth
        )
        
        # Higher harmonic content
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.TRIANGLE,
            c_to_m_ratio=2.0,  # Octave up
            modulation_index=1.2,
            amplitude=0.5,
            envelope=ADSREnvelope(0.04, 0.1, 0.6, 0.5)
        )
        
        # Detuned component for width
        op3 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.003,  # Slightly detuned for chorus-like effect
            modulation_index=0.8,
            amplitude=0.4,
            envelope=ADSREnvelope(0.08, 0.15, 0.6, 0.45)
        )
        
        voice = FMVoice([(op1, 0.6), (op2, 0.3), (op3, 0.1)])
        return Instrument(voice)
        
    @staticmethod
    def create_saxophone() -> Instrument:
        """Create a saxophone-like FM instrument using advanced techniques."""
        # Use the preset from InstrumentPresets
        return InstrumentPresets.saxophone()
    
    @staticmethod
    def _create_kick_drum() -> Instrument:
        """Create a kick drum using FM synthesis."""
        # Kick drum using FM synthesis
        kick_op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=4.0,  # Higher modulation for initial click
            amplitude=1.0,
            envelope=ADSREnvelope(0.001, 0.1, 0.0, 0.2)  # Quick attack, fast decay
        )
        
        # Sub-bass for body
        kick_op2 = FMOperator(
            carrier_type=WaveformType.SINE, 
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=0.5,
            modulation_index=1.0,
            amplitude=0.8,
            envelope=ADSREnvelope(0.005, 0.2, 0.0, 0.3)  # Slightly slower decay for body
        )
        
        # Create voice for kick
        kick_voice = FMVoice(
            [(kick_op1, 0.6), (kick_op2, 0.4)],
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(kick_voice)
    
    @staticmethod
    def _create_snare_drum() -> Instrument:
        """Create a snare drum using FM synthesis."""
        # Main snare tone
        snare_op1 = FMOperator(
            carrier_type=WaveformType.SQUARE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=1.0,
            modulation_index=3.0,
            amplitude=1.0, 
            envelope=ADSREnvelope(0.001, 0.1, 0.0, 0.1)  # Sharp attack and decay
        )
        
        # Noise component for snare rattle
        snare_op2 = FMOperator(
            carrier_type=WaveformType.NOISE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=1.0,
            modulation_index=0.5,
            amplitude=0.7,
            envelope=ADSREnvelope(0.001, 0.15, 0.1, 0.1)  # Slightly longer decay for rattle
        )
        
        # Create voice for snare
        snare_voice = FMVoice(
            [(snare_op1, 0.5), (snare_op2, 0.5)],
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(snare_voice)
    
    @staticmethod
    def _create_hihat() -> Instrument:
        """Create a hi-hat using FM synthesis."""
        # Hi-hat is mostly noise with some metallic resonance
        hihat_op1 = FMOperator(
            carrier_type=WaveformType.SQUARE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=5.0,  # High ratio for metallic character
            modulation_index=4.0, 
            amplitude=0.8,
            envelope=ADSREnvelope(0.001, 0.05, 0.0, 0.1)  # Very quick for closed hihat
        )
        
        # Additional noise for texture
        hihat_op2 = FMOperator(
            carrier_type=WaveformType.NOISE,
            modulator_type=WaveformType.SINE, 
            c_to_m_ratio=8.0,
            modulation_index=2.0,
            amplitude=0.6,
            envelope=ADSREnvelope(0.001, 0.04, 0.0, 0.08)
        )
        
        # Create voice for hihat
        hihat_voice = FMVoice(
            [(hihat_op1, 0.7), (hihat_op2, 0.3)],
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
            modulation_index=4.0,
            amplitude=0.8,
            envelope=ADSREnvelope(0.001, 0.3, 0.1, 0.4)  # Longer decay and release
        )
        
        # Additional noise for texture
        open_hihat_op2 = FMOperator(
            carrier_type=WaveformType.NOISE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=8.0,
            modulation_index=2.0,
            amplitude=0.6,
            envelope=ADSREnvelope(0.001, 0.25, 0.1, 0.3)
        )
        
        # Create voice for open hihat
        open_hihat_voice = FMVoice(
            [(open_hihat_op1, 0.7), (open_hihat_op2, 0.3)],
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(open_hihat_voice)
        
    @staticmethod
    def _create_ride_cymbal() -> Instrument:
        """Create a ride cymbal using FM synthesis."""
        # Ride has a clear bell-like tone with noise
        ride_op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=3.0,  # Create bell-like harmonic content
            modulation_index=3.0,
            amplitude=0.9,
            envelope=ADSREnvelope(0.001, 0.3, 0.2, 0.8)  # Long decay and sustain
        )
        
        # Noise component for shimmer
        ride_op2 = FMOperator(
            carrier_type=WaveformType.NOISE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=1.0,
            amplitude=0.4,
            envelope=ADSREnvelope(0.001, 0.4, 0.1, 0.7)
        )
        
        # Create voice for ride
        ride_voice = FMVoice(
            [(ride_op1, 0.7), (ride_op2, 0.3)],
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(ride_voice)
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize the city pop generator."""
        self.sample_rate = sample_rate
        self.sequencer = Sequencer(sample_rate)
        
        # Create instruments
        self.ep_piano = self.create_ep_piano()
        self.synth_bass = self.create_synth_bass()
        self.lead_synth = self.create_lead_synth()
        self.bell = InstrumentPresets.bell()
        self.saxophone = self.create_saxophone()
        self.strings = InstrumentPresets.synth_strings()
        
        # Create drum instruments
        self.kick = self._create_kick_drum()
        self.snare = self._create_snare_drum()
        self.hihat = self._create_hihat()
        self.open_hihat = self._create_open_hihat()
        self.ride = self._create_ride_cymbal()
        
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
        
    def generate_track(
        self, 
        root_note: str = 'F', 
        octave: int = 4, 
        tempo: int = 90
    ) -> Sequencer:
        """
        Generate a complete city pop / jazz fusion track.
        
        Args:
            root_note: Root note of the track
            octave: Octave of the root note
            tempo: Tempo in BPM
            
        Returns:
            A sequencer with the complete track
        """
        print(f"Generating city pop track in {root_note} with tempo {tempo} BPM...")
        
        # Start with a clean sequencer
        self.sequencer.clear()
        
        # Convert tempo to seconds per beat
        seconds_per_beat = 60 / tempo
        
        # Initialize harmonic structure
        root = Note(root_note, octave)
        
        # Choose a scale - Typically city pop uses major, lydian, or dorian
        scale_type = random.choice([
            ScaleType.MAJOR, 
            ScaleType.LYDIAN,
            ScaleType.DORIAN
        ])
        
        # If we're using Dorian, we need to adjust the root down a step 
        # to maintain the same key signature
        if scale_type == ScaleType.DORIAN:
            root = root.transpose(-2)  # Down a whole step
        
        scale = Scale(root, scale_type)
        print(f"Using {root.note_name} {scale_type.value} scale")
        
        # Choose a chord progression
        progression_degrees = random.choice(self.CHORD_PROGRESSIONS)
        
        # Apply chord extensions
        chords = []
        for degree in progression_degrees:
            # Choose a random chord extension from the available options for this degree
            chord_type = random.choice(self.CHORD_EXTENSIONS[degree])
            chord = Chord.from_scale_degree(scale, degree, chord_type)
            chords.append(chord)
        
        # Print chord progression
        chord_names = [chord.symbol for chord in chords]
        print(f"Chord progression: {' | '.join(chord_names)}")
        
        # Calculate section durations
        bar_duration = 4 * seconds_per_beat  # 4 beats per bar
        progression_duration = len(chords) * bar_duration  # One bar per chord
        
        # Generate form (song structure)
        form = self._generate_song_form(chords)
        
        # Generate and add each track
        self.track_indices['ep_piano'] = 0  # Track index for piano (first track)
        self._generate_ep_piano_comp(chords, tempo, seconds_per_beat, form)
        
        self.track_indices['synth_bass'] = 1  # Track index for bass
        self._generate_bass_line(chords, tempo, seconds_per_beat, form)
        
        self.track_indices['drums'] = 2  # Track index for drums
        self._generate_drum_pattern(tempo, seconds_per_beat, form)
        
        self.track_indices['lead_synth'] = 3  # Track index for melody
        self._generate_melody(scale, chords, tempo, seconds_per_beat, form)
        
        self.track_indices['bell'] = 4  # Track index for bell accents
        self._generate_bell_accent(scale, chords, tempo, seconds_per_beat)
        
        # Add optional saxophone part for richer arrangements
        if random.random() < 0.7:  # 70% chance to include saxophone
            self.track_indices['saxophone'] = 5
            self._generate_saxophone_part(scale, chords, tempo, seconds_per_beat, form)
        
        # Add optional string pad for atmosphere
        next_track = 6 if self.track_indices['saxophone'] is not None else 5
        if random.random() < 0.5:  # 50% chance to include strings
            self.track_indices['strings'] = next_track
            self._generate_string_pad(scale, chords, tempo, seconds_per_beat, form)
        
        # Apply effects to each track
        self._apply_track_effects(tempo)
        
        return self.sequencer
        
    def _generate_song_form(self, chords: List[Chord]) -> Dict[str, List[int]]:
        """
        Generate the song form (intro, verse, chorus, etc.).
        
        Args:
            chords: List of chords in the progression
            
        Returns:
            Dictionary mapping section names to bar indices
        """
        print("Generating song form...")
        
        # Create a structure typical for city pop songs
        # We'll create a form where each section is a number of repeats of the chord progression
        
        # Determine number of bars in the chord progression
        progression_length = len(chords)
        
        # City pop typically has an intro, verse, chorus, bridge, outro structure
        # Each section is a multiple of the chord progression length
        
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
    
    def _generate_drum_pattern(
        self, 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]]
    ) -> None:
        """
        Generate a city pop drum pattern.
        
        Args:
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating drum pattern...")
        
        # Calculate durations
        bar_duration = 4 * seconds_per_beat
        
        # Total number of bars
        total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
        
        # Define drum patterns (16th notes per bar)
        # City pop often uses a fusion of funk, disco, and jazz rhythms
        
        # Basic patterns for different sections
        kick_patterns = {
            'intro': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Simple kick on 1 and 3
            'verse': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Simple kick on 1 and 3
            'chorus': [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # More active for chorus
            'bridge': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Syncopated for bridge
            'outro': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # Simple for outro
        }
        
        snare_patterns = {
            'intro': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # Standard backbeat
            'verse': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # Standard backbeat
            'chorus': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.7, 0], # Added ghost note
            'bridge': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.8], # Slight variation
            'outro': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],    # Standard backbeat
        }
        
        hihat_patterns = {
            'intro': [0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0],  # Basic pattern
            'verse': [0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0],  # Basic pattern
            'chorus': [0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0],  # Basic pattern
            'bridge': [0, 0, 0.6, 0, 0, 0, 0.6, 0, 0, 0, 0.6, 0, 0, 0, 0.6, 0],  # Sparser for contrast
            'outro': [0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0],  # Basic pattern
        }
        
        ride_patterns = {
            'intro': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No ride in intro
            'verse': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No ride in verse
            'chorus': [0.7, 0, 0, 0, 0.5, 0, 0.5, 0, 0.7, 0, 0, 0, 0.5, 0, 0.5, 0],  # Ride in chorus
            'bridge': [0.7, 0, 0.5, 0, 0.7, 0, 0.5, 0, 0.7, 0, 0.5, 0, 0.7, 0, 0.5, 0],  # More ride in bridge
            'outro': [0.7, 0, 0, 0, 0.5, 0, 0.5, 0, 0.7, 0, 0, 0, 0.5, 0, 0.5, 0],  # Ride in outro
        }
        
        # Open hihat patterns (used for accents)
        open_hihat_patterns = {
            'intro': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No open hats in intro
            'verse': [0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0],  # Occasional open hat
            'chorus': [0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0.7],  # More open hats
            'bridge': [0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0.7],  # Consistent open hats
            'outro': [0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0],  # Occasional open hat
        }
        
        # Fill patterns for transitions
        fill_kick = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
        fill_snare = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        
        # For each bar, determine which section it belongs to and apply the appropriate pattern
        for bar in range(total_bars):
            # Determine which section this bar belongs to
            current_section = None
            for section, bar_indices in form.items():
                if bar in bar_indices:
                    if 'verse' in section:
                        current_section = 'verse'
                    elif 'chorus' in section:
                        current_section = 'chorus'
                    elif 'bridge' in section:
                        current_section = 'bridge'
                    elif 'intro' in section:
                        current_section = 'intro'
                    elif 'outro' in section:
                        current_section = 'outro'
                    break
            
            # Skip if we couldn't determine the section (shouldn't happen)
            if current_section is None:
                continue
            
            # Determine if this is a fill bar (typically the last bar of a section)
            is_fill_bar = False
            for section, bar_indices in form.items():
                if bar == bar_indices[-1] and section != 'outro':
                    is_fill_bar = True
                    break
            
            # Start time for this bar
            start_time = bar * bar_duration
            
            # Choose appropriate patterns based on section
            kick_pattern = fill_kick if is_fill_bar else kick_patterns[current_section]
            snare_pattern = fill_snare if is_fill_bar else snare_patterns[current_section]
            hihat_pattern = hihat_patterns[current_section]
            ride_pattern = ride_patterns[current_section]
            open_hihat_pattern = open_hihat_patterns[current_section]
            
            # Add drum hits for this bar
            for i in range(16):  # 16 divisions per bar (16th notes)
                note_time = start_time + (i / 16) * bar_duration
                
                # Add kick drum notes
                if kick_pattern[i] > 0:
                    self.sequencer.add_note(
                        self.kick,
                        "C2",  # Note doesn't matter much for drums, just using low note
                        note_time,
                        0.1,  # Short duration
                        kick_pattern[i] * 0.7  # Volume scaling
                    )
                
                # Add snare drum notes
                if snare_pattern[i] > 0:
                    self.sequencer.add_note(
                        self.snare,
                        "D2",
                        note_time,
                        0.1,
                        snare_pattern[i] * 0.6
                    )
                
                # Add hihat notes
                if hihat_pattern[i] > 0:
                    self.sequencer.add_note(
                        self.hihat,
                        "F#3",
                        note_time,
                        0.05,
                        hihat_pattern[i] * 0.4
                    )
                
                # Add open hihat notes
                if open_hihat_pattern[i] > 0:
                    self.sequencer.add_note(
                        self.open_hihat,
                        "G#3",
                        note_time,
                        0.2,  # Longer duration for open hihat
                        open_hihat_pattern[i] * 0.35
                    )
                
                # Add ride cymbal notes
                if ride_pattern[i] > 0:
                    self.sequencer.add_note(
                        self.ride,
                        "A3",
                        note_time,
                        0.15,
                        ride_pattern[i] * 0.3
                    )
            
            # Add drum fills at the end of sections
            if is_fill_bar:
                # Add some extra flourishes in the last quarter of the bar
                for i in range(12, 16):
                    if random.random() < 0.6:  # 60% chance for each possible hit
                        note_time = start_time + (i / 16) * bar_duration
                        
                        # Randomly choose between snare and tom-like sounds
                        if random.random() < 0.7:
                            # Snare hit
                            self.sequencer.add_note(
                                self.snare,
                                "D2",
                                note_time,
                                0.1,
                                random.uniform(0.4, 0.7)
                            )
                        else:
                            # Use kick at different pitches for tom-like sounds
                            pitch = random.choice(["C2", "E2", "G2"])
                            self.sequencer.add_note(
                                self.kick,
                                pitch,
                                note_time,
                                0.1,
                                random.uniform(0.4, 0.6)
                            )
    
    def _generate_ep_piano_comp(
        self, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate electric piano chord comping.
        
        Args:
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating electric piano comping...")
        
        # Create rhythmic patterns for comping - use syncopation
        verse_rhythm_patterns = [
            [1, 0, 0, 0, 1, 0, 0.5, 0, 0.5, 0, 0, 0],  # Syncopated pattern 1
            [1, 0, 0, 0, 0.7, 0, 0, 0, 1, 0, 0, 0.5],  # Syncopated pattern 2
            [0.7, 0, 0.5, 0, 0, 0, 1, 0, 0, 0, 0.7, 0]  # Syncopated pattern 3
        ]
        
        # More active patterns for chorus
        chorus_rhythm_patterns = [
            [1, 0, 0.5, 0, 1, 0, 0.7, 0, 0.5, 0, 0.7, 0],  # More active pattern 1
            [1, 0, 0.7, 0, 0.5, 0, 1, 0, 0.7, 0, 0.5, 0],  # More active pattern 2
            [0.8, 0, 0.6, 0, 1, 0, 0.6, 0, 0.8, 0, 0.7, 0]  # More active pattern 3
        ]
        
        # Simple patterns for intro/outro
        intro_rhythm_patterns = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0],  # Simple pattern 1
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # Simpler pattern 2
        ]
        
        # Sparse pattern for bridge
        bridge_rhythm_patterns = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0, 0],  # Sparse pattern 1
            [0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sparse pattern 2
        ]
        
        bar_duration = 4 * seconds_per_beat
        
        # Calculate total bars if form is provided
        if form:
            total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
        else:
            # Default to one chord per bar if no form is provided
            total_bars = len(chords)
        
        # For each bar in the song
        for bar in range(total_bars):
            # Determine which section this bar belongs to
            current_section = None
            if form:
                for section, bar_indices in form.items():
                    if bar in bar_indices:
                        if 'verse' in section:
                            current_section = 'verse'
                        elif 'chorus' in section:
                            current_section = 'chorus'
                        elif 'bridge' in section:
                            current_section = 'bridge'
                        elif 'intro' in section:
                            current_section = 'intro'
                        elif 'outro' in section:
                            current_section = 'outro'
                        break
            
            # Default to verse if no section is found or no form provided
            if current_section is None:
                current_section = 'verse'
            
            # Get the chord for this bar (cycle through progression)
            chord_idx = bar % len(chords)
            chord = chords[chord_idx]
            
            # Choose appropriate rhythm pattern based on section
            if current_section == 'verse':
                pattern = random.choice(verse_rhythm_patterns)
            elif current_section == 'chorus':
                pattern = random.choice(chorus_rhythm_patterns)
            elif current_section == 'bridge':
                pattern = random.choice(bridge_rhythm_patterns)
            else:  # intro or outro
                pattern = random.choice(intro_rhythm_patterns)
            
            # Determine if the chord should be voiced as a block chord or spread out
            # More spread voicings in chorus and bridge for interest
            if current_section == 'chorus' or current_section == 'bridge':
                voicing_style = random.choice(["block", "spread", "spread"])  # 2/3 chance for spread
            else:
                voicing_style = random.choice(["block", "spread"])  # 50/50
            
            # Start time for this bar
            start_time = bar * bar_duration
            
            # Get the chord notes
            chord_notes = chord.notes
            
            # Add extended voicings for certain sections
            if current_section in ['chorus', 'bridge']:
                # Add 9ths, 13ths extensions for interest in these sections
                chord_notes = self._add_city_pop_extensions(chord).notes
            
            # Generate the comping for this bar
            for beat_idx, velocity in enumerate(pattern):
                if velocity > 0:
                    # Calculate the time for this beat
                    beat_time = start_time + (beat_idx / 12) * bar_duration
                    
                    # Duration is either short or long, with longer durations in slower sections
                    if current_section in ['intro', 'outro', 'bridge']:
                        note_duration = random.choice([0.5, 0.8, 1.2]) * seconds_per_beat
                    else:
                        note_duration = random.choice([0.2, 0.5, 0.8]) * seconds_per_beat
                    
                    if voicing_style == "block":
                        # Play all notes at once
                        for note in chord_notes:
                            self.sequencer.add_note(
                                self.ep_piano,
                                note,
                                beat_time,
                                note_duration,
                                velocity * 0.5  # Adjust volume
                            )
                    else:
                        # Spread the notes slightly
                        for i, note in enumerate(chord_notes):
                            spread_time = beat_time + i * 0.03  # 30ms between notes
                            self.sequencer.add_note(
                                self.ep_piano,
                                note,
                                spread_time,
                                note_duration,
                                velocity * 0.5  # Adjust volume
                            )
    
    def _add_city_pop_extensions(self, chord: Chord) -> Chord:
        """
        Add city pop style extensions to a chord.
        
        Args:
            chord: The original chord
            
        Returns:
            A new chord with extensions added
        """
        # City pop often uses 9ths, 11ths, 13ths
        chord_type = chord.chord_type
        root = chord.root
        
        # For major7 chords, add 9ths or 13ths
        if chord_type == ChordType.MAJOR7.value:
            # Create custom chord with added extensions
            notes = chord.notes.copy()
            
            # Add 9th (up 14 semitones from root)
            ninth = root.transpose(14)
            
            # Sometimes add 13th (up 21 semitones from root)
            if random.random() < 0.4:  # 40% chance
                thirteenth = root.transpose(21)
                notes.extend([ninth, thirteenth])
            else:
                notes.append(ninth)
            
            # Return a custom chord with these notes
            return Chord(root, chord_type)
            
        # For dominant7 chords, add 9ths or altered 5ths
        elif chord_type == ChordType.DOMINANT7.value:
            # More complex alterations for dominant chords
            notes = chord.notes.copy()
            
            # Add 9th or b9 or #9
            extensions = [
                root.transpose(13),  # b9
                root.transpose(14),  # 9
                root.transpose(15),  # #9
            ]
            extension = random.choice(extensions)
            notes.append(extension)
            
            # Return a custom chord with these notes
            return Chord(root, chord_type)
        
        # Return original chord if no extensions applied
        return chord
    
    def _generate_bass_line(
        self, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate a funky bass line for the chord progression.
        
        Args:
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating synth bass line...")
        
        bar_duration = 4 * seconds_per_beat
        
        # Calculate total bars if form is provided
        if form:
            total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
        else:
            # Default to one chord per bar if no form is provided
            total_bars = len(chords)
            
        # Bass patterns for different sections - 16th note patterns (16 positions per bar)
        # Basic patterns
        basic_patterns = [
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # Pattern 1
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # Pattern 2
        ]
        
        # More funky/active patterns for chorus
        funky_patterns = [
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # Funky Pattern 1
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # Funky Pattern 2
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],  # Funky Pattern 3
        ]
        
        # More complex patterns for bridge
        complex_patterns = [
            [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],  # Complex Pattern 1
            [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],  # Complex Pattern 2
        ]
        
        # Simple patterns for intro/outro
        simple_patterns = [
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Simple Pattern 1
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Simple Pattern 2
        ]
        
        # Map sections to patterns
        section_patterns = {
            'intro': simple_patterns,
            'verse': basic_patterns,
            'chorus': funky_patterns,
            'bridge': complex_patterns,
            'outro': simple_patterns
        }
        
        # For each bar in the song
        for bar in range(total_bars):
            # Determine which section this bar belongs to
            current_section = None
            if form:
                for section, bar_indices in form.items():
                    if bar in bar_indices:
                        if 'verse' in section:
                            current_section = 'verse'
                        elif 'chorus' in section:
                            current_section = 'chorus'
                        elif 'bridge' in section:
                            current_section = 'bridge'
                        elif 'intro' in section:
                            current_section = 'intro'
                        elif 'outro' in section:
                            current_section = 'outro'
                        break
            
            # Default to verse if no section is found or no form provided
            if current_section is None:
                current_section = 'verse'
                
            # Get the chord for this bar (cycle through progression)
            chord_idx = bar % len(chords)
            chord = chords[chord_idx]
            
            # Choose a bass pattern appropriate for the section
            pattern = random.choice(section_patterns[current_section])
            
            # Start time for this bar
            start_time = bar * bar_duration
            
            # Get the root note and other chord tones for the bass line
            root_note = chord.root.transpose(-12)  # Down an octave
            fifth_note = chord.root.transpose(-12 + 7)  # Fifth up from root
            octave_note = chord.root.transpose(0)  # Same as chord root
            
            # Get chord quality-specific tones
            if 'minor' in chord.chord_type:
                third_note = root_note.transpose(3)  # Minor third
            else:
                third_note = root_note.transpose(4)  # Major third
                
            if 'dominant' in chord.chord_type or 'minor7' in chord.chord_type:
                seventh_note = root_note.transpose(10)  # Minor seventh
            elif 'major7' in chord.chord_type:
                seventh_note = root_note.transpose(11)  # Major seventh
            else:
                seventh_note = None
            
            # Determine pattern strategy based on section
            if current_section == 'chorus':
                # More melodic and active for chorus
                pattern_strategy = random.choice(["melodic", "walking", "octave_jumps"])
            elif current_section == 'bridge':
                # More complex for bridge
                pattern_strategy = random.choice(["melodic", "chromatic", "walking"])
            elif current_section in ['intro', 'outro']:
                # Simpler for intro/outro
                pattern_strategy = "root_dominant"
            else:  # verse
                # More standard for verse
                pattern_strategy = random.choice(["root_dominant", "walking", "octave_jumps"])
            
            # Track the previous note to allow for smooth voice leading
            prev_note = root_note
            
            for i, velocity in enumerate(pattern):
                if velocity > 0:
                    # Time for this note
                    note_time = start_time + (i / 16) * bar_duration
                    
                    # Duration, slightly varied
                    # Longer notes in intro/outro, shorter in active sections
                    if current_section in ['intro', 'outro']:
                        duration = random.uniform(0.2, 0.4) * seconds_per_beat
                    else:
                        duration = random.uniform(0.1, 0.3) * seconds_per_beat
                    
                    # Choose note based on pattern strategy
                    if pattern_strategy == "root_dominant":
                        # Mostly root notes with occasional fifths
                        if random.random() < 0.2:  # 20% chance for fifth
                            note = fifth_note
                        else:
                            note = root_note
                    elif pattern_strategy == "walking":
                        # Walking bass line feel
                        if i % 4 == 0:  # On main beats
                            note = root_note
                        elif i % 4 == 2:  # On offbeats
                            note = fifth_note
                        else:
                            # Use passing tones
                            passing_options = [
                                root_note.transpose(2),  # Major second
                                third_note,              # Third
                                fifth_note
                            ]
                            note = random.choice(passing_options)
                    elif pattern_strategy == "octave_jumps":
                        # Jumps between octaves
                        if i % 8 < 4:
                            note = root_note
                        else:
                            note = octave_note
                    elif pattern_strategy == "melodic":
                        # More melodic bassline using chord tones
                        if i % 4 == 0:  # On main beats
                            note = root_note
                        else:
                            # Choose from chord tones
                            chord_tones = [root_note, third_note, fifth_note]
                            if seventh_note:
                                chord_tones.append(seventh_note)
                            note = random.choice(chord_tones)
                    elif pattern_strategy == "chromatic":
                        # Use chromatic approaches for jazz feel
                        if i % 4 == 0:  # On main beats
                            note = root_note
                        elif random.random() < 0.3:  # 30% chance for chromatic approach
                            semitones = random.choice([-1, 1])  # Up or down 1 semitone
                            note = prev_note.transpose(semitones)
                        else:
                            # Use chord tones
                            note = random.choice([root_note, third_note, fifth_note])
                    
                    # Add the note to the sequencer
                    self.sequencer.add_note(
                        self.synth_bass,
                        note,
                        note_time,
                        duration,
                        velocity * 0.6  # Adjust volume
                    )
                    
                    # Track this note for the next iteration
                    prev_note = note
    
    class MarkovMelodyGenerator:
        """Generate melodies using Markov chains."""
        
        def __init__(self):
            """Initialize the Markov melody generator."""
            # Define transition probabilities between scale degrees
            # Based on analysis of city pop melodies
            self.transitions = {
                1: {1: 0.1, 2: 0.3, 3: 0.3, 5: 0.2, 6: 0.1},
                2: {1: 0.4, 3: 0.3, 5: 0.2, 7: 0.1},
                3: {2: 0.3, 4: 0.3, 5: 0.3, 7: 0.1},
                4: {3: 0.4, 5: 0.4, 7: 0.2},
                5: {1: 0.3, 4: 0.3, 6: 0.3, 7: 0.1},
                6: {5: 0.4, 7: 0.3, 2: 0.3},
                7: {1: 0.6, 6: 0.3, 5: 0.1}
            }
            
            # Extended transitions for pentatonic-like leaps found in city pop
            self.leaps = {
                1: {3: 0.5, 5: 0.5},  # Leaps to 3rd and 5th from root
                3: {5: 0.6, 1: 0.4},  # Leaps to 5th and root from 3rd
                5: {1: 0.7, 3: 0.3},  # Leaps to root and 3rd from 5th
            }
            
            # Style-specific melodic patterns - common city pop melodic motifs
            self.motifs = [
                [1, 3, 5, 4, 3, 2, 1],           # Descending motif
                [5, 6, 1, 2, 3],                 # Ascending motif
                [1, 5, 1, 3, 1],                 # Arpeggiated motif
                [5, 3, 2, 3, 5, 6, 5],           # Neighbor tone motif
                [1, 3, 5, 3, 5, 6, 5, 3, 1],     # Wave-like motif
            ]
        
        def generate_phrase(
            self, 
            scale: Scale, 
            start_degree: int, 
            length: int, 
            use_motif: bool = False,
            octave: int = 0
        ) -> List[Note]:
            """
            Generate a melodic phrase using Markov transitions.
            
            Args:
                scale: The scale to use
                start_degree: Starting scale degree (1-7)
                length: Number of notes in the phrase
                use_motif: Whether to use a predefined motif
                octave: Octave adjustment (0 = no adjustment, 1 = up an octave)
                
            Returns:
                List of notes in the phrase
            """
            if use_motif and length >= 5:
                # Use a predefined motif (truncated or extended as needed)
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
                        if random.random() < 0.3 and current in self.leaps:
                            # Occasionally use leaps for city pop flavor
                            next_candidates = list(self.leaps[current].keys())
                            next_weights = list(self.leaps[current].values())
                            next_degree = random.choices(
                                next_candidates,
                                weights=next_weights,
                                k=1
                            )[0]
                        else:
                            # Use regular transitions
                            next_candidates = list(self.transitions[current].keys())
                            next_weights = list(self.transitions[current].values())
                            next_degree = random.choices(
                                next_candidates,
                                weights=next_weights,
                                k=1
                            )[0]
                        
                        degrees.append(next_degree)
                        current = next_degree
            else:
                # Generate using Markov chain
                degrees = [start_degree]
                current = start_degree
                
                for _ in range(length - 1):
                    if random.random() < 0.2 and current in self.leaps:
                        # Occasionally use leaps for city pop flavor
                        next_candidates = list(self.leaps[current].keys())
                        next_weights = list(self.leaps[current].values())
                        next_degree = random.choices(
                            next_candidates,
                            weights=next_weights,
                            k=1
                        )[0]
                    else:
                        # Use regular transitions
                        next_candidates = list(self.transitions[current].keys())
                        next_weights = list(self.transitions[current].values())
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
    
    def _generate_melody(
        self, 
        scale: Scale, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate a melody that fits the chord progression.
        
        Args:
            scale: The scale to use for melody generation
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating melody...")
        
        # We'll create a melody that works with the given chords
        bar_duration = 4 * seconds_per_beat
        
        # Create Markov melody generator
        markov_generator = self.MarkovMelodyGenerator()
        
        # Calculate total bars if form is provided
        if form:
            total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
        else:
            # Default to length of chord progression if no form provided
            total_bars = len(chords)
            
        # For each section in the form, create a distinct melody
        # Store melodies for each section type to reuse
        section_melodies = {}
        
        # For each section in the form
        if form:
            for section, bar_indices in form.items():
                # Don't generate melody for intro and outro
                if section in ['intro', 'outro']:
                    continue
                    
                # Determine section type (verse, chorus, bridge)
                section_type = None
                if 'verse' in section:
                    section_type = 'verse'
                elif 'chorus' in section:
                    section_type = 'chorus'
                elif 'bridge' in section:
                    section_type = 'bridge'
                
                # Skip if we couldn't determine the section type
                if section_type is None:
                    continue
                
                # Check if we already have a melody for this section type
                if section_type in section_melodies:
                    # Reuse the existing melody with subtle variations
                    self._apply_section_melody(
                        section_melodies[section_type],
                        scale,
                        chords,
                        bar_indices,
                        seconds_per_beat,
                        bar_duration,
                        variation=True  # Apply variation
                    )
                else:
                    # Generate a new melody for this section type
                    section_melody = self._generate_section_melody(
                        scale,
                        chords,
                        markov_generator,
                        section_type,
                        len(bar_indices),
                        seconds_per_beat,
                        bar_duration
                    )
                    
                    # Store the melody for potential reuse
                    section_melodies[section_type] = section_melody
                    
                    # Apply the melody to this section
                    self._apply_section_melody(
                        section_melody,
                        scale,
                        chords,
                        bar_indices,
                        seconds_per_beat,
                        bar_duration
                    )
        else:
            # No form provided, just generate a simple melody
            # Treat the whole thing as a verse section
            bar_indices = list(range(total_bars))
            section_melody = self._generate_section_melody(
                scale,
                chords,
                markov_generator,
                'verse',
                total_bars,
                seconds_per_beat,
                bar_duration
            )
            
            # Apply the melody
            self._apply_section_melody(
                section_melody,
                scale,
                chords,
                bar_indices,
                seconds_per_beat,
                bar_duration
            )
    
    def _generate_section_melody(
        self,
        scale: Scale,
        chords: List[Chord],
        markov_generator: MarkovMelodyGenerator,
        section_type: str,
        num_bars: int,
        seconds_per_beat: float,
        bar_duration: float
    ) -> List[Tuple[Note, float, float, float]]:
        """
        Generate a melody for a specific section type.
        
        Args:
            scale: The scale to use
            chords: List of chords
            markov_generator: Markov chain generator
            section_type: Type of section ('verse', 'chorus', 'bridge')
            num_bars: Number of bars in the section
            seconds_per_beat: Duration of one beat in seconds
            bar_duration: Duration of one bar in seconds
            
        Returns:
            List of tuples (note, start_time, duration, velocity)
        """
        # Create phrases appropriate for the section type
        if section_type == 'verse':
            # Verses are more sparse and laid-back
            num_phrases = random.randint(2, 3)
            phrase_density = 0.6  # Percentage of time filled with notes vs. rests
            use_motifs = False  # Less structured
            octave = 0  # Standard register
        elif section_type == 'chorus':
            # Choruses are more dense and memorable
            num_phrases = random.randint(2, 4)
            phrase_density = 0.8  # More notes, fewer rests
            use_motifs = True  # More structured with recognizable motifs
            octave = 1  # Higher register
        else:  # bridge
            # Bridges provide contrast
            num_phrases = random.randint(1, 2)
            phrase_density = 0.7
            use_motifs = random.choice([True, False])
            octave = random.choice([0, 1])  # Either register
        
        # Calculate how long each phrase should be
        section_duration = num_bars * bar_duration
        phrase_duration = section_duration / num_phrases
        
        # Create the phrases
        section_melody = []
        
        for phrase_idx in range(num_phrases):
            # Determine start time for this phrase
            phrase_start = phrase_idx * phrase_duration
            
            # Each phrase doesn't use the full available time - leave some space
            phrase_length = random.uniform(0.7, 0.9) * phrase_duration * phrase_density
            
            # Create a rhythmic pattern for the phrase
            rhythm = []
            remaining_time = phrase_length
            
            # City pop often uses triplet feels or syncopation
            use_triplets = random.random() < 0.3  # 30% chance for triplet feel
            
            while remaining_time > 0.1:  # Minimum note duration
                if use_triplets:
                    # Triplet feel (division by 3)
                    duration_options = [1/3, 2/3, 1.0, 4/3]  # In beats
                else:
                    # Straight feel with some syncopation
                    duration_options = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # In beats
                
                duration_beats = random.choice(duration_options)
                duration = duration_beats * seconds_per_beat
                
                if duration > remaining_time:
                    duration = remaining_time
                
                rhythm.append(duration)
                remaining_time -= duration
            
            # Choose starting note based on the current chord
            chord_idx = int(phrase_start / bar_duration) % len(chords)
            chord = chords[chord_idx]
            
            # Use either 1, 3, or 5 scale degree as starting point
            if chord.chord_type == ChordType.MAJOR.value or chord.chord_type == ChordType.MAJOR7.value:
                start_degree_options = [1, 3, 5]
            elif chord.chord_type == ChordType.MINOR.value or chord.chord_type == ChordType.MINOR7.value:
                start_degree_options = [1, 3, 5]  # 3 is b3 in minor context
            else:
                start_degree_options = [1, 5]
            
            start_degree = random.choice(start_degree_options)
            
            # Generate the phrase using Markov chain
            markov_notes = markov_generator.generate_phrase(
                scale, 
                start_degree, 
                len(rhythm),
                use_motif=use_motifs,
                octave=octave
            )
            
            # Generate the melody notes
            current_time = phrase_start
            
            for note_idx, (note, duration) in enumerate(zip(markov_notes, rhythm)):
                # Determine if this should be a rest
                is_rest = random.random() < 0.2  # 20% chance for rest
                
                if not is_rest:
                    # Note duration slightly shorter than the rhythm duration for phrasing
                    note_duration = duration * random.uniform(0.8, 0.95)
                    
                    # Add some dynamic variation
                    if section_type == 'chorus':
                        # Louder for chorus
                        velocity = random.uniform(0.5, 0.8)
                    else:
                        velocity = random.uniform(0.4, 0.7)
                    
                    # Add to section melody
                    section_melody.append((note, current_time, note_duration, velocity))
                
                current_time += duration
                
        return section_melody
    
    def _apply_section_melody(
        self,
        section_melody: List[Tuple[Note, float, float, float]],
        scale: Scale,
        chords: List[Chord],
        bar_indices: List[int],
        seconds_per_beat: float,
        bar_duration: float,
        variation: bool = False
    ) -> None:
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
            
            # Add note to sequencer
            self.sequencer.add_note(
                self.lead_synth,
                note,
                abs_start_time,
                duration,
                velocity
            )

    def _generate_bell_accent(
        self, 
        scale: Scale, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate bell accents for color and texture.
        
        Args:
            scale: The scale to use
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Adding bell accents...")
        
        # Bell accents are sparse and typically hit on chord changes
        # or in strategic locations
        
        bar_duration = 4 * seconds_per_beat
        
        # Bells are used differently in different sections
        # Calculate total bars if form is provided
        if form:
            total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
            
            # Prepare section-specific accent strategies
            section_accent_strategies = {
                'intro': 0.2,     # 20% of bars get accents
                'verse': 0.1,     # 10% of bars get accents
                'chorus': 0.3,    # 30% of bars get accents
                'bridge': 0.4,    # 40% of bars get accents
                'outro': 0.2      # 20% of bars get accents
            }
            
            # Get high notes from the scale for bell accents
            high_notes = [note.transpose(24) for note in scale.notes]  # Two octaves up
            
            # For each bar, decide whether to add bell accents
            for bar in range(total_bars):
                # Determine which section this bar belongs to
                current_section = None
                for section, bar_indices in form.items():
                    if bar in bar_indices:
                        if 'verse' in section:
                            current_section = 'verse'
                        elif 'chorus' in section:
                            current_section = 'chorus'
                        elif 'bridge' in section:
                            current_section = 'bridge'
                        elif 'intro' in section:
                            current_section = 'intro'
                        elif 'outro' in section:
                            current_section = 'outro'
                        break
                
                # Skip if we couldn't determine the section
                if current_section is None:
                    continue
                
                # Probability for this section
                accent_probability = section_accent_strategies[current_section]
                
                # Decide whether to add an accent
                if random.random() < accent_probability:
                    # Possible times within this bar
                    possible_times = [
                        bar * bar_duration,                      # Beat 1
                        bar * bar_duration + seconds_per_beat,   # Beat 2
                        bar * bar_duration + 2 * seconds_per_beat, # Beat 3
                        bar * bar_duration + 3 * seconds_per_beat  # Beat 4
                    ]
                    
                    # Choose a time (weighted to prefer beats 1 and 3)
                    weights = [0.4, 0.1, 0.4, 0.1]  # Weights for beats 1,2,3,4
                    time = random.choices(possible_times, weights=weights, k=1)[0]
                    
                    # Find which chord we're in
                    chord_idx = bar % len(chords)
                    current_chord = chords[chord_idx]
                    
                    # Choose a high note from current chord or scale
                    # More chord tones in chorus for stability
                    if current_section == 'chorus':
                        chord_tone_probability = 0.8  # 80% in chorus
                    else:
                        chord_tone_probability = 0.7  # 70% otherwise
                        
                    if random.random() < chord_tone_probability:
                        # Choose a chord tone
                        chord_notes = [note.transpose(12) for note in current_chord.notes]  # Up an octave
                        chosen_note = random.choice(chord_notes)
                    else:
                        # Choose a scale tone
                        chosen_note = random.choice(high_notes)
                    
                    # Bell duration varies by section
                    if current_section in ['intro', 'outro']:
                        # Longer, atmospheric bells in intro/outro
                        bell_duration = random.uniform(2.0, 3.0) * seconds_per_beat
                        volume = 0.25  # Quieter
                    elif current_section == 'bridge':
                        # Varied bell durations in bridge
                        bell_duration = random.uniform(1.0, 2.5) * seconds_per_beat
                        volume = 0.3
                    else:
                        # Standard bell durations in verse/chorus
                        bell_duration = random.uniform(1.5, 2.0) * seconds_per_beat
                        volume = 0.3
                    
                    # Add the bell note
                    self.sequencer.add_note(
                        self.bell,
                        chosen_note,
                        time,
                        bell_duration,
                        volume
                    )
        else:
            # If no form is provided, use the original approach
            song_duration = len(chords) * bar_duration
            
            # Number of bell accents to add
            num_accents = random.randint(3, 6)
            
            # Possible times for accents (prefer chord changes)
            accent_times = []
            
            # Add chord change points
            for i in range(len(chords)):
                accent_times.append(i * bar_duration)
            
            # Add some additional potential accent points
            for i in range(len(chords)):
                accent_times.append(i * bar_duration + 2 * seconds_per_beat)  # On beat 3
            
            # Choose random accent times
            chosen_times = random.sample(accent_times, min(num_accents, len(accent_times)))
            
            # Get high notes from the scale for bell accents
            high_notes = [note.transpose(24) for note in scale.notes]  # Two octaves up
            
            for time in chosen_times:
                # Find which chord we're in
                chord_idx = int(time / bar_duration) % len(chords)
                current_chord = chords[chord_idx]
                
                # Choose a high note from current chord or scale
                if random.random() < 0.7:  # 70% chance for chord tone
                    chord_notes = [note.transpose(12) for note in current_chord.notes]  # Up an octave
                    chosen_note = random.choice(chord_notes)
                else:
                    chosen_note = random.choice(high_notes)
                
                # Bell duration is fairly long
                bell_duration = random.uniform(1.5, 2.5) * seconds_per_beat
                
                # Add the bell note
                self.sequencer.add_note(
                    self.bell,
                    chosen_note,
                    time,
                    bell_duration,
                    0.3  # Low volume
                )
            
    def _generate_saxophone_part(
        self, 
        scale: Scale, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate a saxophone solo part for city pop flavor.
        
        Args:
            scale: The scale to use for melody generation
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating saxophone part...")
        
        bar_duration = 4 * seconds_per_beat
        
        # Collection of jazz-like phrases
        phrase_rhythms = [
            # Long notes with space
            [2, 0, 1, 0, 1],
            # Eighth note runs
            [0.5, 0.5, 0.5, 0.5, 1, 0, 0.5, 0.5],
            # Triplet feel
            [0.66, 0.66, 0.66, 1, 0, 0.66, 0.66, 0.66],
            # Syncopated pattern
            [0.75, 0.25, 1, 0, 0.5, 0.5, 1],
        ]
        
        # More jazzy/complex patterns for bridge/solo sections
        complex_phrase_rhythms = [
            # More complex triplet feel
            [0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 1, 0, 0.5, 0.5],
            # Bebop-like eighths
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0],
            # Complex syncopation
            [0.75, 0.25, 0.5, 0.5, 0.25, 0.75, 1, 0, 0.5],
        ]
        
        if form:
            # In structured song form, the saxophone typically features in specific sections
            # Saxophone is more prominent in bridge sections and certain choruses
            saxophone_sections = []
            
            # Identify sections where saxophone should be prominent
            for section, bar_indices in form.items():
                # Saxophone plays in bridge and final choruses
                if 'bridge' in section:
                    saxophone_sections.extend(bar_indices)
                elif 'chorus' in section and section != 'chorus_1':  # Skip first chorus
                    # 70% chance to include saxophone in later choruses
                    if random.random() < 0.7:
                        saxophone_sections.extend(bar_indices)
            
            # Add occasional fills in other sections
            for section, bar_indices in form.items():
                if section not in ['bridge', 'chorus_2', 'chorus_3']:
                    # Just add a few fills in other sections
                    if 'verse' in section:
                        # Choose about 20% of verse bars for fills
                        num_fills = max(1, int(len(bar_indices) * 0.2))
                        fill_bars = random.sample(bar_indices, num_fills)
                        saxophone_sections.extend(fill_bars)
            
            # Sort sections by bar number
            saxophone_sections.sort()
            
            # Generate saxophone part for each identified bar
            for bar in saxophone_sections:
                # Determine what kind of part to play based on which section this is
                in_bridge = False
                in_chorus = False
                
                for section, bar_indices in form.items():
                    if bar in bar_indices:
                        if 'bridge' in section:
                            in_bridge = True
                        elif 'chorus' in section:
                            in_chorus = True
                        break
                
                # Choose phrase rhythm based on section
                if in_bridge:
                    # More complex phrases in bridge - saxophone feature
                    rhythm = random.choice(complex_phrase_rhythms)
                else:
                    # Standard phrases elsewhere
                    rhythm = random.choice(phrase_rhythms)
                
                # Scale durations based on tempo
                scaled_rhythm = [r * seconds_per_beat for r in rhythm]
                
                # Get the chord for this bar
                chord_idx = bar % len(chords)
                chord = chords[chord_idx]
                
                # Start time for this bar
                current_time = bar * bar_duration
                
                # Generate notes for this phrase
                for duration in scaled_rhythm:
                    if duration > 0:  # Skip rests (0 duration)
                        # Choose note based on chord and scale
                        
                        # Bridge sections have more scale tones for exploration
                        if in_bridge:
                            is_chord_tone = random.random() < 0.6  # 60% chance for chord tone
                        else:
                            is_chord_tone = random.random() < 0.8  # 80% chance for chord tone otherwise
                        
                        # Get notes in a higher register
                        octave_adjust = 12  # One octave up
                        
                        if is_chord_tone:
                            chord_notes = [note.transpose(octave_adjust) for note in chord.notes]
                            
                            # Add extensions more often in bridge sections
                            extension_chance = 0.5 if in_bridge else 0.3
                            
                            if random.random() < extension_chance:
                                # Add 9th (14 semitones above root)
                                ninth = chord.root.transpose(octave_adjust + 14)
                                # Sometimes add 11th or 13th in bridge sections
                                if in_bridge and random.random() < 0.3:
                                    if random.random() < 0.5:
                                        # Add 11th (17 semitones above root)
                                        eleventh = chord.root.transpose(octave_adjust + 17)
                                        chord_notes.append(eleventh)
                                    else:
                                        # Add 13th (21 semitones above root)
                                        thirteenth = chord.root.transpose(octave_adjust + 21)
                                        chord_notes.append(thirteenth)
                                chord_notes.append(ninth)
                                
                            chosen_note = random.choice(chord_notes)
                        else:
                            # Choose from scale, but in a higher register
                            scale_notes = [note.transpose(octave_adjust) for note in scale.notes]
                            chosen_note = random.choice(scale_notes)
                        
                        # Adjust velocity based on note duration and section
                        if duration < 0.5 * seconds_per_beat:
                            # Shorter notes
                            velocity = random.uniform(0.5, 0.7)
                        else:
                            # Longer notes
                            velocity = random.uniform(0.4, 0.6)
                            
                        # Louder in bridge sections
                        if in_bridge:
                            velocity *= 1.1  # 10% louder
                            
                        # Add vibrato to longer notes by slightly reducing the actual duration
                        actual_duration = duration
                        if duration > 0.75 * seconds_per_beat:
                            actual_duration = duration * 0.95  # Slight gap for phrasing
                        
                        # Add the note to the sequencer
                        self.sequencer.add_note(
                            self.saxophone,
                            chosen_note,
                            current_time,
                            actual_duration,
                            velocity
                        )
                    
                    # Move to next note in the phrase
                    current_time += duration
        else:
            # If no form provided, use the original approach with a standalone solo
            song_duration = len(chords) * bar_duration
            
            # Select a portion of the song for the saxophone solo
            # Typically after the first statement of the chord progression
            start_bar = random.randint(len(chords), len(chords) * 2 - 2)
            solo_duration_bars = random.randint(4, 8)  # 4-8 bar solo
            
            start_time = start_bar * bar_duration
            end_time = start_time + solo_duration_bars * bar_duration
            
            # Limit to song duration
            end_time = min(end_time, song_duration)
            
            current_time = start_time
            
            while current_time < end_time:
                # Determine what chord we're currently playing over
                current_bar = int(current_time / bar_duration)
                chord_idx = current_bar % len(chords)
                current_chord = chords[chord_idx]
                
                # Choose a rhythmic phrase for this part of the solo
                rhythm = random.choice(phrase_rhythms)
                
                # Scale durations based on tempo
                scaled_rhythm = [r * seconds_per_beat for r in rhythm]
                
                # Generate notes for this phrase
                for duration in scaled_rhythm:
                    if duration > 0:  # Skip rests (0 duration)
                        # Choose note based on chord and scale
                        
                        # Determine if we're playing a chord tone or scale tone
                        is_chord_tone = random.random() < 0.7  # 70% chance for chord tone
                        
                        # Get notes in a higher register
                        if is_chord_tone:
                            chord_notes = [note.transpose(12) for note in current_chord.notes]
                            # Sometimes add extensions (9ths, 11ths, 13ths)
                            if random.random() < 0.3:  # 30% chance for extensions
                                # Add 9th (14 semitones above root)
                                ninth = current_chord.root.transpose(14)
                                chord_notes.append(ninth)
                            chosen_note = random.choice(chord_notes)
                        else:
                            # Choose from scale, but in a higher register
                            scale_notes = [note.transpose(12) for note in scale.notes]
                            chosen_note = random.choice(scale_notes)
                        
                        # Shorter notes need higher velocity for emphasis
                        if duration < 0.5 * seconds_per_beat:
                            velocity = random.uniform(0.5, 0.7)
                        else:
                            velocity = random.uniform(0.4, 0.6)
                        
                        # Add vibrato to longer notes by slightly reducing the actual duration
                        actual_duration = duration
                        if duration > 0.75 * seconds_per_beat:
                            actual_duration = duration * 0.95  # Slight gap for phrasing
                        
                        # Add the note to the sequencer
                        self.sequencer.add_note(
                            self.saxophone,
                            chosen_note,
                            current_time,
                            actual_duration,
                            velocity
                        )
                    
                    # Move to next note in the phrase
                    current_time += duration
    
    def _generate_string_pad(
        self, 
        scale: Scale, 
        chords: List[Chord], 
        tempo: int, 
        seconds_per_beat: float,
        form: Dict[str, List[int]] = None
    ) -> None:
        """
        Generate a string pad background for atmosphere.
        
        Args:
            scale: The scale to use
            chords: List of chords in the progression
            tempo: Tempo in BPM
            seconds_per_beat: Duration of one beat in seconds
            form: Song form structure
        """
        print("Generating string pad...")
        
        bar_duration = 4 * seconds_per_beat
        
        # Calculate total bars if form is provided
        if form:
            total_bars = max(max(bar_indices) for bar_indices in form.values()) + 1
            
            # String pads are used differently in different sections
            # Skip strings in some sections for arrangement variation
            string_section_probabilities = {
                'intro': 0.9,    # 90% chance to use strings in intro
                'verse': 0.4,    # 40% chance in verses (sparser)
                'chorus': 0.8,   # 80% chance in choruses (fuller)
                'bridge': 0.6,   # 60% chance in bridge 
                'outro': 0.9     # 90% chance in outro
            }
            
            # For each bar in the song
            for bar in range(total_bars):
                # Determine which section this bar belongs to
                current_section = None
                for section, bar_indices in form.items():
                    if bar in bar_indices:
                        if 'verse' in section:
                            current_section = 'verse'
                        elif 'chorus' in section:
                            current_section = 'chorus'
                        elif 'bridge' in section:
                            current_section = 'bridge'
                        elif 'intro' in section:
                            current_section = 'intro'
                        elif 'outro' in section:
                            current_section = 'outro'
                        break
                
                # Skip if we couldn't determine the section
                if current_section is None:
                    continue
                
                # Determine if we should have strings in this bar
                if random.random() > string_section_probabilities[current_section]:
                    continue  # Skip this bar
                
                # Get the chord for this bar
                chord_idx = bar % len(chords)
                chord = chords[chord_idx]
                
                # Start time for this bar
                start_time = bar * bar_duration
                
                # Strings sustain into the next chord slightly for smooth transitions
                # Longer in intro/outro, shorter in verses
                if current_section in ['intro', 'outro']:
                    duration = bar_duration * 1.1  # More overlap
                elif current_section == 'verse':
                    duration = bar_duration * 1.0  # Exactly one bar
                else:
                    duration = bar_duration * 1.05  # Slight overlap
                
                # Choose a subset of the chord tones for the pad
                # Often with wide voicings
                chord_notes = chord.notes.copy()
                
                # Add extensions based on section
                if current_section in ['chorus', 'bridge']:
                    # More complex extensions in these sections
                    extension_chance = 0.7  # 70% chance
                    can_add_upper_extensions = True  # Can add 11ths, 13ths
                elif current_section in ['intro', 'outro']:
                    # Some extensions, but simpler
                    extension_chance = 0.5  # 50% chance
                    can_add_upper_extensions = False  # Only 9ths
                else:  # verse
                    # Fewer extensions in verses for cleaner sound
                    extension_chance = 0.3  # 30% chance
                    can_add_upper_extensions = False  # Only 9ths
                    
                # Add extensions
                if random.random() < extension_chance:
                    # Always consider 9ths
                    ninth = chord.root.transpose(14)  # 14 semitones = 9th
                    chord_notes.append(ninth)
                    
                    # Only add upper extensions in certain sections
                    if can_add_upper_extensions and random.random() < 0.5:
                        if random.random() < 0.5:
                            # Add 11th (17 semitones from root)
                            eleventh = chord.root.transpose(17)
                            chord_notes.append(eleventh)
                        else:
                            # Add 13th (21 semitones from root)
                            thirteenth = chord.root.transpose(21)
                            chord_notes.append(thirteenth)
                
                # Move some notes to different octaves for wider voicing
                voicing = []
                
                # Different voicing strategies based on section
                if current_section == 'chorus':
                    # Fuller voicings in chorus
                    lowest_octave_adjust = -12  # Go lower
                    chord_notes_to_use = chord_notes  # Use all notes
                    
                    # Create voicing with wide spread
                    for i, note in enumerate(chord_notes_to_use):
                        if i == 0:  # Root
                            voicing.append(note.transpose(lowest_octave_adjust))  # Down an octave
                        elif i == 1 and len(chord_notes_to_use) > 2:  # First upper note
                            voicing.append(note)  # Keep at original octave
                        elif i == len(chord_notes_to_use) - 1:  # Highest note
                            voicing.append(note.transpose(12))  # Up an octave
                        else:
                            # Randomly place other notes
                            octave_adjust = random.choice([0, 12])  # Original or up an octave
                            voicing.append(note.transpose(octave_adjust))
                    
                elif current_section == 'bridge':
                    # More clustered voicings in bridge
                    # Take a subset of notes for a cleaner sound
                    chord_notes_to_use = chord_notes[:3] + chord_notes[-1:]  # Root, 3rd, 5th, plus highest extension
                    
                    # Create clustered voicing
                    for i, note in enumerate(chord_notes_to_use):
                        if i == 0:  # Root
                            voicing.append(note)  # Keep at original octave for bridge
                        else:
                            # Keep notes closer together
                            voicing.append(note)  # All at same octave
                    
                else:  # intro, verse, outro
                    # Standard wide voicings
                    chord_notes_to_use = chord_notes
                    
                    for i, note in enumerate(chord_notes_to_use):
                        # Lower root, raise upper voices
                        if i == 0:  # Root
                            voicing.append(note.transpose(-12))  # Down an octave
                        elif i == len(chord_notes_to_use) - 1:  # Highest note
                            voicing.append(note)  # Keep as is
                        else:
                            if random.random() < 0.5:
                                voicing.append(note)  # Keep as is
                            else:
                                voicing.append(note.transpose(12))  # Up an octave
                
                # Adjust velocity based on section
                if current_section in ['intro', 'outro']:
                    # Louder in bookend sections
                    velocity = random.uniform(0.25, 0.35)
                elif current_section == 'chorus':
                    # Medium in chorus (so it doesn't overpower)
                    velocity = random.uniform(0.2, 0.3)
                else:
                    # Quieter in verse and bridge
                    velocity = random.uniform(0.15, 0.25)
                
                # Add each note of the chord to create the pad
                for note in voicing:
                    self.sequencer.add_note(
                        self.strings,
                        note,
                        start_time,
                        duration,
                        velocity
                    )
        else:
            # If no form provided, use the original approach
            # String pads typically follow the chord progression but with longer, sustained notes
            for chord_idx, chord in enumerate(chords):
                # String pads are mostly played as sustained chords
                start_time = chord_idx * bar_duration
                
                # Strings sustain into the next chord slightly for smooth transitions
                duration = bar_duration * 1.05  # Overlap slightly
                
                # Choose a subset of the chord tones for the pad
                # Often with wide voicings
                chord_notes = chord.notes
                
                # Sometimes add a 9th for color
                if random.random() < 0.4:  # 40% chance
                    ninth = chord.root.transpose(14)  # 14 semitones = 9th
                    chord_notes.append(ninth)
                
                # Move some notes to different octaves for wider voicing
                voicing = []
                for i, note in enumerate(chord_notes):
                    # Lower root, raise upper voices
                    if i == 0:  # Root
                        voicing.append(note.transpose(-12))  # Down an octave
                    elif i == len(chord_notes) - 1:  # Highest note
                        voicing.append(note)  # Keep as is
                    else:
                        if random.random() < 0.5:
                            voicing.append(note)  # Keep as is
                        else:
                            voicing.append(note.transpose(12))  # Up an octave
                
                # Play strings with low velocity for background
                velocity = random.uniform(0.2, 0.3)
                
                # Add each note of the chord to create the pad
                for note in voicing:
                    self.sequencer.add_note(
                        self.strings,
                        note,
                        start_time,
                        duration,
                        velocity
                    )
    
    def _apply_track_effects(self, tempo: int) -> None:
        """
        Apply effects to each instrument track for enhanced sound.
        
        Args:
            tempo: Tempo in BPM (for time-based effects)
        """
        print("Applying effects to tracks...")
        
        # Calculate delay time based on tempo (usually 1/8th or 1/16th note)
        eighth_note_delay = 60 / tempo / 2
        sixteenth_note_delay = 60 / tempo / 4
        
        # Apply effects to electric piano
        if self.track_indices['ep_piano'] is not None:
            piano_effects = EffectChain()
            # Add chorus for width
            piano_effects.add_effect(ChorusEffect(rate=0.6, depth=0.3, mix=0.2))
            # Add reverb for space
            piano_effects.add_effect(ReverbEffect(room_size=0.4, damping=0.5, mix=0.25))
            self.sequencer.apply_effects_to_track(self.track_indices['ep_piano'], piano_effects)
        
        # Apply effects to bass
        if self.track_indices['synth_bass'] is not None:
            bass_effects = EffectChain()
            # Light chorus for movement
            bass_effects.add_effect(ChorusEffect(rate=0.3, depth=0.2, mix=0.1))
            self.sequencer.apply_effects_to_track(self.track_indices['synth_bass'], bass_effects)
        
        # Apply effects to drums
        if self.track_indices['drums'] is not None:
            drums_effects = EffectChain()
            # Add reverb for city pop drum sound
            drums_effects.add_effect(ReverbEffect(room_size=0.4, damping=0.7, mix=0.2))
            # City pop often has compressed drums with reverb
            self.sequencer.apply_effects_to_track(self.track_indices['drums'], drums_effects)
        
        # Apply effects to lead synth
        if self.track_indices['lead_synth'] is not None:
            lead_effects = EffectChain()
            # Add delay for rhythmic interest
            lead_effects.add_effect(DelayEffect(delay_time=eighth_note_delay, feedback=0.3, mix=0.25))
            # Add reverb for space
            lead_effects.add_effect(ReverbEffect(room_size=0.5, damping=0.4, mix=0.2))
            self.sequencer.apply_effects_to_track(self.track_indices['lead_synth'], lead_effects)
        
        # Apply effects to bell
        if self.track_indices['bell'] is not None:
            bell_effects = EffectChain()
            # Add lots of reverb for atmosphere
            bell_effects.add_effect(ReverbEffect(room_size=0.8, damping=0.3, mix=0.4))
            self.sequencer.apply_effects_to_track(self.track_indices['bell'], bell_effects)
        
        # Apply effects to saxophone
        if self.track_indices['saxophone'] is not None:
            sax_effects = EffectChain()
            # Add delay for jazz feel
            sax_effects.add_effect(DelayEffect(delay_time=eighth_note_delay, feedback=0.25, mix=0.2))
            # Add reverb for space
            sax_effects.add_effect(ReverbEffect(room_size=0.5, damping=0.6, mix=0.3))
            self.sequencer.apply_effects_to_track(self.track_indices['saxophone'], sax_effects)
        
        # Apply effects to strings
        if self.track_indices['strings'] is not None:
            string_effects = EffectChain()
            # Add chorus for width
            string_effects.add_effect(ChorusEffect(rate=0.7, depth=0.6, mix=0.4))
            # Add reverb for space
            string_effects.add_effect(ReverbEffect(room_size=0.8, damping=0.3, mix=0.5))
            self.sequencer.apply_effects_to_track(self.track_indices['strings'], string_effects)

def generate_and_play_track(
    root_note: str = 'F', 
    octave: int = 4, 
    tempo: int = 90,
    output_file: str = "output/citypop_track.wav"
) -> str:
    """
    Generate and save a city pop track to a WAV file.
    
    Args:
        root_note: Root note of the track
        octave: Octave of the root note
        tempo: Tempo in BPM
        output_file: Path to the output WAV file
        
    Returns:
        The absolute path to the saved WAV file
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    generator = CityPopGenerator()
    sequencer = generator.generate_track(root_note, octave, tempo)
    
    print(f"Saving the generated track to {output_file}...")
    return sequencer.save_to_wav(output_file)

if __name__ == "__main__":
    # Generate a track in F major at 88 BPM
    generate_and_play_track('F', 4, 88)
