"""
Music Theory Algorithm Library

This module provides algorithmic implementations of fundamental music theory concepts
including notes, scales, intervals, chords, and rhythm.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Dict, Optional, Union, Tuple, ClassVar
from dataclasses import dataclass
import math


class NoteName(Enum):
    """Enum representing the 12 notes in Western music."""
    C = 0
    C_SHARP = 1  # Also D♭
    D = 2
    D_SHARP = 3  # Also E♭
    E = 4
    F = 5
    F_SHARP = 6  # Also G♭
    G = 7
    G_SHARP = 8  # Also A♭
    A = 9
    A_SHARP = 10  # Also B♭
    B = 11


# Mapping for string representations
NOTE_STR_MAP = {
    NoteName.C: 'C',
    NoteName.C_SHARP: 'C#',
    NoteName.D: 'D',
    NoteName.D_SHARP: 'D#',
    NoteName.E: 'E',
    NoteName.F: 'F',
    NoteName.F_SHARP: 'F#',
    NoteName.G: 'G',
    NoteName.G_SHARP: 'G#',
    NoteName.A: 'A',
    NoteName.A_SHARP: 'A#',
    NoteName.B: 'B'
}

# Reverse mapping from string to NoteName
STR_NOTE_MAP = {v: k for k, v in NOTE_STR_MAP.items()}


class Note:
    """Represents a musical note with a name and octave."""
    
    def __init__(self, note: Union[str, NoteName], octave: int = 4):
        """
        Initialize a Note object.
        
        Args:
            note: The name of the note (e.g., 'C', 'F#') or NoteName enum
            octave: The octave number (default is 4)
        
        Raises:
            ValueError: If the note name is invalid
        """
        if isinstance(note, str):
            if note not in STR_NOTE_MAP:
                raise ValueError(f"Invalid note name: {note}. Must be one of {list(STR_NOTE_MAP.keys())}")
            self._note_enum = STR_NOTE_MAP[note]
        else:
            self._note_enum = note
            
        self.octave = octave
    
    @property
    def note_enum(self) -> NoteName:
        """Get the NoteName enum value."""
        return self._note_enum
    
    @property
    def note_name(self) -> str:
        """Get the string representation of the note name."""
        return NOTE_STR_MAP[self._note_enum]
    
    @property
    def midi_number(self) -> int:
        """Calculate the MIDI note number for this note."""
        return (self.octave + 1) * 12 + self._note_enum.value
    
    @property
    def frequency(self) -> float:
        """Calculate the frequency in Hz (A4 = 440Hz)."""
        # Using the formula: f = 440 * 2^((n-69)/12)
        # where n is the MIDI note number
        return 440 * (2 ** ((self.midi_number - 69) / 12))
    
    def __eq__(self, other: object) -> bool:
        """Compare two notes for equality."""
        if not isinstance(other, Note):
            return NotImplemented
        return self._note_enum == other._note_enum and self.octave == other.octave
    
    def __str__(self) -> str:
        """String representation of the note."""
        return f"{self.note_name}{self.octave}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the note."""
        return f"Note('{self.note_name}', {self.octave})"
    
    def transpose(self, semitones: int) -> Note:
        """
        Transpose the note by the specified number of semitones.
        
        Args:
            semitones: Number of semitones to transpose (positive or negative)
            
        Returns:
            A new Note object representing the transposed note
        """
        new_midi = self.midi_number + semitones
        new_octave = (new_midi // 12) - 1
        new_note_value = new_midi % 12
        new_note_enum = NoteName(new_note_value)
        return Note(new_note_enum, new_octave)
    
    @classmethod
    def from_midi_number(cls, midi_number: int) -> Note:
        """
        Create a Note from a MIDI note number.
        
        Args:
            midi_number: MIDI note number (0-127)
            
        Returns:
            A Note object corresponding to the MIDI note number
        
        Raises:
            ValueError: If midi_number is out of range
        """
        if not 0 <= midi_number <= 127:
            raise ValueError(f"MIDI note number must be between 0 and 127, got {midi_number}")
            
        octave = (midi_number // 12) - 1
        note_value = midi_number % 12
        note_enum = NoteName(note_value)
        return cls(note_enum, octave)


class IntervalType(Enum):
    """Types of musical intervals."""
    PERFECT = 'P'
    MAJOR = 'M'
    MINOR = 'm'
    DIMINISHED = 'd'
    AUGMENTED = 'A'


class Interval:
    """Represents musical intervals."""
    
    # Interval names and their semitone counts
    INTERVALS: ClassVar[Dict[str, int]] = {
        'P1': 0,   # Perfect unison
        'm2': 1,   # Minor second
        'M2': 2,   # Major second
        'm3': 3,   # Minor third
        'M3': 4,   # Major third
        'P4': 5,   # Perfect fourth
        'TT': 6,   # Tritone
        'P5': 7,   # Perfect fifth
        'm6': 8,   # Minor sixth
        'M6': 9,   # Major sixth
        'm7': 10,  # Minor seventh
        'M7': 11,  # Major seventh
        'P8': 12   # Perfect octave
    }
    
    # The reverse mapping from semitones to interval name
    SEMITONES_TO_INTERVAL: ClassVar[Dict[int, str]] = {
        0: 'P1', 1: 'm2', 2: 'M2', 3: 'm3', 4: 'M3', 5: 'P4',
        6: 'TT', 7: 'P5', 8: 'm6', 9: 'M6', 10: 'm7', 11: 'M7', 12: 'P8'
    }
    
    def __init__(self, name: str):
        """
        Initialize an Interval object.
        
        Args:
            name: The name of the interval (e.g., 'P5', 'M3')
            
        Raises:
            ValueError: If the interval name is invalid
        """
        if name not in self.INTERVALS:
            raise ValueError(f"Invalid interval: {name}. Must be one of {list(self.INTERVALS.keys())}")
        
        self.name = name
        self.semitones = self.INTERVALS[name]
    
    def __str__(self) -> str:
        """String representation of the interval."""
        return self.name
    
    def __repr__(self) -> str:
        """Detailed string representation of the interval."""
        return f"Interval('{self.name}')"
    
    @classmethod
    def from_semitones(cls, semitones: int) -> Interval:
        """
        Create an Interval from a number of semitones.
        
        Args:
            semitones: Number of semitones (0-12)
            
        Returns:
            An Interval object
            
        Raises:
            ValueError: If semitones is out of range
        """
        semitones = semitones % 12  # Normalize to one octave
        if semitones not in cls.SEMITONES_TO_INTERVAL:
            raise ValueError(f"Invalid semitone count: {semitones}")
        
        name = cls.SEMITONES_TO_INTERVAL[semitones]
        return cls(name)
    
    @classmethod
    def between(cls, note1: Note, note2: Note) -> Interval:
        """
        Calculate the interval between two notes.
        
        Args:
            note1: The first note
            note2: The second note
            
        Returns:
            The interval between the notes
        """
        semitones = (note2.midi_number - note1.midi_number) % 12
        return cls.from_semitones(semitones)
    
    def apply_to(self, note: Note) -> Note:
        """
        Apply this interval to a note.
        
        Args:
            note: The starting note
            
        Returns:
            The resulting note after applying the interval
        """
        return note.transpose(self.semitones)


class ScaleType(Enum):
    """Types of musical scales."""
    MAJOR = 'major'
    NATURAL_MINOR = 'natural_minor'
    HARMONIC_MINOR = 'harmonic_minor'
    MELODIC_MINOR = 'melodic_minor'
    DORIAN = 'dorian'
    PHRYGIAN = 'phrygian'
    LYDIAN = 'lydian'
    MIXOLYDIAN = 'mixolydian'
    LOCRIAN = 'locrian'
    BLUES = 'blues'
    PENTATONIC_MAJOR = 'pentatonic_major'
    PENTATONIC_MINOR = 'pentatonic_minor'
    CHROMATIC = 'chromatic'


class Scale:
    """Represents musical scales."""
    
    # Defining scale patterns in terms of semitones from the root
    SCALE_PATTERNS: ClassVar[Dict[str, List[int]]] = {
        ScaleType.MAJOR.value: [0, 2, 4, 5, 7, 9, 11],                    # Major (Ionian)
        ScaleType.NATURAL_MINOR.value: [0, 2, 3, 5, 7, 8, 10],            # Natural minor (Aeolian)
        ScaleType.HARMONIC_MINOR.value: [0, 2, 3, 5, 7, 8, 11],           # Harmonic minor
        ScaleType.MELODIC_MINOR.value: [0, 2, 3, 5, 7, 9, 11],            # Melodic minor (ascending)
        ScaleType.DORIAN.value: [0, 2, 3, 5, 7, 9, 10],                   # Dorian mode
        ScaleType.PHRYGIAN.value: [0, 1, 3, 5, 7, 8, 10],                 # Phrygian mode
        ScaleType.LYDIAN.value: [0, 2, 4, 6, 7, 9, 11],                   # Lydian mode
        ScaleType.MIXOLYDIAN.value: [0, 2, 4, 5, 7, 9, 10],               # Mixolydian mode
        ScaleType.LOCRIAN.value: [0, 1, 3, 5, 6, 8, 10],                  # Locrian mode
        ScaleType.BLUES.value: [0, 3, 5, 6, 7, 10],                       # Blues scale
        ScaleType.PENTATONIC_MAJOR.value: [0, 2, 4, 7, 9],                # Major pentatonic
        ScaleType.PENTATONIC_MINOR.value: [0, 3, 5, 7, 10],               # Minor pentatonic
        ScaleType.CHROMATIC.value: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # Chromatic scale
    }
    
    def __init__(self, root: Note, scale_type: Union[str, ScaleType] = ScaleType.MAJOR):
        """
        Initialize a Scale object.
        
        Args:
            root: The root note of the scale
            scale_type: The type of scale (default is ScaleType.MAJOR)
            
        Raises:
            ValueError: If scale_type is invalid
        """
        if isinstance(scale_type, ScaleType):
            scale_type = scale_type.value
            
        if scale_type not in self.SCALE_PATTERNS:
            raise ValueError(f"Invalid scale type: {scale_type}")
        
        self.root = root
        self.scale_type = scale_type
        self.pattern = self.SCALE_PATTERNS[scale_type]
    
    @property
    def notes(self) -> List[Note]:
        """
        Get all notes in the scale.
        
        Returns:
            List of Note objects in the scale
        """
        return [self.root.transpose(interval) for interval in self.pattern]
    
    def get_degree(self, degree: int) -> Note:
        """
        Get a specific scale degree (1-based index).
        
        Args:
            degree: The scale degree (1 for root, 2 for second, etc.)
            
        Returns:
            The note at the specified scale degree
            
        Raises:
            ValueError: If degree is out of range
        """
        if degree < 1 or degree > len(self.pattern):
            raise ValueError(f"Invalid scale degree: {degree}. Must be between 1 and {len(self.pattern)}")
        
        # Convert to 0-based index
        idx = degree - 1
        return self.root.transpose(self.pattern[idx])
    
    def contains(self, note: Note) -> bool:
        """
        Check if a note belongs to the scale.
        
        Args:
            note: The note to check
            
        Returns:
            True if the note is in the scale, False otherwise
        """
        # Normalize to same octave for comparison
        note_name = note.note_name
        scale_note_names = [n.note_name for n in self.notes]
        return note_name in scale_note_names
    
    def __str__(self) -> str:
        """String representation of the scale."""
        note_strs = [str(note) for note in self.notes]
        return f"{self.root.note_name} {self.scale_type}: {', '.join(note_strs)}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the scale."""
        return f"Scale({repr(self.root)}, '{self.scale_type}')"
    
    @property
    def relative_minor(self) -> Scale:
        """
        Get the relative minor scale of a major scale.
        
        Returns:
            The relative minor scale
            
        Raises:
            ValueError: If this is not a major scale
        """
        if self.scale_type != ScaleType.MAJOR.value:
            raise ValueError("Only major scales have relative minors")
        
        # Relative minor root is at the 6th degree of the major scale
        minor_root = self.get_degree(6)
        return Scale(minor_root, ScaleType.NATURAL_MINOR)
    
    @property
    def relative_major(self) -> Scale:
        """
        Get the relative major scale of a minor scale.
        
        Returns:
            The relative major scale
            
        Raises:
            ValueError: If this is not a natural minor scale
        """
        if self.scale_type != ScaleType.NATURAL_MINOR.value:
            raise ValueError("Only natural minor scales have relative majors")
        
        # Relative major root is at the 3rd degree of the minor scale
        major_root = self.get_degree(3)
        return Scale(major_root, ScaleType.MAJOR)
    
    @property
    def parallel_minor(self) -> Scale:
        """
        Get the parallel minor scale of a major scale.
        
        Returns:
            The parallel minor scale
            
        Raises:
            ValueError: If this is not a major scale
        """
        if self.scale_type != ScaleType.MAJOR.value:
            raise ValueError("Only major scales have parallel minors")
        
        # Parallel minor has the same root as the major scale
        return Scale(self.root, ScaleType.NATURAL_MINOR)
    
    @property
    def parallel_major(self) -> Scale:
        """
        Get the parallel major scale of a minor scale.
        
        Returns:
            The parallel major scale
            
        Raises:
            ValueError: If this is not a natural minor scale
        """
        if self.scale_type != ScaleType.NATURAL_MINOR.value:
            raise ValueError("Only natural minor scales have parallel majors")
        
        # Parallel major has the same root as the minor scale
        return Scale(self.root, ScaleType.MAJOR)


class ChordType(Enum):
    """Types of musical chords."""
    MAJOR = 'major'
    MINOR = 'minor'
    DIMINISHED = 'diminished'
    AUGMENTED = 'augmented'
    SUS2 = 'sus2'
    SUS4 = 'sus4'
    DOMINANT7 = 'dominant7'
    MAJOR7 = 'major7'
    MINOR7 = 'minor7'
    DIMINISHED7 = 'diminished7'
    HALF_DIMINISHED7 = 'half_diminished7'
    AUGMENTED7 = 'augmented7'
    MAJOR6 = 'major6'
    MINOR6 = 'minor6'
    ADD9 = 'add9'
    ADD11 = 'add11'


class Chord:
    """Represents musical chords."""
    
    # Defining common chord types by their interval structure
    CHORD_TYPES: ClassVar[Dict[str, List[int]]] = {
        ChordType.MAJOR.value: [0, 4, 7],                  # Major (1, 3, 5)
        ChordType.MINOR.value: [0, 3, 7],                  # Minor (1, b3, 5)
        ChordType.DIMINISHED.value: [0, 3, 6],             # Diminished (1, b3, b5)
        ChordType.AUGMENTED.value: [0, 4, 8],              # Augmented (1, 3, #5)
        ChordType.SUS2.value: [0, 2, 7],                   # Suspended 2nd (1, 2, 5)
        ChordType.SUS4.value: [0, 5, 7],                   # Suspended 4th (1, 4, 5)
        ChordType.DOMINANT7.value: [0, 4, 7, 10],          # Dominant 7th (1, 3, 5, b7)
        ChordType.MAJOR7.value: [0, 4, 7, 11],             # Major 7th (1, 3, 5, 7)
        ChordType.MINOR7.value: [0, 3, 7, 10],             # Minor 7th (1, b3, 5, b7)
        ChordType.DIMINISHED7.value: [0, 3, 6, 9],         # Diminished 7th (1, b3, b5, bb7)
        ChordType.HALF_DIMINISHED7.value: [0, 3, 6, 10],   # Half-diminished 7th (1, b3, b5, b7)
        ChordType.AUGMENTED7.value: [0, 4, 8, 10],         # Augmented 7th (1, 3, #5, b7)
        ChordType.MAJOR6.value: [0, 4, 7, 9],              # Major 6th (1, 3, 5, 6)
        ChordType.MINOR6.value: [0, 3, 7, 9],              # Minor 6th (1, b3, 5, 6)
        ChordType.ADD9.value: [0, 4, 7, 14],               # Add9 (1, 3, 5, 9)
        ChordType.ADD11.value: [0, 4, 7, 17],              # Add11 (1, 3, 5, 11)
    }
    
    # Mapping for chord symbols
    CHORD_SYMBOLS: ClassVar[Dict[str, str]] = {
        ChordType.MAJOR.value: '',
        ChordType.MINOR.value: 'm',
        ChordType.DIMINISHED.value: '°',
        ChordType.AUGMENTED.value: '+',
        ChordType.SUS2.value: 'sus2',
        ChordType.SUS4.value: 'sus4',
        ChordType.DOMINANT7.value: '7',
        ChordType.MAJOR7.value: 'maj7',
        ChordType.MINOR7.value: 'm7',
        ChordType.DIMINISHED7.value: '°7',
        ChordType.HALF_DIMINISHED7.value: 'ø7',
        ChordType.AUGMENTED7.value: '+7',
        ChordType.MAJOR6.value: '6',
        ChordType.MINOR6.value: 'm6',
        ChordType.ADD9.value: 'add9',
        ChordType.ADD11.value: 'add11',
    }
    
    def __init__(
        self, 
        root: Note, 
        chord_type: Union[str, ChordType] = ChordType.MAJOR, 
        inversion: int = 0
    ):
        """
        Initialize a Chord object.
        
        Args:
            root: The root note of the chord
            chord_type: The type of chord (default is ChordType.MAJOR)
            inversion: The inversion of the chord (0 for root position, 1 for first inversion, etc.)
            
        Raises:
            ValueError: If chord_type is invalid
        """
        if isinstance(chord_type, ChordType):
            chord_type = chord_type.value
            
        if chord_type not in self.CHORD_TYPES:
            raise ValueError(f"Invalid chord type: {chord_type}")
        
        self.root = root
        self.chord_type = chord_type
        self.intervals = self.CHORD_TYPES[chord_type]
        
        if inversion < 0:
            raise ValueError(f"Inversion must be non-negative, got {inversion}")
        
        self.inversion = inversion % len(self.intervals)
    
    @property
    def notes(self) -> List[Note]:
        """
        Get all notes in the chord.
        
        Returns:
            List of Note objects in the chord
        """
        base_notes = [self.root.transpose(interval) for interval in self.intervals]
        
        # Apply inversion by moving notes from the front to the back
        result = base_notes.copy()
        for i in range(self.inversion):
            note = result.pop(0)
            result.append(note.transpose(12))  # Move up an octave
            
        return result
    
    @property
    def symbol(self) -> str:
        """Get the chord symbol."""
        return f"{self.root.note_name}{self.CHORD_SYMBOLS[self.chord_type]}"
    
    def __str__(self) -> str:
        """String representation of the chord."""
        notes_str = ', '.join(str(note) for note in self.notes)
        inversion_text = "" if self.inversion == 0 else f" ({self.inversion} inversion)"
        return f"{self.symbol}{inversion_text}: {notes_str}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the chord."""
        return f"Chord({repr(self.root)}, '{self.chord_type}', {self.inversion})"
    
    @classmethod
    def from_scale_degree(
        cls, 
        scale: Scale, 
        degree: int, 
        chord_type: Optional[Union[str, ChordType]] = None
    ) -> Chord:
        """
        Create a chord from a scale degree.
        
        Args:
            scale: The scale to build the chord from
            degree: The scale degree to build the chord on (1-7)
            chord_type: Override the default chord type
            
        Returns:
            A new chord built on the specified scale degree
        """
        root = scale.get_degree(degree)
        
        # Determine default chord type based on scale and degree
        if chord_type is None:
            if scale.scale_type == ScaleType.MAJOR.value:
                # In major scale, chords are:
                # I: major, ii: minor, iii: minor, IV: major, V: major, vi: minor, vii°: diminished
                degree_chord_types = {
                    1: ChordType.MAJOR.value, 
                    2: ChordType.MINOR.value, 
                    3: ChordType.MINOR.value, 
                    4: ChordType.MAJOR.value,
                    5: ChordType.MAJOR.value, 
                    6: ChordType.MINOR.value, 
                    7: ChordType.DIMINISHED.value
                }
                chord_type = degree_chord_types[degree]
            elif scale.scale_type == ScaleType.NATURAL_MINOR.value:
                # In natural minor scale, chords are:
                # i: minor, ii°: diminished, III: major, iv: minor, v: minor, VI: major, VII: major
                degree_chord_types = {
                    1: ChordType.MINOR.value, 
                    2: ChordType.DIMINISHED.value, 
                    3: ChordType.MAJOR.value, 
                    4: ChordType.MINOR.value,
                    5: ChordType.MINOR.value, 
                    6: ChordType.MAJOR.value, 
                    7: ChordType.MAJOR.value
                }
                chord_type = degree_chord_types[degree]
            else:
                # Default to major for other scales
                chord_type = ChordType.MAJOR.value
                
        return cls(root, chord_type)


@dataclass
class ChordProgression:
    """Represents a chord progression."""
    
    scale: Scale
    degrees: List[int]
    name: Optional[str] = None
    
    # Common chord progressions by Roman numerals
    COMMON_PROGRESSIONS: ClassVar[Dict[str, List[int]]] = {
        'I-IV-V': [1, 4, 5],
        'I-V-vi-IV': [1, 5, 6, 4],
        'ii-V-I': [2, 5, 1],
        'I-vi-IV-V': [1, 6, 4, 5],
        'vi-IV-I-V': [6, 4, 1, 5],
        'I-IV-V-IV': [1, 4, 5, 4],
        'i-iv-v': [1, 4, 5],        # Minor equivalents
        'i-VI-III-VII': [1, 6, 3, 7],
        'i-VII-VI-VII': [1, 7, 6, 7]
    }
    
    def __post_init__(self):
        """Validate and initialize after construction."""
        # Generate the chords
        self._chords = [Chord.from_scale_degree(self.scale, degree) for degree in self.degrees]
    
    @classmethod
    def from_name(cls, scale: Scale, progression_name: str) -> ChordProgression:
        """
        Create a chord progression from a predefined name.
        
        Args:
            scale: The scale to build the progression from
            progression_name: A predefined progression name
            
        Returns:
            A ChordProgression object
            
        Raises:
            ValueError: If progression_name is invalid
        """
        if progression_name not in cls.COMMON_PROGRESSIONS:
            valid_names = list(cls.COMMON_PROGRESSIONS.keys())
            raise ValueError(f"Invalid progression name: {progression_name}. Must be one of {valid_names}")
            
        degrees = cls.COMMON_PROGRESSIONS[progression_name]
        return cls(scale, degrees, progression_name)
    
    @property
    def chords(self) -> List[Chord]:
        """Get the chords in the progression."""
        return self._chords
    
    def __str__(self) -> str:
        """String representation of the chord progression."""
        chord_symbols = [chord.symbol for chord in self.chords]
        name_str = f" ({self.name})" if self.name else ""
        return f"Progression in {self.scale.root.note_name} {self.scale.scale_type}{name_str}: {' - '.join(chord_symbols)}"
    
    def transpose(self, semitones: int) -> ChordProgression:
        """
        Transpose the entire progression by the specified number of semitones.
        
        Args:
            semitones: Number of semitones to transpose
            
        Returns:
            A new transposed chord progression
        """
        new_root = self.scale.root.transpose(semitones)
        new_scale = Scale(new_root, self.scale.scale_type)
        return ChordProgression(new_scale, self.degrees, self.name)


class RhythmPattern(Enum):
    """Common rhythmic patterns."""
    QUARTER_NOTES = 'quarter_notes'
    EIGHTH_NOTES = 'eighth_notes'
    BASIC_ROCK = 'basic_rock'
    BACKBEAT = 'backbeat'
    WALTZ = 'waltz'
    BOSSA_NOVA = 'bossa_nova'
    CLAVE_SON = 'clave_son'


class Rhythm:
    """Represents rhythmic patterns."""
    
    # Common rhythmic patterns (1 = beat, 0 = rest)
    PATTERNS: ClassVar[Dict[str, List[int]]] = {
        RhythmPattern.QUARTER_NOTES.value: [1, 1, 1, 1],                      # Simple quarter notes
        RhythmPattern.EIGHTH_NOTES.value: [1, 1, 1, 1, 1, 1, 1, 1],           # Simple eighth notes
        RhythmPattern.BASIC_ROCK.value: [1, 0, 1, 0, 1, 0, 1, 0],             # Basic rock beat
        RhythmPattern.BACKBEAT.value: [1, 0, 0, 1, 0, 1, 0, 0],               # Emphasizing the backbeat (2 and 4)
        RhythmPattern.WALTZ.value: [1, 0, 0, 1, 0, 0, 1, 0, 0],               # Basic 3/4 waltz
        RhythmPattern.BOSSA_NOVA.value: [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], # Bossa nova pattern
        RhythmPattern.CLAVE_SON.value: [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]         # Son clave pattern (3-2)
    }
    
    def __init__(
        self, 
        pattern: Union[str, RhythmPattern, List[int]] = RhythmPattern.QUARTER_NOTES, 
        tempo: int = 120
    ):
        """
        Initialize a Rhythm object.
        
        Args:
            pattern: Name of a predefined pattern, RhythmPattern enum, or a custom pattern list
            tempo: Tempo in beats per minute (default is 120)
            
        Raises:
            ValueError: If pattern is invalid
        """
        self.tempo = tempo
        
        if isinstance(pattern, list):
            self.pattern = pattern
            self.name = "Custom"
        else:
            if isinstance(pattern, RhythmPattern):
                pattern = pattern.value
                
            if pattern in self.PATTERNS:
                self.pattern = self.PATTERNS[pattern]
                self.name = pattern
            else:
                valid_patterns = list(self.PATTERNS.keys())
                raise ValueError(f"Invalid pattern: {pattern}. Must be one of {valid_patterns} or a list")
    
    def __str__(self) -> str:
        """String representation of the rhythm."""
        pattern_str = ' '.join(['X' if beat else '.' for beat in self.pattern])
        return f"{self.name} ({self.tempo} BPM): {pattern_str}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the rhythm."""
        return f"Rhythm('{self.name}', {self.tempo})"
    
    @property
    def duration(self) -> float:
        """
        Calculate the duration of the pattern in seconds.
        
        Returns:
            Duration in seconds
        """
        beats_per_second = self.tempo / 60
        return len(self.pattern) / beats_per_second
    
    def overlay(self, other: Rhythm) -> Rhythm:
        """
        Overlay another rhythm on top of this one.
        
        Args:
            other: Another rhythm pattern
            
        Returns:
            A new rhythm that combines both patterns
        """
        # Find the least common multiple of the pattern lengths
        pattern1, pattern2 = self.pattern, other.pattern
        lcm = self._lcm(len(pattern1), len(pattern2))
        
        # Extend patterns to the LCM length
        extended1 = self._extend_pattern(pattern1, lcm)
        extended2 = self._extend_pattern(pattern2, lcm)
        
        # Combine the patterns (1 OR 1 = 1, 0 OR 0 = 0, 1 OR 0 = 1)
        combined = [a or b for a, b in zip(extended1, extended2)]
        
        # Use the faster tempo of the two
        new_tempo = max(self.tempo, other.tempo)
        
        return Rhythm(combined, new_tempo)
    
    @staticmethod
    def _gcd(a: int, b: int) -> int:
        """Calculate the greatest common divisor."""
        while b:
            a, b = b, a % b
        return a
    
    @classmethod
    def _lcm(cls, a: int, b: int) -> int:
        """Calculate the least common multiple."""
        return a * b // cls._gcd(a, b)
    
    @staticmethod
    def _extend_pattern(pattern: List[int], length: int) -> List[int]:
        """Extend a pattern to a given length."""
        repeat_count = length // len(pattern)
        remainder = length % len(pattern)
        return pattern * repeat_count + pattern[:remainder]


def example() -> None:
    """Demonstrate the music theory library capabilities."""
    # Create a C major scale
    c4 = Note('C', 4)
    c_major = Scale(c4, ScaleType.MAJOR)
    print(f"C Major scale: {c_major}")
    
    # Create a chord progression
    progression = ChordProgression.from_name(c_major, 'I-IV-V-vi')
    print(f"Chord progression: {progression}")
    
    # Transpose to another key
    g_progression = progression.transpose(7)  # Up a perfect fifth
    print(f"Transposed progression: {g_progression}")
    
    # Create a rhythm
    basic_rhythm = Rhythm(RhythmPattern.BASIC_ROCK, tempo=100)
    print(f"Rhythm: {basic_rhythm}")
    
    # Calculate and display intervals
    e4 = Note('E', 4)
    g4 = Note('G', 4)
    interval1 = Interval.between(c4, e4)
    interval2 = Interval.between(e4, g4)
    print(f"Interval from C4 to E4: {interval1}")
    print(f"Interval from E4 to G4: {interval2}")


if __name__ == "__main__":
    example()
