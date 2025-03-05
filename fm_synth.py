"""
FM Synthesis Module

This module provides Frequency Modulation synthesis capabilities that
can be used with the music_theory.py module to generate instrument sounds.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Callable, Any
from enum import Enum
from dataclasses import dataclass
import sounddevice as sd
import soundfile as sf
import os
import psutil
from functools import lru_cache

# Try to import music_theory if available
try:
    from music_theory import Note
    _MUSIC_THEORY_AVAILABLE = True
except ImportError:
    _MUSIC_THEORY_AVAILABLE = False
    # Define minimal Note class for standalone use
    class Note:
        def __init__(self, note: str, octave: int = 4):
            self.note_name = note
            self.octave = octave
            
        @property
        def frequency(self) -> float:
            # Approximate implementation for standalone use
            note_values = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                          'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
            semitones_from_a4 = (self.octave - 4) * 12 + note_values.get(self.note_name, 0) - 9
            return 440 * (2 ** (semitones_from_a4 / 12))


class WaveformType(Enum):
    """Basic waveform types for synthesis."""
    SINE = "sine"
    SQUARE = "square"
    SAWTOOTH = "saw"
    TRIANGLE = "triangle"
    NOISE = "noise"


@dataclass
class ADSREnvelope:
    """ADSR (Attack, Decay, Sustain, Release) envelope for amplitude shaping."""
    attack: float  # seconds
    decay: float   # seconds
    sustain: float # level (0-1)
    release: float # seconds
    
    def get_envelope(self, duration: float, sample_rate: int) -> np.ndarray:
        """
        Generate an ADSR envelope as a numpy array.
        
        Args:
            duration: Total duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Numpy array containing the envelope values
        """
        total_samples = int(duration * sample_rate)
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)
        
        # Calculate sustain samples
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        # Ensure sustain samples is at least 0
        sustain_samples = max(0, sustain_samples)
        
        # Create segments
        attack_env = np.linspace(0, 1, attack_samples) if attack_samples > 0 else np.array([])
        decay_env = np.linspace(1, self.sustain, decay_samples) if decay_samples > 0 else np.array([])
        sustain_env = np.ones(sustain_samples) * self.sustain if sustain_samples > 0 else np.array([])
        release_env = np.linspace(self.sustain, 0, release_samples) if release_samples > 0 else np.array([])
        
        # Combine segments
        envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])
        
        # Ensure envelope is exactly the right length
        if len(envelope) > total_samples:
            envelope = envelope[:total_samples]
        elif len(envelope) < total_samples:
            envelope = np.pad(envelope, (0, total_samples - len(envelope)), 'constant')
            
        return envelope


class Waveform:
    """Basic waveform generator class."""
    
    @staticmethod
    @lru_cache(maxsize=32)
    def generate(
        wave_type: Union[WaveformType, str],
        frequency: float,
        duration: float,
        sample_rate: int = 44100,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate a waveform of the specified type.
        
        Args:
            wave_type: Type of waveform (sine, square, sawtooth, triangle, noise)
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Amplitude of the waveform (0-1)
            
        Returns:
            Numpy array containing the waveform
        """
        if isinstance(wave_type, str):
            wave_type = WaveformType(wave_type)
        
        # Memory protection - check available memory before allocating arrays
        try:
            # Calculate estimated memory needed (samples * 8 bytes for float64)
            num_samples = int(duration * sample_rate)
            est_memory_bytes = num_samples * 8
            
            # Get available system memory
            available_memory = psutil.virtual_memory().available
            
            # If we're going to use more than 70% of available memory, reduce quality
            if est_memory_bytes > available_memory * 0.7:
                # Options: reduce duration or sample rate
                if duration > 1.0:
                    # If duration is long enough, reduce it
                    old_duration = duration
                    duration = min(duration, available_memory * 0.5 / (sample_rate * 8))
                    num_samples = int(duration * sample_rate)
                    print(f"Warning: Reduced duration from {old_duration:.2f}s to {duration:.2f}s due to memory constraints")
                else:
                    # If duration is already short, reduce sample rate
                    old_sample_rate = sample_rate
                    sample_rate = max(8000, int(sample_rate * (available_memory * 0.5 / est_memory_bytes)))
                    num_samples = int(duration * sample_rate)
                    print(f"Warning: Reduced sample rate from {old_sample_rate}Hz to {sample_rate}Hz due to memory constraints")
        except Exception as e:
            # If memory check fails, still try to generate with original parameters
            print(f"Memory check error: {e}")
            num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        if wave_type == WaveformType.SINE:
            return amplitude * np.sin(2 * np.pi * frequency * t)
        
        elif wave_type == WaveformType.SQUARE:
            return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        
        elif wave_type == WaveformType.SAWTOOTH:
            return amplitude * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
        
        elif wave_type == WaveformType.TRIANGLE:
            return amplitude * (2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1)
        
        elif wave_type == WaveformType.NOISE:
            return amplitude * (2 * np.random.random(num_samples) - 1)
        
        else:
            raise ValueError(f"Unsupported waveform type: {wave_type}")


class FMOperator:
    """FM synthesis operator combining carrier and modulator."""
    
    def __init__(
        self,
        carrier_type: Union[WaveformType, str] = WaveformType.SINE,
        modulator_type: Union[WaveformType, str] = WaveformType.SINE,
        c_to_m_ratio: float = 1.0,
        modulation_index: float = 0.0,
        amplitude: float = 1.0,
        envelope: Optional[ADSREnvelope] = None
    ):
        """
        Initialize an FM Operator.
        
        Args:
            carrier_type: Carrier waveform type
            modulator_type: Modulator waveform type
            c_to_m_ratio: Carrier to modulator frequency ratio
            modulation_index: FM modulation index (intensity)
            amplitude: Amplitude scaling (0-1)
            envelope: ADSR envelope for amplitude shaping
        """
        self.carrier_type = carrier_type if isinstance(carrier_type, WaveformType) else WaveformType(carrier_type)
        self.modulator_type = modulator_type if isinstance(modulator_type, WaveformType) else WaveformType(modulator_type)
        self.c_to_m_ratio = c_to_m_ratio
        self.modulation_index = modulation_index
        self.amplitude = amplitude
        self.envelope = envelope or ADSREnvelope(0.01, 0.1, 0.7, 0.3)
    
    def generate(
        self, 
        frequency: float, 
        duration: float, 
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Generate FM synthesis waveform.
        
        Args:
            frequency: Base frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Numpy array with the generated waveform
        """
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Calculate carrier and modulator frequencies
        carrier_freq = frequency
        modulator_freq = frequency * self.c_to_m_ratio
        
        # Generate modulator with appropriate modulation index
        modulator = self.modulation_index * Waveform.generate(
            self.modulator_type, 
            modulator_freq, 
            duration, 
            sample_rate
        )
        
        # Generate carrier with frequency modulation
        carrier_phase = 2 * np.pi * carrier_freq * t + modulator
        carrier = self.amplitude * np.sin(carrier_phase)
        
        # Apply envelope
        envelope = self.envelope.get_envelope(duration, sample_rate)
        return carrier * envelope


class FMOperatorWithFeedback(FMOperator):
    """FM synthesis operator with feedback path for more complex timbres."""
    
    def __init__(
        self,
        feedback_amount: float = 0.0,
        **kwargs
    ):
        """
        Initialize an FM Operator with feedback.
        
        Args:
            feedback_amount: Amount of feedback (0-1)
            **kwargs: Other parameters passed to FMOperator
        """
        super().__init__(**kwargs)
        self.feedback_amount = feedback_amount
    
    def generate(
        self, 
        frequency: float, 
        duration: float, 
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Generate FM synthesis waveform with feedback.
        
        Args:
            frequency: Base frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Numpy array with the generated waveform
        """
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        output = np.zeros(num_samples)
        
        # Calculate carrier and modulator frequencies
        carrier_freq = frequency
        modulator_freq = frequency * self.c_to_m_ratio
        
        # Generate modulator
        modulator = self.modulation_index * Waveform.generate(
            self.modulator_type, 
            modulator_freq, 
            duration, 
            sample_rate
        )
        
        # Apply feedback through sample-by-sample processing
        for i in range(num_samples):
            # Apply feedback from previous output sample
            if i > 0:
                feedback = output[i-1] * self.feedback_amount
            else:
                feedback = 0
                
            # Calculate phase with modulation and feedback
            phase = 2 * np.pi * carrier_freq * t[i] + modulator[i] + feedback
            output[i] = self.amplitude * np.sin(phase)
            
        # Apply envelope
        envelope = self.envelope.get_envelope(duration, sample_rate)
        return output * envelope


class OperatorRoutingType(Enum):
    """Routing types for complex FM operator configurations."""
    PARALLEL = "parallel"  # Operators are summed together (default)
    SERIES = "series"      # Operators modulate each other in series
    STACKED = "stacked"    # Complex multi-level modulation


class FMVoice:
    """A voice combining multiple FM operators to create complex timbres."""
    
    def __init__(
        self, 
        operators: List[Tuple[FMOperator, float]],
        routing_type: Union[OperatorRoutingType, str] = OperatorRoutingType.PARALLEL
    ):
        """
        Initialize an FM Voice with multiple operators.
        
        Args:
            operators: List of tuples containing (operator, mix_level)
                      where mix_level is a value between 0 and 1
            routing_type: How operators are connected (parallel, series, stacked)
        """
        self.operators = operators
        self.routing_type = routing_type if isinstance(routing_type, OperatorRoutingType) else OperatorRoutingType(routing_type)
    
    def generate(
        self, 
        frequency: float, 
        duration: float, 
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Generate a complex FM sound using multiple operators.
        
        Args:
            frequency: Base frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Numpy array with the generated waveform
        """
        num_samples = int(duration * sample_rate)
        result = np.zeros(num_samples)
        
        if self.routing_type == OperatorRoutingType.PARALLEL:
            # Standard parallel routing (all operators are summed)
            for operator, mix_level in self.operators:
                operator_output = operator.generate(frequency, duration, sample_rate)
                result += operator_output * mix_level
                
        elif self.routing_type == OperatorRoutingType.SERIES:
            # Series routing (each operator modulates the next one)
            # Requires at least one operator
            if not self.operators:
                return result
                
            # Start with the first operator
            first_op, first_mix = self.operators[0]
            result = first_op.generate(frequency, duration, sample_rate) * first_mix
            
            # Each subsequent operator modulates the next using the 
            # output of the previous as a phase modulation
            for i in range(1, len(self.operators)):
                current_op, current_mix = self.operators[i]
                
                # Create carrier and modulator waveforms
                t = np.linspace(0, duration, num_samples, endpoint=False)
                
                # Use result as modulation
                modulation = result * current_op.modulation_index
                
                # Generate carrier with the modulation
                carrier_freq = frequency
                carrier_phase = 2 * np.pi * carrier_freq * t + modulation
                output = current_op.amplitude * np.sin(carrier_phase)
                
                # Apply envelope
                envelope = current_op.envelope.get_envelope(duration, sample_rate)
                output = output * envelope * current_mix
                
                result = output
                
        elif self.routing_type == OperatorRoutingType.STACKED:
            # Stacked modulation (complex multi-level network)
            # Requires at least 2 operators
            if len(self.operators) < 2:
                return result
                
            # In stacked mode, the first operator modulates the second,
            # the output of the second modulates the third, and so on
            t = np.linspace(0, duration, num_samples, endpoint=False)
            modulator_outputs = []
            
            # Generate first modulator
            first_op, _ = self.operators[0]
            first_output = first_op.generate(frequency, duration, sample_rate)
            modulator_outputs.append(first_output)
            
            # Generate each subsequent modulator, modulated by the previous ones
            for i in range(1, len(self.operators) - 1):
                current_op, _ = self.operators[i]
                prev_modulation = modulator_outputs[-1] * current_op.modulation_index
                
                # Generate carrier with the modulation
                carrier_freq = frequency * current_op.c_to_m_ratio
                carrier_phase = 2 * np.pi * carrier_freq * t + prev_modulation
                output = current_op.amplitude * np.sin(carrier_phase)
                
                # Apply envelope
                envelope = current_op.envelope.get_envelope(duration, sample_rate)
                output = output * envelope
                
                modulator_outputs.append(output)
            
            # Final carrier is modulated by all previous operators
            final_op, final_mix = self.operators[-1]
            
            # Sum all modulations weighted by their amplitudes
            total_modulation = np.zeros(num_samples)
            for i, mod_output in enumerate(modulator_outputs):
                _, mix_level = self.operators[i]
                total_modulation += mod_output * mix_level
            
            # Apply final modulation to carrier
            carrier_freq = frequency
            carrier_phase = 2 * np.pi * carrier_freq * t + total_modulation
            result = final_op.amplitude * np.sin(carrier_phase)
            
            # Apply envelope
            envelope = final_op.envelope.get_envelope(duration, sample_rate)
            result = result * envelope * final_mix
            
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(result))
        if max_amplitude > 1.0:
            result = result / max_amplitude
            
        return result


class Instrument:
    """Base class for FM synthesized instruments."""
    
    def __init__(self, voice: FMVoice):
        """
        Initialize an instrument with an FM voice.
        
        Args:
            voice: The FMVoice to use for sound generation
        """
        self.voice = voice
    
    def play_note(
        self, 
        note: Union[Note, str], 
        duration: float, 
        amplitude: float = 0.5,
        sample_rate: int = 44100,
        output_file: Optional[str] = None
    ) -> None:
        """
        Play or save a note using the instrument voice.
        
        Args:
            note: Note object or note name string (e.g., 'C4')
            duration: Duration in seconds
            amplitude: Volume (0-1)
            sample_rate: Sample rate in Hz
            output_file: Path to output .wav file. If None, plays the audio instead.
        """
        # Convert string note to Note object if needed
        if isinstance(note, str):
            if _MUSIC_THEORY_AVAILABLE:
                # Parse string like 'C4' to Note object
                note_name = ''.join(c for c in note if not c.isdigit())
                octave = int(''.join(c for c in note if c.isdigit()))
                note = Note(note_name, octave)
            else:
                # Simple parser for standalone mode
                note_name = ''.join(c for c in note if not c.isdigit())
                octave = int(''.join(c for c in note if c.isdigit() or c == '-'))
                note = Note(note_name, octave)
        
        # Check available memory before generating waveform
        try:
            # Calculate estimated memory needed (samples * 8 bytes for float64)
            est_memory_bytes = int(duration * sample_rate * 8)
            available_memory = psutil.virtual_memory().available
            
            if est_memory_bytes > available_memory * 0.7:  # Using 70% of available memory is risky
                # Reduce quality or duration if memory is low
                if duration > 1.0:
                    # Reduce duration if possible
                    duration = min(duration, available_memory * 0.5 / (sample_rate * 8))
                    print(f"Warning: Reduced note duration to {duration:.2f}s due to memory constraints")
                else:
                    # Or reduce sample rate if duration is already short
                    old_sample_rate = sample_rate
                    sample_rate = max(8000, int(sample_rate * (available_memory * 0.5 / est_memory_bytes)))
                    print(f"Warning: Reduced sample rate to {sample_rate}Hz due to memory constraints")
        except Exception as e:
            print(f"Memory check error: {e}")
        
        # Generate waveform
        waveform = self.voice.generate(note.frequency, duration, sample_rate)
        
        # Scale by amplitude
        waveform = waveform * amplitude
        
        # Save to file or play
        if output_file:
            sf.write(output_file, waveform, sample_rate)
        else:
            # Play audio
            sd.play(waveform, sample_rate)
            sd.wait()
    
    def render_note(
        self, 
        note: Union[Note, str], 
        duration: float, 
        amplitude: float = 0.5,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Render a note to a numpy array without playing it.
        
        Args:
            note: Note object or note name string (e.g., 'C4')
            duration: Duration in seconds
            amplitude: Volume (0-1)
            sample_rate: Sample rate in Hz
            
        Returns:
            Numpy array with the rendered waveform
        """
        # Convert string note to Note object if needed
        if isinstance(note, str):
            if _MUSIC_THEORY_AVAILABLE:
                # Parse string like 'C4' to Note object
                note_name = ''.join(c for c in note if not c.isdigit())
                octave = int(''.join(c for c in note if c.isdigit()))
                note = Note(note_name, octave)
            else:
                # Simple parser for standalone mode
                note_name = ''.join(c for c in note if not c.isdigit())
                octave = int(''.join(c for c in note if c.isdigit() or c == '-'))
                note = Note(note_name, octave)
        
        # Check available memory before generating waveform
        try:
            # Calculate estimated memory needed (samples * 8 bytes for float64)
            est_memory_bytes = int(duration * sample_rate * 8)
            available_memory = psutil.virtual_memory().available
            
            if est_memory_bytes > available_memory * 0.7:  # Using 70% of available memory is risky
                # Adjust settings based on available memory
                if duration > 1.0:
                    # Reduce duration if possible
                    duration = min(duration, available_memory * 0.5 / (sample_rate * 8))
                    print(f"Warning: Reduced note duration to {duration:.2f}s due to memory constraints")
                else:
                    # Or reduce sample rate if duration is already short
                    sample_rate = max(8000, int(sample_rate * (available_memory * 0.5 / est_memory_bytes)))
                    print(f"Warning: Reduced sample rate to {sample_rate}Hz due to memory constraints")
        except Exception as e:
            print(f"Memory check error: {e}")
        
        # Generate waveform
        waveform = self.voice.generate(note.frequency, duration, sample_rate)
        
        # Scale by amplitude
        return waveform * amplitude


class InstrumentPresets:
    """Factory for creating preset instruments using FM synthesis."""
    
    @staticmethod
    def piano() -> Instrument:
        """Create a piano-like FM instrument."""
        # Piano-like settings with two operators
        op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=2.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.005, 0.1, 0.7, 0.3)
        )
        
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=0.5,
            modulation_index=1.0,
            amplitude=0.5,
            envelope=ADSREnvelope(0.001, 0.1, 0.5, 0.5)
        )
        
        voice = FMVoice([(op1, 0.7), (op2, 0.3)])
        return Instrument(voice)
    
    @staticmethod
    def electric_piano() -> Instrument:
        """Create an electric piano FM instrument with complex operator routing."""
        # Bell component for the attack
        op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=3.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.001, 0.5, 0.2, 0.8)
        )
        
        # Tine component
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.TRIANGLE,
            c_to_m_ratio=14.0,  # Creates the characteristic metallic timbre
            modulation_index=0.3,
            amplitude=0.7,
            envelope=ADSREnvelope(0.001, 0.3, 0.1, 0.7)
        )
        
        # Fundamental tone
        op3 = FMOperatorWithFeedback(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=1.0,
            amplitude=0.9,
            envelope=ADSREnvelope(0.005, 0.1, 0.7, 0.2),
            feedback_amount=0.15  # Slight feedback for added richness
        )
        
        # Combine operators using series routing for more complex interactions
        voice = FMVoice(
            [(op1, 0.4), (op2, 0.15), (op3, 0.45)],
            routing_type=OperatorRoutingType.PARALLEL
        )
        
        return Instrument(voice)
    
    @staticmethod
    def brass() -> Instrument:
        """Create a brass-like FM instrument."""
        # Brass-like settings
        op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=4.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.1, 0.1, 0.8, 0.2)
        )
        
        voice = FMVoice([(op1, 1.0)])
        return Instrument(voice)
    
    @staticmethod
    def saxophone() -> Instrument:
        """Create a saxophone-like FM instrument."""
        # Saxophone body resonance
        op1 = FMOperatorWithFeedback(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=3.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.08, 0.2, 0.7, 0.3),
            feedback_amount=0.3  # Feedback adds characteristic saxophone edge
        )
        
        # Breath noise and harmonics
        op2 = FMOperator(
            carrier_type=WaveformType.TRIANGLE,
            modulator_type=WaveformType.NOISE,
            c_to_m_ratio=2.0,
            modulation_index=0.3,
            amplitude=0.2,
            envelope=ADSREnvelope(0.05, 0.1, 0.6, 0.2)
        )
        
        # Reed vibration
        op3 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SQUARE,
            c_to_m_ratio=3.0,
            modulation_index=1.0,
            amplitude=0.4,
            envelope=ADSREnvelope(0.03, 0.15, 0.5, 0.25)
        )
        
        # Combine with stacked routing for complex harmonics
        voice = FMVoice(
            [(op1, 0.6), (op2, 0.2), (op3, 0.2)],
            routing_type=OperatorRoutingType.STACKED
        )
        
        return Instrument(voice)
    
    @staticmethod
    def bell() -> Instrument:
        """Create a bell-like FM instrument."""
        # Bell-like settings with inharmonic partials
        op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=2.0,
            modulation_index=5.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.001, 0.5, 0.3, 2.0)
        )
        
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=3.5,
            modulation_index=2.0,
            amplitude=0.5,
            envelope=ADSREnvelope(0.001, 1.0, 0.2, 3.0)
        )
        
        voice = FMVoice([(op1, 0.6), (op2, 0.4)])
        return Instrument(voice)
    
    @staticmethod
    def bass() -> Instrument:
        """Create a bass-like FM instrument."""
        # Bass-like settings
        op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.TRIANGLE,
            c_to_m_ratio=1.0,
            modulation_index=2.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.01, 0.2, 0.8, 0.3)
        )
        
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SQUARE,
            c_to_m_ratio=0.5,
            modulation_index=0.5,
            amplitude=0.3,
            envelope=ADSREnvelope(0.01, 0.1, 0.4, 0.5)
        )
        
        voice = FMVoice([(op1, 0.8), (op2, 0.2)])
        return Instrument(voice)
    
    @staticmethod
    def synth_strings() -> Instrument:
        """Create a string ensemble FM instrument."""
        # Main string tone
        op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=1.0,
            amplitude=1.0,
            envelope=ADSREnvelope(0.2, 0.3, 0.8, 0.5)  # Slow attack for strings
        )
        
        # String chorus/ensemble effect
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.TRIANGLE,
            c_to_m_ratio=1.01,  # Very slightly detuned
            modulation_index=0.3,
            amplitude=0.8,
            envelope=ADSREnvelope(0.3, 0.4, 0.7, 0.6)
        )
        
        # Second detuned oscillator for width
        op3 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=0.99,  # Slightly detuned the other way
            modulation_index=0.3,
            amplitude=0.7,
            envelope=ADSREnvelope(0.25, 0.35, 0.7, 0.55)
        )
        
        # Bow noise component
        op4 = FMOperator(
            carrier_type=WaveformType.NOISE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=0.1,
            amplitude=0.1,
            envelope=ADSREnvelope(0.1, 0.2, 0.3, 0.4)
        )
        
        voice = FMVoice([
            (op1, 0.4), 
            (op2, 0.3), 
            (op3, 0.25), 
            (op4, 0.05)
        ])
        
        return Instrument(voice)
    
    @staticmethod
    def flute() -> Instrument:
        """Create a flute-like FM instrument."""
        # Flute-like settings
        op1 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=1.0,
            modulation_index=0.2,
            amplitude=1.0,
            envelope=ADSREnvelope(0.1, 0.1, 0.9, 0.3)
        )
        
        op2 = FMOperator(
            carrier_type=WaveformType.SINE,
            modulator_type=WaveformType.SINE,
            c_to_m_ratio=2.0,
            modulation_index=0.1,
            amplitude=0.2,
            envelope=ADSREnvelope(0.05, 0.2, 0.6, 0.1)
        )
        
        voice = FMVoice([(op1, 0.9), (op2, 0.1)])
        return Instrument(voice)


class AudioEffect:
    """Base class for audio effects processing."""
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process an audio signal.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        return audio  # Base implementation just returns the input


class DelayEffect(AudioEffect):
    """Delay/echo effect for audio processing."""
    
    def __init__(
        self, 
        delay_time: float = 0.3, 
        feedback: float = 0.4, 
        mix: float = 0.5
    ):
        """
        Initialize a delay effect.
        
        Args:
            delay_time: Delay time in seconds
            feedback: Feedback amount (0-1)
            mix: Wet/dry mix ratio (0-1)
        """
        self.delay_time = delay_time
        self.feedback = max(0.0, min(0.99, feedback))  # Clip to prevent instability
        self.mix = max(0.0, min(1.0, mix))  # Clip to 0-1
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply a delay effect to an audio signal.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal with delay effect
        """
        # Calculate delay in samples
        delay_samples = int(self.delay_time * sample_rate)
        
        # Create output buffer with extra room for delay tails
        output_length = len(audio) + int(delay_samples * 4)  # Extra room for delay tail
        output = np.zeros(output_length)
        
        # Copy dry signal
        output[:len(audio)] = audio * (1 - self.mix)
        
        # Wet signal starts as a copy of input
        wet = np.copy(audio)
        
        # Apply delay with feedback
        for i in range(len(audio)):
            if i + delay_samples < output_length:
                output[i + delay_samples] += wet[i] * self.mix
                
        # Apply feedback - multiple delay taps with decreasing volume
        for fb_level in range(1, 5):  # Apply several feedback iterations
            feedback_amount = self.feedback ** fb_level  # Decrease with each iteration
            if feedback_amount < 0.01:  # Stop when feedback gets too quiet
                break
                
            for i in range(len(audio)):
                delay_pos = i + delay_samples * (fb_level + 1)
                if delay_pos < output_length:
                    output[delay_pos] += wet[i] * self.mix * feedback_amount
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 1.0:
            output = output / max_amplitude
            
        return output


class ReverbEffect(AudioEffect):
    """Simple reverb effect using multiple delayed signals."""
    
    def __init__(
        self, 
        room_size: float = 0.6, 
        damping: float = 0.5, 
        mix: float = 0.3
    ):
        """
        Initialize a reverb effect.
        
        Args:
            room_size: Size of the simulated room (0-1)
            damping: Damping of high frequencies (0-1)
            mix: Wet/dry mix ratio (0-1)
        """
        self.room_size = max(0.1, min(0.99, room_size))  # Clip to prevent instability
        self.damping = max(0.0, min(0.99, damping))  # Clip to 0-1
        self.mix = max(0.0, min(1.0, mix))  # Clip to 0-1
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply a reverb effect to an audio signal.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal with reverb effect
        """
        # Create multiple delay lines with different times and feedback
        num_delays = 8
        
        # Base reverb time calculation - longer for larger rooms
        base_delay = 0.05 + self.room_size * 0.1
        
        # Create output with room for reverb tail
        tail_seconds = 2.0 + self.room_size * 3.0
        output_length = len(audio) + int(tail_seconds * sample_rate)
        output = np.zeros(output_length)
        
        # Copy dry signal
        output[:len(audio)] = audio * (1 - self.mix)
        
        # Apply multiple delay lines for a more realistic reverb
        for i in range(num_delays):
            # Adjust delay time and feedback for each delay line
            delay_time = base_delay * (0.7 + 0.3 * (i / num_delays))
            feedback = self.room_size * (0.4 + 0.6 * (1 - self.damping) * ((num_delays - i) / num_delays))
            
            # Apply delay with decreasing feedback
            delay_samples = int(delay_time * sample_rate)
            
            # Add initial delay
            for j in range(len(audio)):
                if j + delay_samples < output_length:
                    output[j + delay_samples] += audio[j] * self.mix * (1.0 / num_delays)
            
            # Apply feedback
            for fb in range(1, 10):  # Multiple feedback iterations
                fb_amount = feedback ** fb
                if fb_amount < 0.01:  # Stop when feedback gets too quiet
                    break
                    
                for j in range(len(audio)):
                    delay_pos = j + delay_samples * (fb + 1)
                    if delay_pos < output_length:
                        # Apply damping as low-pass filter effect by making each feedback
                        # tap have less high frequency content
                        dampened = audio[j] * fb_amount * (1 - self.damping * (fb / 10))
                        output[delay_pos] += dampened * self.mix * (1.0 / num_delays)
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 1.0:
            output = output / max_amplitude
            
        return output


class ChorusEffect(AudioEffect):
    """Chorus effect that creates a richer sound by adding slightly delayed and modulated copies of the signal."""
    
    def __init__(
        self, 
        rate: float = 0.5, 
        depth: float = 0.7, 
        mix: float = 0.5
    ):
        """
        Initialize a chorus effect.
        
        Args:
            rate: Rate of the LFO in Hz (0.1-5)
            depth: Depth of the effect (0-1)
            mix: Wet/dry mix ratio (0-1)
        """
        self.rate = max(0.1, min(5.0, rate))  # Clip rate to reasonable range
        self.depth = max(0.0, min(1.0, depth))  # Clip to 0-1
        self.mix = max(0.0, min(1.0, mix))  # Clip to 0-1
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply a chorus effect to an audio signal.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal with chorus effect
        """
        # Create output buffer
        output = np.zeros(len(audio))
        
        # Add dry signal
        output += audio * (1 - self.mix)
        
        # Create 3 chorus voices with different modulation phases
        num_voices = 3
        
        for voice in range(num_voices):
            # Create a modulation LFO with different phase for each voice
            phase = voice * (2 * np.pi / num_voices)
            lfo = np.sin(2 * np.pi * self.rate * np.linspace(0, len(audio) / sample_rate, len(audio)) + phase)
            
            # Calculate delay time for each sample based on LFO
            # Base delay of 20ms + modulation
            base_delay_ms = 20
            max_depth_ms = 10
            
            for i in range(len(audio)):
                # Calculate delay in samples for this point in time
                delay_ms = base_delay_ms + lfo[i] * self.depth * max_depth_ms
                delay_samples = int((delay_ms / 1000) * sample_rate)
                
                # Get delayed sample (with bounds checking)
                source_idx = i - delay_samples
                if source_idx >= 0 and source_idx < len(audio):
                    output[i] += audio[source_idx] * (self.mix / num_voices)
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 1.0:
            output = output / max_amplitude
            
        return output


class EffectChain:
    """A chain of audio effects that can be applied sequentially."""
    
    def __init__(self, effects: List[AudioEffect] = None):
        """
        Initialize an effect chain.
        
        Args:
            effects: List of audio effects to apply in sequence
        """
        self.effects = effects or []
    
    def add_effect(self, effect: AudioEffect) -> None:
        """
        Add an effect to the chain.
        
        Args:
            effect: The effect to add
        """
        self.effects.append(effect)
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through the entire effect chain.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        result = audio
        for effect in self.effects:
            result = effect.process(result, sample_rate)
        return result


class Sequencer:
    """Simple sequencer for playing or saving sequences of notes with specified instruments."""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize a sequencer.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.tracks: List[np.ndarray] = []
        self.track_durations: List[float] = []
        self.track_effects: List[Optional[EffectChain]] = []
        self._memory_efficient = False  # Flag for memory-efficient processing mode
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
    
    def add_note(
        self, 
        instrument: Instrument, 
        note: Union[Note, str], 
        start_time: float, 
        duration: float, 
        amplitude: float = 0.5
    ) -> None:
        """
        Add a note to the sequence.
        
        Args:
            instrument: The instrument to play the note
            note: The note to play
            start_time: Start time in seconds
            duration: Duration in seconds
            amplitude: Volume (0-1)
        """
        # Render the note
        note_waveform = instrument.render_note(note, duration, amplitude, self.sample_rate)
        
        # Calculate track duration
        note_end_time = start_time + duration
        
        # Find or create a track that's long enough
        track_idx = None
        for i, track_duration in enumerate(self.track_durations):
            # Check both track duration and physical array length
            required_samples = int(note_end_time * self.sample_rate)
            if track_duration >= note_end_time and len(self.tracks[i]) >= required_samples:
                track_idx = i
                break
        
        if track_idx is None:
            # Create a new track
            track_length = int(note_end_time * self.sample_rate)
            # Ensure the track is at least as large as the note waveform
            track_length = max(track_length, len(note_waveform))
            self.tracks.append(np.zeros(track_length))
            self.track_durations.append(note_end_time)
            self.track_effects.append(None)  # No effects by default
            track_idx = len(self.tracks) - 1
        
        # Add the note to the track
        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + len(note_waveform)
        
        # Make sure the track is long enough
        if end_sample > len(self.tracks[track_idx]):
            # Extend the track
            self.tracks[track_idx] = np.pad(
                self.tracks[track_idx], 
                (0, end_sample - len(self.tracks[track_idx])), 
                'constant'
            )
            self.track_durations[track_idx] = end_sample / self.sample_rate
        
        # Safety check for slice length
        actual_samples = min(len(note_waveform), end_sample - start_sample)
        
        # Mix the note into the track
        self.tracks[track_idx][start_sample:start_sample + actual_samples] += note_waveform[:actual_samples]
    
    def apply_effects_to_track(self, track_idx: int, effect_chain: EffectChain) -> None:
        """
        Apply effects to a specific track.
        
        Args:
            track_idx: Index of the track to apply effects to
            effect_chain: Chain of effects to apply
        """
        if track_idx < 0 or track_idx >= len(self.tracks):
            raise ValueError(f"Invalid track index: {track_idx}")
        
        self.track_effects[track_idx] = effect_chain
    
    def render(self) -> np.ndarray:
        """
        Render all tracks into a single audio array, applying effects if specified.
        
        Returns:
            Numpy array with the rendered sequence
        """
        if not self.tracks:
            return np.array([])
        
        # Check available memory before processing
        try:
            # Estimate memory requirements
            mem_per_track = sum(len(track) * 8 for track in self.tracks)  # 8 bytes per float64
            available_memory = psutil.virtual_memory().available
            
            # Calculate available memory in GB for easier display
            available_memory_gb = available_memory / (1024 ** 3)
            required_memory_gb = mem_per_track / (1024 ** 3)
            
            # Only do drastic memory optimization if we're really low on memory
            # Your system has 32GB, so we can be more generous with memory usage
            memory_threshold = 0.9  # Use 90% of available memory as threshold
            
            if mem_per_track > available_memory * memory_threshold:
                print(f"Warning: Low memory for rendering. Available: {available_memory_gb:.2f}GB, Required: {required_memory_gb:.2f}GB")
                print("Attempting to optimize...")
                
                # First, try optimizing effect processing instead of removing tracks
                self._memory_efficient = True  # Flag to use more memory-efficient effect processing
                
                # If we still need to reduce tracks, do so intelligently
                if len(self.tracks) > 3 and mem_per_track > available_memory * 0.8:
                    # Calculate how many tracks we can keep based on memory
                    memory_per_track = mem_per_track / len(self.tracks)
                    max_tracks_possible = int((available_memory * 0.75) / memory_per_track)
                    
                    # Ensure we keep at least 3 tracks for better sound quality
                    preserved_tracks = max(3, max_tracks_possible)
                    preserved_tracks = min(preserved_tracks, len(self.tracks))  # Don't exceed actual track count
                    
                    # Keep these tracks in order of importance:
                    # 1. Bass and drums (rhythm section)
                    # 2. Lead melody instruments
                    # 3. Supporting instruments
                    print(f"Memory optimization: Processing {preserved_tracks} of {len(self.tracks)} tracks")
                    
                    # Keep the most important tracks
                    self.tracks = self.tracks[:preserved_tracks]
                    self.track_durations = self.track_durations[:preserved_tracks]
                    self.track_effects = self.track_effects[:preserved_tracks]
                    
                # DO NOT truncate tracks - this would make the song too short
                # Instead, maintain original durations and handle length differences during mixing
        except Exception as e:
            print(f"Memory check error: {e}")
        
        # We need to use the maximum duration rather than the maximum length
        # This allows us to respect the full musical structure
        max_duration = max(self.track_durations) if self.track_durations else 0
        
        # Calculate the expected length in samples
        expected_length = int(max_duration * self.sample_rate)
        
        # Process each track with its effects
        processed_tracks = []
        track_info = []  # Store debugging info
        
        for i, track in enumerate(self.tracks):
            try:
                # Apply effects if specified
                if self.track_effects[i] is not None:
                    if self._memory_efficient and len(track) > 100000:  # For long tracks
                        # Memory-efficient processing: process in chunks
                        chunk_size = 44100 * 5  # 5 seconds at a time
                        num_chunks = (len(track) + chunk_size - 1) // chunk_size
                        processed = np.zeros(len(track) + int(len(track) * 0.3))  # Allow 30% growth for effects
                        
                        print(f"Processing track {i} in {num_chunks} chunks to save memory")
                        
                        for chunk in range(num_chunks):
                            start = chunk * chunk_size
                            end = min(start + chunk_size, len(track))
                            
                            # Process this chunk
                            chunk_data = track[start:end]
                            processed_chunk = self.track_effects[i].process(chunk_data, self.sample_rate)
                            
                            # Determine where to put this in the output
                            # For the first chunk, start at 0
                            # For later chunks, overlap slightly with previous chunk
                            if chunk == 0:
                                out_start = 0
                            else:
                                # Overlap by 0.1 seconds (helps smooth transitions)
                                overlap = int(0.1 * self.sample_rate)
                                out_start = start - overlap
                            
                            out_end = out_start + len(processed_chunk)
                            
                            # Ensure the output buffer is large enough
                            if out_end > len(processed):
                                processed = np.pad(processed, (0, out_end - len(processed)), 'constant')
                            
                            # Mix in this chunk (simple crossfade for overlapping regions)
                            if chunk > 0:
                                # Apply crossfade in the overlap region
                                fade_in = np.linspace(0, 1, min(overlap, len(processed_chunk)))
                                fade_out = np.linspace(1, 0, min(overlap, len(processed_chunk)))
                                
                                # Apply fades only to the overlap region
                                overlap_region = min(overlap, len(processed_chunk))
                                processed[out_start:out_start+overlap_region] *= fade_out[:overlap_region]
                                processed_chunk[:overlap_region] *= fade_in[:overlap_region]
                            
                            # Add the processed chunk to the output
                            processed[out_start:out_start+len(processed_chunk)] += processed_chunk
                        
                        # Trim any excess padding we added
                        processed = processed[:int(len(track) * 1.2)]
                    else:
                        # Standard processing for shorter tracks or when memory is not a concern
                        processed = self.track_effects[i].process(track, self.sample_rate)
                else:
                    processed = track
                
                track_info.append({
                    "index": i,
                    "original_length": len(track),
                    "processed_length": len(processed),
                    "duration": self.track_durations[i]
                })
                
                processed_tracks.append(processed)
            except Exception as e:
                print(f"Error processing track {i}: {e}")
                # Add silence instead of failing completely
                processed_tracks.append(np.zeros(expected_length))
        
        # Print debug info about track lengths
        print(f"Expected total duration: {max_duration:.2f}s ({expected_length} samples)")
        for info in track_info:
            print(f"Track {info['index']}: {info['duration']:.2f}s, {info['original_length']} → {info['processed_length']} samples")
        
        # Mix all tracks - ensure minimum length of at least 3 seconds to avoid very short outputs
        min_output_samples = max(expected_length, int(3.0 * self.sample_rate))
        result = np.zeros(min_output_samples)
        
        for i, track in enumerate(processed_tracks):
            # Ensure each track is properly sized before adding
            if len(track) < min_output_samples:
                # Pad shorter tracks with silence
                track = np.pad(track, (0, min_output_samples - len(track)), 'constant')
            elif len(track) > min_output_samples:
                # If track is longer than our buffer, expand the buffer
                old_size = len(result)
                result = np.pad(result, (0, len(track) - old_size), 'constant')
                min_output_samples = len(result)
            
            # Add the properly sized track
            result[:len(track)] += track[:len(result)]
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(result))
        if max_amplitude > 1.0:
            result = result / max_amplitude
        
        return result
    
    def play(self) -> None:
        """Play the rendered sequence."""
        rendered = self.render()
        if len(rendered) > 0:
            sd.play(rendered, self.sample_rate)
            sd.wait()
    
    def save_to_wav(self, filename: str = "output/track.wav") -> str:
        """
        Save the rendered sequence to a WAV file.
        
        Args:
            filename: Path to save the WAV file
            
        Returns:
            The absolute path to the saved file
        """
        rendered = self.render()
        if len(rendered) > 0:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save the file
            sf.write(filename, rendered, self.sample_rate)
            abs_path = os.path.abspath(filename)
            print(f"Saved audio to: {abs_path}")
            return abs_path
        else:
            print("No audio to save")
            return ""
    
    def clear(self) -> None:
        """Clear all tracks."""
        self.tracks = []
        self.track_durations = []
        self.track_effects = []


def example():
    """Demonstrate the FM synthesis module capabilities."""
    if not _MUSIC_THEORY_AVAILABLE:
        print("Note: music_theory module not found, using standalone mode")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Create instruments with our new features
    piano = InstrumentPresets.piano()
    ep = InstrumentPresets.electric_piano()  # New enhanced electric piano
    sax = InstrumentPresets.saxophone()      # New saxophone with feedback
    strings = InstrumentPresets.synth_strings()  # New string ensemble
    
    print("Generating and saving C major chord with piano to WAV files...")
    piano.play_note("C4", 0.5, output_file="output/piano_C4.wav")
    piano.play_note("E4", 0.5, output_file="output/piano_E4.wav")
    piano.play_note("G4", 0.5, output_file="output/piano_G4.wav")
    
    print("Generating and saving C minor chord with electric piano to WAV files...")
    ep.play_note("C4", 0.5, output_file="output/ep_C4.wav")
    ep.play_note("Eb4", 0.5, output_file="output/ep_Eb4.wav")
    ep.play_note("G4", 0.5, output_file="output/ep_G4.wav")
    
    # Demo saxophone with complex operator routing
    print("Generating C major scale with saxophone to WAV files...")
    for note in ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]:
        sax.play_note(note, 0.3, output_file=f"output/sax_{note}.wav")
    
    # Create a sequencer and generate a melody with effects
    print("Creating a sequence with effects processing...")
    seq = Sequencer()
    
    # Create a piano track (track 0)
    seq.add_note(piano, "C4", 0.0, 0.5)
    seq.add_note(piano, "E4", 0.5, 0.5)
    seq.add_note(piano, "G4", 1.0, 0.5)
    seq.add_note(piano, "C5", 1.5, 1.0)
    
    # Create a saxophone track (track 1)
    seq.add_note(sax, "G3", 0.25, 0.5, 0.3)
    seq.add_note(sax, "C4", 0.75, 0.5, 0.3)
    seq.add_note(sax, "D4", 1.25, 0.5, 0.3)
    seq.add_note(sax, "G4", 1.75, 1.0, 0.3)
    
    # Create a string pad track (track 2)
    seq.add_note(strings, "C3", 0.0, 2.5, 0.2)
    seq.add_note(strings, "G3", 0.0, 2.5, 0.2)
    seq.add_note(strings, "E3", 0.0, 2.5, 0.2)
    
    # Apply effects to tracks
    
    # Reverb for piano
    piano_effects = EffectChain()
    piano_effects.add_effect(ReverbEffect(room_size=0.3, mix=0.2))
    seq.apply_effects_to_track(0, piano_effects)
    
    # Delay for saxophone
    sax_effects = EffectChain()
    sax_effects.add_effect(DelayEffect(delay_time=0.25, feedback=0.3, mix=0.3))
    seq.apply_effects_to_track(1, sax_effects)
    
    # Chorus for strings
    string_effects = EffectChain()
    string_effects.add_effect(ChorusEffect(rate=0.8, depth=0.5, mix=0.4))
    string_effects.add_effect(ReverbEffect(room_size=0.8, damping=0.6, mix=0.3))
    seq.apply_effects_to_track(2, string_effects)
    
    # Save the sequence to a WAV file
    print("Saving the full sequence with effects to WAV file...")
    seq.save_to_wav("output/full_sequence.wav")
    
    # Demonstrate operator routing types
    print("\nDemonstrating different operator routing types:")
    
    # Create an instrument with operators in parallel (default)
    print("Creating an instrument with parallel routing...")
    op1 = FMOperator(
        carrier_type=WaveformType.SINE,
        modulator_type=WaveformType.SINE,
        c_to_m_ratio=1.0,
        modulation_index=2.0,
        amplitude=1.0,
        envelope=ADSREnvelope(0.01, 0.1, 0.8, 0.2)
    )
    
    op2 = FMOperator(
        carrier_type=WaveformType.SINE,
        modulator_type=WaveformType.TRIANGLE,
        c_to_m_ratio=2.0,
        modulation_index=1.0,
        amplitude=0.5,
        envelope=ADSREnvelope(0.01, 0.2, 0.6, 0.3)
    )
    
    # Create the same instrument with different routing types
    voice_parallel = FMVoice([(op1, 0.6), (op2, 0.4)], routing_type=OperatorRoutingType.PARALLEL)
    voice_series = FMVoice([(op1, 0.6), (op2, 0.4)], routing_type=OperatorRoutingType.SERIES)
    voice_stacked = FMVoice([(op1, 0.6), (op2, 0.4)], routing_type=OperatorRoutingType.STACKED)
    
    inst_parallel = Instrument(voice_parallel)
    inst_series = Instrument(voice_series)
    inst_stacked = Instrument(voice_stacked)
    
    print("Saving notes with different operator routing to WAV files...")
    inst_parallel.play_note("C4", 1.0, 0.5, output_file="output/parallel_routing.wav")
    inst_series.play_note("C4", 1.0, 0.5, output_file="output/series_routing.wav")
    inst_stacked.play_note("C4", 1.0, 0.5, output_file="output/stacked_routing.wav")
    
    # Demonstrate an operator with feedback
    print("\nDemonstrating operator feedback:")
    
    # Create an operator with varying amounts of feedback
    feedback_op_none = FMOperatorWithFeedback(
        carrier_type=WaveformType.SINE,
        modulator_type=WaveformType.SINE,
        c_to_m_ratio=1.0,
        modulation_index=1.0,
        feedback_amount=0.0
    )
    
    feedback_op_medium = FMOperatorWithFeedback(
        carrier_type=WaveformType.SINE,
        modulator_type=WaveformType.SINE,
        c_to_m_ratio=1.0,
        modulation_index=1.0,
        feedback_amount=0.3
    )
    
    feedback_op_high = FMOperatorWithFeedback(
        carrier_type=WaveformType.SINE,
        modulator_type=WaveformType.SINE,
        c_to_m_ratio=1.0,
        modulation_index=1.0,
        feedback_amount=0.6
    )
    
    # Create voices with single operators
    voice_no_fb = FMVoice([(feedback_op_none, 1.0)])
    voice_medium_fb = FMVoice([(feedback_op_medium, 1.0)])
    voice_high_fb = FMVoice([(feedback_op_high, 1.0)])
    
    # Create instruments
    inst_no_fb = Instrument(voice_no_fb)
    inst_medium_fb = Instrument(voice_medium_fb)
    inst_high_fb = Instrument(voice_high_fb)
    
    print("Saving notes with different feedback levels to WAV files...")
    inst_no_fb.play_note("C4", 1.0, 0.5, output_file="output/no_feedback.wav")
    inst_medium_fb.play_note("C4", 1.0, 0.5, output_file="output/medium_feedback.wav") 
    inst_high_fb.play_note("C4", 1.0, 0.5, output_file="output/high_feedback.wav")
    
    print(f"All WAV files saved to: {os.path.abspath('output')}")


if __name__ == "__main__":
    example()

