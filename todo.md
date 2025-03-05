# FM Synth - Audio Enhancement Roadmap

This document outlines the plans and implementation steps for enhancing the FM synthesizer with advanced audio features.

## 1. Stereophonic Output Implementation

### Objectives
- Convert the current monophonic synthesis engine to full stereophonic output
- Add panning controls and stereo width capabilities
- Ensure proper stereo mixing in all audio processing stages

### Implementation Tasks
- [ ] Modify `Waveform.generate` to support stereo output (return shape: `(samples, 2)`)
- [ ] Add panning parameter to control sound positioning in the stereo field
- [ ] Update `FMOperator` and `FMOperatorWithFeedback` classes for stereo generation
- [ ] Add stereo width control to `FMVoice` class for creating naturalistic stereo fields
- [ ] Enhance `Sequencer` class to properly handle and mix stereo tracks
- [ ] Update audio effects to process stereo signals correctly
- [ ] Modify `Instrument.play_note` and `render_note` to support stereo signals

### Technical Details
- Implement constant power panning law for natural positioning
- Add independent L/R channel control in audio effects
- Support per-note panning in sequencer operations
- Ensure backward compatibility with existing code

## 2. LFO (Low Frequency Oscillator) Implementation

### Objectives
- Add flexible LFO system for dynamic parameter modulation
- Support various waveform types for different modulation characteristics
- Allow multiple modulation targets with configurable ranges

### Implementation Tasks
- [ ] Create `LFOType` enum with standard waveforms (sine, triangle, square, saw, random)
- [ ] Implement `ModulationTarget` enum defining parameters that can be modulated
- [ ] Develop `LFO` class with rate, depth, waveform, and phase controls
- [ ] Create `ModulationMatrix` to manage routing of LFOs to parameters
- [ ] Integrate LFO processing with FM synthesis operators
- [ ] Add methods to connect/disconnect LFOs to parameters
- [ ] Implement sample-accurate modulation calculation

### Technical Details
- Support for modulating: amplitude, frequency, modulation index, panning, feedback
- Allow bipolar and unipolar modulation modes
- Implement smoothing for discontinuous waveforms
- Add symmetry control for triangle and sawtooth waveforms
- Enable tempo-synchronized LFO rates

## 3. Enhanced Envelope Implementation

### Objectives
- Replace basic ADSR envelopes with more flexible DAHDSR envelopes
- Add support for various curve shapes in envelope segments
- Allow envelopes to modulate multiple synthesis parameters

### Implementation Tasks
- [ ] Create `EnvelopeType` enum for different curve shapes
- [ ] Implement `DAHDSREnvelope` class with Delay, Attack, Hold, Decay, Sustain, Release stages
- [ ] Add curve controls for each envelope segment
- [ ] Support multiple envelopes per operator targeting different parameters
- [ ] Add time scaling for tempo-synchronized envelopes
- [ ] Update operator classes to use enhanced envelopes
- [ ] Ensure backward compatibility with existing code

### Technical Details
- Implement curve types: linear, exponential, logarithmic, sigmoid, squared, cubed
- Support for independent curve shape per envelope segment
- Enable envelope retriggering and time scaling
- Allow parameter-specific envelope behavior

## 4. Integration and Testing

### Objectives
- Ensure all new systems work together harmoniously
- Maintain backward compatibility
- Create examples demonstrating new capabilities

### Implementation Tasks
- [ ] Develop integration tests for stereophonic output
- [ ] Create test cases for LFO modulation
- [ ] Verify envelope behavior across different parameters
- [ ] Update examples to showcase new features
- [ ] Document API changes and additions
- [ ] Optimize performance for real-time usage

## 5. Presets and Examples

### Objectives
- Create a library of presets showcasing the new features
- Develop example code for common use cases

### Implementation Tasks
- [ ] Create stereo instrument presets
- [ ] Develop LFO modulation presets for common effects (vibrato, tremolo, auto-pan)
- [ ] Design envelope presets for various articulations
- [ ] Update genre generators to use new features
- [ ] Document preset design techniques

## Timeline

- Phase 1: Stereophonic Output Implementation (2 weeks)
- Phase 2: Enhanced Envelope Implementation (2 weeks)
- Phase 3: LFO Implementation (3 weeks)
- Phase 4: Integration and Testing (1 week)
- Phase 5: Presets and Examples (1 week)

Total estimated time: 9 weeks