"""
Physical Modeling Guitar Synthesizer - Stratocaster Overdrive v1.0
=================================================================
Pure physics-based guitar synthesis from MIDI input.
No samples, no AI — only math and physics.

Dependencies: numpy, scipy, mido, soundfile
    pip install numpy scipy mido soundfile

Architecture:
    MIDI → String Model (Extended Karplus-Strong with waveguide)
       → Pickup Simulation (position-based comb filtering)
       → Body Resonance (parametric EQ modeling Strat body)
       → Overdrive (asymmetric tube-style waveshaping)
       → Cabinet Simulation (speaker response filter)
       → Output WAV
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import mido
import struct
import os
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StratConfig:
    """Stratocaster physical parameters and overdrive settings."""
    
    # Audio
    sample_rate: int = 44100
    
    # String physical properties (steel strings, standard tuning)
    # Tension, linear density, and damping per string
    string_damping: List[float] = field(default_factory=lambda: [
        0.996, 0.996, 0.997, 0.998, 0.998, 0.999  # E2 to E4 (heavier strings damp faster)
    ])
    
    # Pluck characteristics
    pluck_position: float = 0.73    # 73% from bridge (near middle pickup area)
    pluck_width: float = 0.05       # Width of pluck excitation
    pluck_sharpness: float = 0.7    # 0=soft(finger), 1=hard(pick)
    
    # Fret/string buzz parameters
    fret_buzz_amount: float = 0.02
    
    # Strat single-coil pickup positions (fraction of string length from bridge)
    pickup_bridge_pos: float = 0.117    # Bridge pickup
    pickup_middle_pos: float = 0.178    # Middle pickup  
    pickup_neck_pos: float = 0.268      # Neck pickup
    
    # Pickup selector: 0=bridge, 1=bridge+mid, 2=mid, 3=mid+neck, 4=neck
    pickup_selector: int = 0  # Bridge pickup for overdrive tone
    
    # Pickup electrical properties (single coil)
    pickup_resonance_freq: float = 3800.0   # Hz - single coil resonance
    pickup_resonance_q: float = 2.5         # Q factor
    pickup_inductance: float = 2.2          # Henries (approximate)
    
    # Guitar body resonances (Strat-specific)
    body_resonances: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        # (frequency_hz, gain_db, Q)
        (95.0, 3.0, 4.0),      # Main air resonance
        (200.0, 2.0, 3.5),     # Body mode 1
        (420.0, -2.0, 5.0),    # Anti-resonance
        (550.0, 1.5, 4.0),     # Body mode 2
        (700.0, -1.0, 6.0),    # Notch
        (1200.0, 1.0, 3.0),    # Upper body
        (2100.0, -1.5, 4.0),   # Scoop (Strat quack region)
        (3200.0, 2.0, 3.0),    # Presence
    ])
    
    # Overdrive parameters
    overdrive_gain: float = 25.0        # Pre-gain (amount of drive)
    overdrive_tone: float = 0.6         # Tone knob 0-1
    overdrive_mix: float = 1.0          # Wet/dry
    overdrive_asymmetry: float = 0.3    # Tube-like asymmetric clipping
    overdrive_sag: float = 0.05         # Power supply sag simulation
    
    # Tube stages (simulating a cranked amp)
    tube_stages: int = 2                # Number of gain stages
    tube_bias: float = 0.1              # Tube bias point
    
    # Tone stack (classic Fender-style)
    tonestack_bass: float = 0.6
    tonestack_mid: float = 0.4         # Strat mid-scoop
    tonestack_treble: float = 0.7
    
    # Cabinet simulation
    cabinet_enabled: bool = True
    cabinet_resonance: float = 80.0     # Speaker resonance Hz
    cabinet_rolloff: float = 4500.0     # High frequency rolloff Hz
    
    # Output
    master_volume: float = 0.8
    noise_floor: float = 0.001          # Subtle analog noise


# =============================================================================
# MIDI PARSER
# =============================================================================

@dataclass
class NoteEvent:
    """Parsed MIDI note event."""
    note: int           # MIDI note number
    velocity: int       # 0-127
    start_time: float   # seconds
    duration: float     # seconds
    channel: int = 0
    

def parse_midi(midi_path: str) -> List[NoteEvent]:
    """Parse MIDI file into note events with timing."""
    mid = mido.MidiFile(midi_path)
    events = []
    
    # Track active notes: {(channel, note): (velocity, start_time)}
    active_notes = {}
    
    for track in mid.tracks:
        current_time = 0.0  # in seconds
        
        for msg in track:
            # Convert delta time to seconds
            current_time += mido.tick2second(msg.time, mid.ticks_per_beat, 
                                              _get_tempo(mid))
            
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active_notes[key] = (msg.velocity, current_time)
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes:
                    velocity, start_time = active_notes.pop(key)
                    duration = current_time - start_time
                    events.append(NoteEvent(
                        note=msg.note,
                        velocity=velocity,
                        start_time=start_time,
                        duration=max(duration, 0.05),  # minimum 50ms
                        channel=msg.channel
                    ))
    
    # Close any remaining notes
    for key, (velocity, start_time) in active_notes.items():
        events.append(NoteEvent(
            note=key[1],
            velocity=velocity,
            start_time=start_time,
            duration=1.0,
            channel=key[0]
        ))
    
    events.sort(key=lambda e: e.start_time)
    return events


def _get_tempo(mid: mido.MidiFile) -> int:
    """Extract tempo from MIDI file."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    return 500000  # Default 120 BPM


def midi_note_to_freq(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


# =============================================================================
# STRING MODEL - Extended Karplus-Strong with Waveguide
# =============================================================================

class GuitarString:
    """
    Physical model of a guitar string using extended Karplus-Strong 
    with digital waveguide elements.
    
    Physics modeled:
    - Vibrating string (delay line + filtering)
    - Frequency-dependent decay (higher harmonics die faster)  
    - Stiffness (inharmonicity of steel strings)
    - Pluck position filtering
    - String coupling and sympathetic resonance
    """
    
    def __init__(self, frequency: float, config: StratConfig):
        self.frequency = frequency
        self.config = config
        self.sr = config.sample_rate
        
        # Delay line length for fundamental frequency
        # Account for the loop filter's phase delay (~0.5 samples)
        self.delay_length = self.sr / frequency
        self.int_delay = int(self.delay_length)
        self.frac_delay = self.delay_length - self.int_delay
        
        # Delay line (the "string")
        self.delay_line = np.zeros(self.int_delay + 2)  # +2 for interpolation
        self.write_pos = 0
        
        # Determine which string this is (for damping parameters)
        self.string_index = self._get_string_index(frequency)
        
        # Loop filter state (first-order averaging + damping)
        self.loop_filter_state = 0.0
        
        # Dynamics filter (frequency-dependent damping)
        self.dynamics_filter_state = 0.0
        
        # Allpass filter for fractional delay
        self.allpass_state = 0.0
        
        # String stiffness (inharmonicity coefficient)
        self.stiffness = self._calculate_stiffness(frequency)
        
        # Damping coefficient
        base_damping = config.string_damping[min(self.string_index, 5)]
        # Higher notes decay faster
        freq_factor = 1.0 - (frequency - 82.0) / 2000.0 * 0.008
        self.damping = base_damping * max(freq_factor, 0.98)
        
        # DC blocker state
        self.dc_x_prev = 0.0
        self.dc_y_prev = 0.0
        
    def _get_string_index(self, freq: float) -> int:
        """Determine which guitar string this note is on."""
        open_strings = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]
        for i, open_freq in enumerate(open_strings):
            if freq < open_freq * 2.0:  # Within one octave of open string
                return i
        return 5
    
    def _calculate_stiffness(self, freq: float) -> float:
        """
        Calculate inharmonicity based on string physics.
        B = π³ * E * d⁴ / (64 * T * L²)
        For steel guitar strings, B is typically 0.0001 to 0.005
        """
        # Approximate - heavier strings have more inharmonicity
        if freq < 100:
            return 0.0003
        elif freq < 200:
            return 0.0002
        elif freq < 300:
            return 0.00015
        else:
            return 0.0001
    
    def excite(self, velocity: float, pluck_position: float = None):
        """
        Initialize the delay line with a pluck excitation.
        
        Models the initial displacement of the string when plucked.
        The pluck position creates a comb-filter effect on harmonics.
        """
        if pluck_position is None:
            pluck_position = self.config.pluck_position
            
        n = len(self.delay_line)
        
        # --- Excitation signal ---
        # Combination of noise burst and triangular wave
        # mimics the complex initial string displacement
        
        # Noise component (pick scrape / attack transient)
        noise = np.random.uniform(-1, 1, n)
        
        # Triangular pluck shape (string displacement)
        pluck_idx = int(pluck_position * n)
        triangle = np.zeros(n)
        for i in range(n):
            if i <= pluck_idx:
                triangle[i] = i / max(pluck_idx, 1)
            else:
                triangle[i] = (n - i) / max(n - pluck_idx, 1)
        
        # Mix based on pluck sharpness (pick vs finger)
        sharpness = self.config.pluck_sharpness
        excitation = (1.0 - sharpness) * triangle + sharpness * noise * 0.5
        
        # --- Pluck position comb filter ---
        # The pluck position determines which harmonics are excited
        # Harmonics at n/(pluck_position) are suppressed
        comb_delay = max(int(pluck_position * n), 1)
        comb_filtered = np.zeros(n)
        for i in range(n):
            if i >= comb_delay:
                comb_filtered[i] = excitation[i] - excitation[i - comb_delay] * 0.5
            else:
                comb_filtered[i] = excitation[i]
        
        # --- Velocity-dependent brightness ---
        # Harder plucks excite more high harmonics
        vel_normalized = velocity / 127.0
        
        # Low-pass filter cutoff based on velocity
        cutoff = 1000.0 + vel_normalized * 6000.0  # 1kHz soft to 7kHz hard
        b, a = signal.butter(2, cutoff / (self.sr / 2), btype='low')
        filtered_excitation = signal.lfilter(b, a, comb_filtered)
        
        # --- Apply to delay line ---
        amplitude = vel_normalized * 0.8
        self.delay_line = filtered_excitation * amplitude
        
        # Reset filter states
        self.loop_filter_state = 0.0
        self.dynamics_filter_state = 0.0
        self.allpass_state = 0.0
        self.write_pos = 0
        
    def generate_samples(self, num_samples: int, damping_envelope: np.ndarray = None) -> np.ndarray:
        """
        Generate audio samples using the extended Karplus-Strong algorithm.
        
        Each sample:
        1. Read from delay line (with fractional interpolation)
        2. Apply loop filter (models frequency-dependent decay)
        3. Apply stiffness allpass (models inharmonicity)
        4. Write back to delay line
        5. Output the sample
        """
        output = np.zeros(num_samples)
        n = len(self.delay_line)
        
        for i in range(num_samples):
            # --- Read from delay line with linear interpolation ---
            read_pos = self.write_pos
            sample = self.delay_line[read_pos]
            
            # --- Loop filter (frequency-dependent damping) ---
            # First-order averaging: y[n] = (1-d)/2 * x[n] + (1+d)/2 * x[n-1]
            # This causes higher harmonics to decay faster (realistic)
            d = self.damping
            if damping_envelope is not None:
                d = d * damping_envelope[min(i, len(damping_envelope) - 1)]
            
            filtered = 0.5 * (1.0 + d) * sample + 0.5 * (1.0 - d) * self.loop_filter_state
            self.loop_filter_state = sample
            
            # --- Additional high-frequency damping ---
            # Models air resistance and internal string friction
            self.dynamics_filter_state = (0.9 * self.dynamics_filter_state + 
                                          0.1 * filtered)
            filtered = 0.85 * filtered + 0.15 * self.dynamics_filter_state
            
            # --- Allpass filter for fractional delay & stiffness ---
            # Fractional delay using first-order allpass
            c = (1.0 - self.frac_delay) / (1.0 + self.frac_delay)
            allpass_out = c * filtered + self.allpass_state
            self.allpass_state = filtered - c * allpass_out
            
            # --- Stiffness (inharmonicity) ---
            # Slight allpass to stretch higher partials
            if self.stiffness > 0:
                stiff_c = self.stiffness
                stiff_out = stiff_c * allpass_out + self.allpass_state * (1.0 - stiff_c)
                allpass_out = allpass_out * (1.0 - self.stiffness) + stiff_out * self.stiffness
            
            # --- DC blocker ---
            dc_out = allpass_out - self.dc_x_prev + 0.995 * self.dc_y_prev
            self.dc_x_prev = allpass_out
            self.dc_y_prev = dc_out
            allpass_out = dc_out
            
            # --- Write back to delay line ---
            self.delay_line[self.write_pos] = allpass_out
            
            # --- Advance write position ---
            self.write_pos = (self.write_pos + 1) % n
            
            output[i] = allpass_out
            
        return output


# =============================================================================
# PICKUP SIMULATION
# =============================================================================

class PickupSimulator:
    """
    Simulates Stratocaster single-coil pickups.
    
    Physics:
    - Pickup senses string velocity at a specific point
    - This creates a comb-filter effect based on pickup position
    - Single-coil has characteristic resonance peak
    - Electromagnetic properties create specific frequency response
    """
    
    def __init__(self, config: StratConfig):
        self.config = config
        self.sr = config.sample_rate
        
        # Build pickup response filters
        self.filters = self._build_pickup_filters()
        
    def _build_pickup_filters(self) -> dict:
        """Create filters modeling pickup electromagnetic response."""
        filters = {}
        
        # Single-coil resonance peak (the "quack")
        # Model as a bandpass resonance
        w0 = 2 * np.pi * self.config.pickup_resonance_freq / self.sr
        Q = self.config.pickup_resonance_q
        alpha = np.sin(w0) / (2 * Q)
        
        # Peaking EQ at resonance frequency
        A = 10 ** (6.0 / 40)  # 6dB boost at resonance
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        filters['resonance_b'] = np.array([b0/a0, b1/a0, b2/a0])
        filters['resonance_a'] = np.array([1.0, a1/a0, a2/a0])
        
        # High-frequency rolloff (eddy current losses)
        rolloff_freq = 6000.0
        filters['rolloff_b'], filters['rolloff_a'] = signal.butter(
            2, rolloff_freq / (self.sr / 2), btype='low'
        )
        
        # Slight high-pass (transformer coupling, no DC)
        filters['hp_b'], filters['hp_a'] = signal.butter(
            1, 30.0 / (self.sr / 2), btype='high'
        )
        
        return filters
    
    def _pickup_comb_filter(self, audio: np.ndarray, frequency: float, 
                             pickup_pos: float) -> np.ndarray:
        """
        Apply comb filtering based on pickup position.
        
        The pickup senses the velocity difference between two close points.
        Harmonics whose nodes fall on the pickup position are cancelled.
        Harmonic n is suppressed when pickup_pos ≈ k/n for integer k.
        """
        delay_samples = int(pickup_pos * self.sr / frequency)
        if delay_samples <= 0 or delay_samples >= len(audio):
            return audio
            
        output = np.copy(audio)
        # Comb filter: emphasizes harmonics based on pickup position
        for i in range(delay_samples, len(audio)):
            output[i] = audio[i] - audio[i - delay_samples] * 0.3
            
        return output
    
    def process(self, audio: np.ndarray, frequency: float) -> np.ndarray:
        """Apply complete pickup simulation to string audio."""
        config = self.config
        
        # Determine active pickups based on selector
        selector = config.pickup_selector
        pickup_configs = {
            0: [(config.pickup_bridge_pos, 1.0)],                           # Bridge
            1: [(config.pickup_bridge_pos, 0.7), (config.pickup_middle_pos, 0.7)],  # Bridge+Mid
            2: [(config.pickup_middle_pos, 1.0)],                           # Middle
            3: [(config.pickup_middle_pos, 0.7), (config.pickup_neck_pos, 0.7)],    # Mid+Neck
            4: [(config.pickup_neck_pos, 1.0)],                             # Neck
        }
        
        pickups = pickup_configs.get(selector, [(config.pickup_bridge_pos, 1.0)])
        
        # Process through each active pickup and sum
        result = np.zeros_like(audio)
        
        for pickup_pos, gain in pickups:
            picked = self._pickup_comb_filter(audio, frequency, pickup_pos)
            result += picked * gain
            
        # Normalize for number of pickups
        result /= len(pickups)
        
        # Apply pickup electromagnetic response
        # Resonance peak
        result = signal.lfilter(self.filters['resonance_b'], 
                                self.filters['resonance_a'], result)
        
        # High-frequency rolloff
        result = signal.lfilter(self.filters['rolloff_b'], 
                                self.filters['rolloff_a'], result)
        
        # High-pass (no DC)
        result = signal.lfilter(self.filters['hp_b'], 
                                self.filters['hp_a'], result)
        
        return result


# =============================================================================
# BODY RESONANCE
# =============================================================================

class BodyResonance:
    """
    Simulates the frequency response of a Stratocaster guitar body.
    
    The body acts as a complex filter with multiple resonances and 
    anti-resonances determined by the wood, shape, and construction.
    Modeled as a series of parametric EQ bands.
    """
    
    def __init__(self, config: StratConfig):
        self.config = config
        self.sr = config.sample_rate
        self.filters = self._build_body_filters()
        
    def _build_body_filters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Build parametric EQ filters for each body resonance."""
        filters = []
        
        for freq, gain_db, Q in self.config.body_resonances:
            if freq >= self.sr / 2:
                continue
                
            w0 = 2 * np.pi * freq / self.sr
            A = 10 ** (gain_db / 40.0)
            alpha = np.sin(w0) / (2 * Q)
            
            # Peaking EQ coefficients
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
            
            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1.0, a1/a0, a2/a0])
            filters.append((b, a))
            
        return filters
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply body resonance filtering."""
        result = audio.copy()
        
        for b, a in self.filters:
            result = signal.lfilter(b, a, result)
            
        return result


# =============================================================================
# OVERDRIVE / TUBE AMP SIMULATION
# =============================================================================

class TubeOverdrive:
    """
    Simulates tube amplifier overdrive using nonlinear waveshaping.
    
    Physics modeled:
    - Tube transfer function (asymmetric soft clipping)
    - Multiple gain stages
    - Inter-stage filtering
    - Power supply sag
    - Tone stack (Fender-style passive EQ)
    - Presence/resonance
    """
    
    def __init__(self, config: StratConfig):
        self.config = config
        self.sr = config.sample_rate
        
    def _tube_waveshaper(self, x: np.ndarray, asymmetry: float = 0.3, 
                          bias: float = 0.1) -> np.ndarray:
        """
        Attempt to model tube triode transfer characteristic.
        
        Uses asymmetric hyperbolic tangent - tubes clip asymmetrically
        because they conduct differently for positive vs negative grid voltage.
        
        Positive half: soft compression (plate current saturation)
        Negative half: harder clipping (cutoff)
        """
        output = np.zeros_like(x)
        
        # Add bias (tubes operate at a DC bias point)
        biased = x + bias
        
        for i in range(len(biased)):
            s = biased[i]
            
            if s >= 0:
                # Positive: soft saturation (plate current limiting)
                # tanh-like but with more gradual onset
                output[i] = np.tanh(s * (1.0 + asymmetry)) / (1.0 + asymmetry * 0.5)
            else:
                # Negative: harder clipping (approaching cutoff)
                # More abrupt transition
                output[i] = np.tanh(s * (1.0 - asymmetry * 0.5)) * (1.0 + asymmetry * 0.3)
                
            # Second-order harmonic generation (even harmonics from asymmetry)
            output[i] += asymmetry * 0.1 * (s * s - 0.5) * np.exp(-abs(s))
            
        return output
    
    def _power_sag(self, audio: np.ndarray, sag_amount: float) -> np.ndarray:
        """
        Simulate power supply sag.
        
        When the signal is loud, the power supply voltage drops slightly,
        causing compression and a "squished" feel.
        """
        if sag_amount <= 0:
            return audio
            
        envelope = np.abs(audio)
        # Slow follower (power supply has large capacitors)
        b, a = signal.butter(1, 20.0 / (self.sr / 2), btype='low')
        envelope = signal.lfilter(b, a, envelope)
        
        # Reduce gain when envelope is high
        gain_reduction = 1.0 / (1.0 + sag_amount * envelope * 10.0)
        
        return audio * gain_reduction
    
    def _interstage_filter(self, audio: np.ndarray, stage: int) -> np.ndarray:
        """
        Coupling capacitor and grid leak resistor between tube stages.
        Acts as a high-pass filter, removing DC and low-frequency buildup.
        Also slight high-frequency rolloff from Miller capacitance.
        """
        # Coupling cap high-pass (~20-50Hz depending on stage)
        hp_freq = 30.0 + stage * 20.0
        b_hp, a_hp = signal.butter(1, hp_freq / (self.sr / 2), btype='high')
        audio = signal.lfilter(b_hp, a_hp, audio)
        
        # Miller cap low-pass (progressively more filtered)
        lp_freq = 8000.0 - stage * 1500.0
        lp_freq = max(lp_freq, 3000.0)
        b_lp, a_lp = signal.butter(1, lp_freq / (self.sr / 2), btype='low')
        audio = signal.lfilter(b_lp, a_lp, audio)
        
        return audio
    
    def _fender_tone_stack(self, audio: np.ndarray) -> np.ndarray:
        """
        Model of a Fender-style passive tone stack.
        
        The Fender tone stack is a passive RC network with Bass, Mid, Treble.
        It has a characteristic mid-scoop that's fundamental to the Strat sound.
        
        Approximated here with parametric EQ bands.
        """
        bass = self.config.tonestack_bass
        mid = self.config.tonestack_mid
        treble = self.config.tonestack_treble
        
        result = audio.copy()
        
        # Bass shelf
        bass_gain_db = (bass - 0.5) * 16.0  # -8 to +8 dB
        if abs(bass_gain_db) > 0.1:
            w0 = 2 * np.pi * 250.0 / self.sr
            A = 10 ** (bass_gain_db / 40)
            alpha = np.sin(w0) / (2 * 0.7)
            
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            
            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1.0, a1/a0, a2/a0])
            result = signal.lfilter(b, a, result)
        
        # Mid peak/cut (the famous Fender scoop)
        mid_gain_db = (mid - 0.5) * 20.0  # -10 to +10 dB
        w0 = 2 * np.pi * 650.0 / self.sr
        A = 10 ** (mid_gain_db / 40)
        alpha = np.sin(w0) / (2 * 1.5)  # Q of 1.5
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        result = signal.lfilter(b, a, result)
        
        # Treble shelf
        treble_gain_db = (treble - 0.5) * 16.0
        if abs(treble_gain_db) > 0.1:
            w0 = 2 * np.pi * 3000.0 / self.sr
            A = 10 ** (treble_gain_db / 40)
            alpha = np.sin(w0) / (2 * 0.7)
            
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            
            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1.0, a1/a0, a2/a0])
            result = signal.lfilter(b, a, result)
        
        # Inherent insertion loss of passive tone stack (~-10dB)
        result *= 0.3
        
        return result
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Full overdrive signal chain."""
        config = self.config
        result = audio.copy()
        
        # --- Pre-gain ---
        result *= config.overdrive_gain
        
        # --- Tube stages ---
        for stage in range(config.tube_stages):
            # Inter-stage coupling filter
            result = self._interstage_filter(result, stage)
            
            # Tube waveshaping
            result = self._tube_waveshaper(
                result, 
                asymmetry=config.overdrive_asymmetry,
                bias=config.tube_bias
            )
            
            # Each stage adds gain
            result *= 1.5
        
        # --- Power supply sag ---
        result = self._power_sag(result, config.overdrive_sag)
        
        # --- Tone stack ---
        result = self._fender_tone_stack(result)
        
        # --- Post-overdrive tone control ---
        # Simple low-pass based on tone knob
        tone_freq = 2000.0 + config.overdrive_tone * 6000.0
        b, a = signal.butter(2, tone_freq / (self.sr / 2), btype='low')
        result = signal.lfilter(b, a, result)
        
        # --- Mix dry/wet ---
        if config.overdrive_mix < 1.0:
            result = config.overdrive_mix * result + (1.0 - config.overdrive_mix) * audio
        
        return result


# =============================================================================
# CABINET SIMULATION
# =============================================================================

class CabinetSimulator:
    """
    Simulates a guitar speaker cabinet using filter modeling.
    
    Guitar speakers have a very limited and colored frequency response:
    - Strong resonance around 80-100Hz
    - Relatively flat midrange
    - Sharp rolloff above 4-5kHz (no tweeter)
    - Various resonances from cabinet construction
    """
    
    def __init__(self, config: StratConfig):
        self.config = config
        self.sr = config.sample_rate
        self.filters = self._build_cabinet_filters()
        
    def _build_cabinet_filters(self) -> dict:
        """Build speaker/cabinet response filters."""
        filters = {}
        
        # Speaker resonance (low end bump)
        res_freq = self.config.cabinet_resonance
        w0 = 2 * np.pi * res_freq / self.sr
        Q = 1.2
        A = 10 ** (4.0 / 40)  # 4dB boost
        alpha = np.sin(w0) / (2 * Q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        filters['resonance_b'] = np.array([b0/a0, b1/a0, b2/a0])
        filters['resonance_a'] = np.array([1.0, a1/a0, a2/a0])
        
        # High frequency rolloff (speaker cone can't reproduce highs)
        # Steep rolloff above ~4.5kHz - this is crucial for guitar sound
        rolloff = min(self.config.cabinet_rolloff, self.sr / 2 - 100)
        filters['rolloff_b'], filters['rolloff_a'] = signal.butter(
            4, rolloff / (self.sr / 2), btype='low'
        )
        
        # Low cut (cabinet doesn't reproduce sub-bass well)
        filters['lowcut_b'], filters['lowcut_a'] = signal.butter(
            2, 60.0 / (self.sr / 2), btype='high'
        )
        
        # Presence peak (speaker breakup area ~2-3kHz)
        pres_freq = 2500.0
        w0 = 2 * np.pi * pres_freq / self.sr
        Q = 2.0
        A = 10 ** (3.0 / 40)
        alpha = np.sin(w0) / (2 * Q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        filters['presence_b'] = np.array([b0/a0, b1/a0, b2/a0])
        filters['presence_a'] = np.array([1.0, a1/a0, a2/a0])
        
        return filters
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply cabinet simulation."""
        if not self.config.cabinet_enabled:
            return audio
            
        result = audio.copy()
        
        # Apply speaker resonance
        result = signal.lfilter(self.filters['resonance_b'], 
                                self.filters['resonance_a'], result)
        
        # Apply high rolloff (most important!)
        result = signal.lfilter(self.filters['rolloff_b'], 
                                self.filters['rolloff_a'], result)
        
        # Apply low cut
        result = signal.lfilter(self.filters['lowcut_b'], 
                                self.filters['lowcut_a'], result)
        
        # Apply presence
        result = signal.lfilter(self.filters['presence_b'], 
                                self.filters['presence_a'], result)
        
        return result


# =============================================================================
# NOTE ENVELOPE
# =============================================================================

class NoteEnvelope:
    """
    Generate amplitude envelope for guitar notes.
    
    Guitar has:
    - Very fast attack (pick strike)
    - Initial brightness burst
    - Exponential decay
    - Optional palm mute / staccato
    """
    
    def __init__(self, config: StratConfig):
        self.config = config
        self.sr = config.sample_rate
        
    def generate(self, num_samples: int, velocity: float, 
                  duration: float) -> np.ndarray:
        """Generate amplitude envelope."""
        t = np.arange(num_samples) / self.sr
        
        # Attack: very fast (< 2ms for pick)
        attack_time = 0.002
        
        # Decay time based on velocity and duration
        decay_time = duration * 0.3
        
        # Sustain level
        sustain = 0.6 + velocity * 0.3
        
        # Release time
        release_time = 0.1
        
        envelope = np.ones(num_samples)
        
        for i in range(num_samples):
            time = t[i]
            
            if time < attack_time:
                # Fast attack
                envelope[i] = time / attack_time
            elif time < attack_time + decay_time:
                # Decay to sustain
                decay_progress = (time - attack_time) / decay_time
                envelope[i] = 1.0 - (1.0 - sustain) * decay_progress
            elif time < duration:
                # Sustain with slow decay (string losing energy)
                sustain_time = time - attack_time - decay_time
                envelope[i] = sustain * np.exp(-sustain_time * 1.5)
            else:
                # Release (finger lift / mute)
                release_progress = (time - duration) / release_time
                if release_progress >= 1.0:
                    envelope[i] = 0.0
                else:
                    prev_level = sustain * np.exp(-(duration - attack_time - decay_time) * 1.5)
                    envelope[i] = prev_level * (1.0 - release_progress) ** 2
        
        return envelope


# =============================================================================
# PICK ATTACK TRANSIENT
# =============================================================================

class PickAttack:
    """
    Synthesize the initial pick attack transient.
    
    When a pick strikes a string, there's a brief burst of broadband noise
    from the pick scraping across the string winding. This is crucial for realism.
    """
    
    def __init__(self, config: StratConfig):
        self.config = config
        self.sr = config.sample_rate
        
    def generate(self, num_samples: int, velocity: float) -> np.ndarray:
        """Generate pick attack transient."""
        attack = np.zeros(num_samples)
        
        # Duration of pick scrape noise (1-5ms depending on velocity)
        duration_samples = int(self.sr * (0.001 + 0.004 * velocity))
        duration_samples = min(duration_samples, num_samples)
        
        if duration_samples <= 0:
            return attack
        
        # Broadband noise burst
        noise = np.random.randn(duration_samples)
        
        # Shape: fast attack, fast decay
        t = np.arange(duration_samples) / self.sr
        env = np.exp(-t * 1000) * velocity * 0.3
        
        # Bandpass filter (pick noise is mid-high frequency)
        if duration_samples > 20:
            nyq = self.sr / 2
            low = min(2000.0 / nyq, 0.99)
            high = min(8000.0 / nyq, 0.99)
            if low < high:
                b, a = signal.butter(2, [low, high], btype='band')
                noise = signal.lfilter(b, a, noise)
        
        attack[:duration_samples] = noise * env
        
        return attack


# =============================================================================
# MAIN SYNTHESIZER ENGINE
# =============================================================================

class StratocasterSynth:
    """
    Main synthesizer engine that combines all physical models.
    
    Signal chain:
    MIDI Note → String Model → Pickup Sim → Body Resonance → 
    Overdrive → Cabinet → Master Volume → Output
    """
    
    def __init__(self, config: StratConfig = None):
        self.config = config or StratConfig()
        self.sr = self.config.sample_rate
        
        # Initialize components
        self.pickup = PickupSimulator(self.config)
        self.body = BodyResonance(self.config)
        self.overdrive = TubeOverdrive(self.config)
        self.cabinet = CabinetSimulator(self.config)
        self.envelope_gen = NoteEnvelope(self.config)
        self.pick_attack = PickAttack(self.config)
        
        print("Stratocaster Physical Model Synth initialized")
        print(f"  Sample rate: {self.sr} Hz")
        print(f"  Pickup: position {self.config.pickup_selector}")
        print(f"  Overdrive gain: {self.config.overdrive_gain}")
        print(f"  Tube stages: {self.config.tube_stages}")
        
    def _synthesize_note(self, note_event: NoteEvent) -> Tuple[np.ndarray, float]:
        """
        Synthesize a single guitar note.
        
        Returns: (audio_array, start_time_seconds)
        """
        freq = midi_note_to_freq(note_event.note)
        velocity = note_event.velocity / 127.0
        
        # Total duration including release tail
        total_duration = note_event.duration + 0.5  # 500ms release tail
        num_samples = int(total_duration * self.sr)
        
        # Limit to reasonable range for guitar (E2 to E6)
        if freq < 70 or freq > 1400:
            print(f"  Warning: Note {note_event.note} ({freq:.1f}Hz) outside guitar range")
            freq = np.clip(freq, 70, 1400)
        
        # --- 1. Generate string vibration ---
        string = GuitarString(freq, self.config)
        
        # Slight randomization of pluck position (human variation)
        pluck_pos = self.config.pluck_position + np.random.uniform(-0.05, 0.05)
        pluck_pos = np.clip(pluck_pos, 0.1, 0.9)
        
        string.excite(note_event.velocity, pluck_pos)
        
        # Generate damping envelope for the string
        damping_env = self.envelope_gen.generate(num_samples, velocity, note_event.duration)
        
        # Generate string audio
        string_audio = string.generate_samples(num_samples)
        
        # Apply amplitude envelope
        string_audio *= damping_env
        
        # --- 2. Add pick attack transient ---
        pick_transient = self.pick_attack.generate(num_samples, velocity)
        string_audio += pick_transient
        
        # --- 3. Pickup simulation ---
        pickup_audio = self.pickup.process(string_audio, freq)
        
        # --- 4. Body resonance ---
        body_audio = self.body.process(pickup_audio)
        
        # --- 5. Overdrive ---
        driven_audio = self.overdrive.process(body_audio)
        
        # --- 6. Cabinet simulation ---
        cab_audio = self.cabinet.process(driven_audio)
        
        return cab_audio, note_event.start_time
    
    def synthesize_midi(self, midi_path: str, output_path: str = "output.wav"):
        """
        Synthesize entire MIDI file to WAV.
        """
        print(f"\nParsing MIDI: {midi_path}")
        notes = parse_midi(midi_path)
        print(f"  Found {len(notes)} notes")
        
        if not notes:
            print("  No notes found!")
            return
        
        # Calculate total duration
        max_end_time = max(n.start_time + n.duration for n in notes)
        total_duration = max_end_time + 1.0  # 1 second tail
        total_samples = int(total_duration * self.sr)
        
        print(f"  Total duration: {total_duration:.2f}s ({total_samples} samples)")
        
        # Output buffer
        output = np.zeros(total_samples)
        
        # Synthesize each note
        for i, note in enumerate(notes):
            freq = midi_note_to_freq(note.note)
            print(f"  Synthesizing note {i+1}/{len(notes)}: "
                  f"MIDI {note.note} ({freq:.1f}Hz) "
                  f"vel={note.velocity} "
                  f"t={note.start_time:.3f}s dur={note.duration:.3f}s")
            
            note_audio, start_time = self._synthesize_note(note)
            
            # Place in output buffer
            start_sample = int(start_time * self.sr)
            end_sample = min(start_sample + len(note_audio), total_samples)
            actual_length = end_sample - start_sample
            
            if actual_length > 0:
                output[start_sample:end_sample] += note_audio[:actual_length]
        
        # --- Post-processing ---
        print("\n  Post-processing...")
        
        # Add subtle analog noise
        if self.config.noise_floor > 0:
            noise = np.random.randn(total_samples) * self.config.noise_floor
            # Shape noise (more hiss-like)
            b, a = signal.butter(2, [200.0 / (self.sr/2), 8000.0 / (self.sr/2)], btype='band')
            noise = signal.lfilter(b, a, noise)
            output += noise
        
        # Master volume and limiting
        output *= self.config.master_volume
        
        # Soft limiter to prevent clipping
        peak = np.max(np.abs(output))
        if peak > 0:
            # Normalize to -1dB
            target = 10 ** (-1.0 / 20.0)  # ~0.89
            if peak > target:
                output = np.tanh(output / peak * 1.5) * target
            else:
                output *= target / peak
        
        # --- Write output ---
        print(f"\n  Writing: {output_path}")
        
        # Convert to 16-bit PCM
        output_16bit = np.int16(output * 32767)
        wavfile.write(output_path, self.sr, output_16bit)
        
        print(f"  Done! File size: {os.path.getsize(output_path) / 1024:.1f} KB")
        
        return output
    
    def synthesize_test(self, output_path: str = "strat_test.wav"):
        """
        Generate a test output without MIDI file.
        Plays a simple chord progression to demonstrate the sound.
        """
        print("\nGenerating test output (no MIDI file needed)...")
        
        # Create some test notes - power chord riff
        notes = []
        
        # E5 power chord (classic rock)
        t = 0.0
        notes.append(NoteEvent(note=40, velocity=100, start_time=t, duration=0.4))  # E2
        notes.append(NoteEvent(note=47, velocity=95, start_time=t, duration=0.4))   # B2
        notes.append(NoteEvent(note=52, velocity=90, start_time=t, duration=0.4))   # E3
        
        t = 0.5
        notes.append(NoteEvent(note=40, velocity=110, start_time=t, duration=0.2))  # E2
        notes.append(NoteEvent(note=47, velocity=105, start_time=t, duration=0.2))  # B2
        
        t = 0.8
        notes.append(NoteEvent(note=43, velocity=100, start_time=t, duration=0.4))  # G2
        notes.append(NoteEvent(note=50, velocity=95, start_time=t, duration=0.4))   # D3
        notes.append(NoteEvent(note=55, velocity=90, start_time=t, duration=0.4))   # G3
        
        t = 1.3
        notes.append(NoteEvent(note=45, velocity=105, start_time=t, duration=0.4))  # A2
        notes.append(NoteEvent(note=52, velocity=100, start_time=t, duration=0.4))  # E3
        notes.append(NoteEvent(note=57, velocity=95, start_time=t, duration=0.4))   # A3
        
        # Single note run
        t = 1.9
        for note in [52, 55, 57, 59, 60, 59, 57, 55]:
            notes.append(NoteEvent(note=note, velocity=90, start_time=t, duration=0.15))
            t += 0.18
        
        # Final power chord
        t += 0.1
        notes.append(NoteEvent(note=40, velocity=120, start_time=t, duration=1.5))  # E2
        notes.append(NoteEvent(note=47, velocity=115, start_time=t, duration=1.5))  # B2
        notes.append(NoteEvent(note=52, velocity=110, start_time=t, duration=1.5))  # E3
        notes.append(NoteEvent(note=56, velocity=105, start_time=t, duration=1.5))  # G#3
        notes.append(NoteEvent(note=59, velocity=100, start_time=t, duration=1.5))  # B3
        
        # Calculate total duration
        max_end = max(n.start_time + n.duration for n in notes)
        total_duration = max_end + 1.5
        total_samples = int(total_duration * self.sr)
        
        output = np.zeros(total_samples)
        
        for i, note in enumerate(notes):
            freq = midi_note_to_freq(note.note)
            print(f"  Note {i+1}/{len(notes)}: MIDI {note.note} ({freq:.1f}Hz)")
            
            note_audio, start_time = self._synthesize_note(note)
            
            start_sample = int(start_time * self.sr)
            end_sample = min(start_sample + len(note_audio), total_samples)
            actual_length = end_sample - start_sample
            
            if actual_length > 0:
                output[start_sample:end_sample] += note_audio[:actual_length]
        
        # Post-processing
        output *= self.config.master_volume
        
        peak = np.max(np.abs(output))
        if peak > 0:
            target = 0.89
            if peak > target:
                output = np.tanh(output / peak * 1.5) * target
            else:
                output *= target / peak
        
        # Add noise floor
        noise = np.random.randn(total_samples) * self.config.noise_floor
        b, a = signal.butter(2, [200.0 / (self.sr/2), 8000.0 / (self.sr/2)], btype='band')
        noise = signal.lfilter(b, a, noise)
        output += noise
        
        output_16bit = np.int16(np.clip(output, -1, 1) * 32767)
        wavfile.write(output_path, self.sr, output_16bit)
        
        print(f"\n  Written: {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
        print(f"  Duration: {total_duration:.2f}s")
        
        return output


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def preset_clean_strat() -> StratConfig:
    """Clean Stratocaster tone (positions 2 or 4 for quack)."""
    config = StratConfig()
    config.overdrive_gain = 1.5
    config.tube_stages = 1
    config.pickup_selector = 1  # Bridge + Middle (quack)
    config.tonestack_treble = 0.7
    config.tonestack_mid = 0.6
    config.tonestack_bass = 0.5
    return config


def preset_crunch_strat() -> StratConfig:
    """Crunchy Strat - edge of breakup."""
    config = StratConfig()
    config.overdrive_gain = 12.0
    config.tube_stages = 2
    config.pickup_selector = 0  # Bridge
    config.overdrive_asymmetry = 0.25
    config.tonestack_treble = 0.65
    config.tonestack_mid = 0.45
    config.tonestack_bass = 0.55
    return config


def preset_overdrive_strat() -> StratConfig:
    """Full overdrive Strat - Hendrix / SRV territory."""
    config = StratConfig()
    config.overdrive_gain = 25.0
    config.tube_stages = 2
    config.pickup_selector = 0  # Bridge
    config.overdrive_asymmetry = 0.3
    config.overdrive_sag = 0.05
    config.tonestack_treble = 0.7
    config.tonestack_mid = 0.4
    config.tonestack_bass = 0.6
    config.pluck_sharpness = 0.8
    return config


def preset_high_gain_strat() -> StratConfig:
    """High gain - hard rock / metal territory."""
    config = StratConfig()
    config.overdrive_gain = 50.0
    config.tube_stages = 3
    config.pickup_selector = 0  # Bridge
    config.overdrive_asymmetry = 0.35
    config.overdrive_sag = 0.08
    config.tonestack_treble = 0.6
    config.tonestack_mid = 0.35
    config.tonestack_bass = 0.7
    config.pluck_sharpness = 0.9
    config.cabinet_rolloff = 4000.0
    return config


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Physical Modeling Stratocaster Synthesizer v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  clean      - Clean Strat (pickup pos 2, light)
  crunch     - Edge of breakup
  overdrive  - Full overdrive (Hendrix/SRV) [default]
  highgain   - High gain rock/metal

Examples:
  python strat_synth.py --test                          # Generate test riff
  python strat_synth.py input.mid -o output.wav         # Process MIDI file
  python strat_synth.py input.mid --preset highgain     # High gain preset
  python strat_synth.py input.mid --gain 30 --pickup 2  # Custom settings
        """
    )
    
    parser.add_argument('midi_file', nargs='?', help='Input MIDI file path')
    parser.add_argument('-o', '--output', default='output.wav', help='Output WAV path')
    parser.add_argument('--test', action='store_true', help='Generate test output (no MIDI needed)')
    parser.add_argument('--preset', choices=['clean', 'crunch', 'overdrive', 'highgain'],
                        default='overdrive', help='Tone preset')
    parser.add_argument('--gain', type=float, help='Override overdrive gain (1-100)')
    parser.add_argument('--pickup', type=int, choices=[0,1,2,3,4],
                        help='Pickup selector (0=bridge, 2=mid, 4=neck)')
    parser.add_argument('--stages', type=int, choices=[1,2,3],
                        help='Number of tube gain stages')
    parser.add_argument('--volume', type=float, default=0.8, help='Master volume (0-1)')
    
    args = parser.parse_args()
    
    # Select preset
    presets = {
        'clean': preset_clean_strat,
        'crunch': preset_crunch_strat,
        'overdrive': preset_overdrive_strat,
        'highgain': preset_high_gain_strat,
    }
    
    config = presets[args.preset]()
    
    # Apply overrides
    if args.gain is not None:
        config.overdrive_gain = args.gain
    if args.pickup is not None:
        config.pickup_selector = args.pickup
    if args.stages is not None:
        config.tube_stages = args.stages
    config.master_volume = args.volume
    
    # Create synth
    synth = StratocasterSynth(config)
    
    if args.test:
        synth.synthesize_test(args.output)
    elif args.midi_file:
        if not os.path.exists(args.midi_file):
            print(f"Error: MIDI file not found: {args.midi_file}")
            return
        synth.synthesize_midi(args.midi_file, args.output)
    else:
        print("Error: Provide a MIDI file or use --test flag")
        parser.print_help()


if __name__ == '__main__':
    main()

config = preset_overdrive_strat()
config.overdrive_gain = 30.0  # customize
synth = StratocasterSynth(config)
synth.synthesize_midi("input.mid", "output.wav")
