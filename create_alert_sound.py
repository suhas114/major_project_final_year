import numpy as np
from scipy.io import wavfile
import os

def create_alert_sound():
    print("Creating alert sound...")
    
    # Sound parameters
    sample_rate = 44100
    duration = 0.5  # seconds
    frequency = 880  # Hz (A5 note)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple beep sound
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Add fade in/out
    fade_duration = 0.1
    fade_length = int(fade_duration * sample_rate)
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    signal[:fade_length] *= fade_in
    signal[-fade_length:] *= fade_out
    
    # Normalize and convert to 16-bit integer
    signal = np.int16(signal * 32767)
    
    # Save as WAV file
    output_file = 'alert.wav'
    wavfile.write(output_file, sample_rate, signal)
    
    if os.path.exists(output_file):
        print(f"Alert sound file created successfully at: {os.path.abspath(output_file)}")
    else:
        print("Failed to create alert sound file!")

if __name__ == "__main__":
    create_alert_sound() 