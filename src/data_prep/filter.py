import torchaudio.functional as F

def clean_machinery_audio(waveform, sample_rate):
    """
    Applies Bandpass filter to isolate mechanical frequencies.
    Removes low rumble (<100Hz) and high hiss (>6000Hz).
    """
    clean_wave = F.highpass_biquad(waveform, sample_rate, cutoff_freq=100.0)
    clean_wave = F.lowpass_biquad(clean_wave, sample_rate, cutoff_freq=6000.0)
    return clean_wave
