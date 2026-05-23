import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from .filter import clean_machinery_audio
import src.config as cfg

class AcousticMachineryDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.file_list = []
        self.labels = []
        
        # Assumes folder structure: data/raw/healthy/ and data/raw/broken/
        class_mapping = {"healthy": 0, "broken": 1}
        for class_name, label in class_mapping.items():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith('.wav'):
                        self.file_list.append(os.path.join(class_dir, file))
                        self.labels.append(label)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.TARGET_SAMPLE_RATE,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        
        waveform, sr = torchaudio.load(file_path)
        
        if sr != cfg.TARGET_SAMPLE_RATE:
            waveform = T.Resample(sr, cfg.TARGET_SAMPLE_RATE)(waveform)

        # Force mono channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply DSP filter
        waveform = clean_machinery_audio(waveform, cfg.TARGET_SAMPLE_RATE)

        # Pad or truncate to ensure uniform tensor sizes (5 seconds)
        max_length = cfg.TARGET_SAMPLE_RATE * cfg.MAX_AUDIO_LENGTH_SEC
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        else:
            padding = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Convert to 2D Spectrogram Image
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        return log_mel_spec, torch.tensor(label, dtype=torch.long)
