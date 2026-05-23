Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import os
... 
... # Paths (adjust if running on Kaggle vs local Ubuntu)
... BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
... RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
... PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
... 
... # Audio Processing
... TARGET_SAMPLE_RATE = 16000
... N_MELS = 64
... N_FFT = 1024
... HOP_LENGTH = 512
... MAX_AUDIO_LENGTH_SEC = 5 # Truncate/pad all audio to 5 seconds
... 
... # Training Hyperparameters
... BATCH_SIZE = 32
... LEARNING_RATE = 1e-4
