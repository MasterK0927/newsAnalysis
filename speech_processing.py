import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import sounddevice as sd
import scipy.signal as signal
from scipy.signal import butter, lfilter
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import pyworld as pw
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import sounddevice as sd
import scipy.signal as signal
from scipy.signal import butter, lfilter
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import pyworld as pw
from typing import List, Optional, Tuple, Dict, Union

class LanguageConfig:
    """Configuration for different language support"""
    LANGUAGE_CONFIGS = {
        'english': {
            'labels': [chr(i) for i in range(97, 123)] + ['<space>', '<blank>'],
            'wav2vec_model': "facebook/wav2vec2-base-960h",
            'lowcut': 300,
            'highcut': 3000
        },
        'hindi': {
            'labels': [
                # Devanagari characters for Hindi (covering most common characters)
                'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ',
                'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
                'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
                'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श',
                'ष', 'स', 'ह', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै',
                'ो', 'ौ', '्', '<space>', '<blank>'
            ],
            'wav2vec_model': "ai4bharat/indicwav2vec_v1_hindi",  # Updated model
            'lowcut': 200,
            'highcut': 3500
        },
        'tamil': {
            'labels': [
                # Tamil Unicode characters
                'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ',
                'க', 'ங', 'ச', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன',
                'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', '்',
                '<space>', '<blank>'
            ],
            'wav2vec_model': "ai4bharat/indicwav2vec_v1_tamil",  # Updated model
            'lowcut': 250,
            'highcut': 3300
        }
    }

    @classmethod
    def get_language_config(cls, language: str) -> Dict:
        """
        Retrieve language-specific configuration.
        
        Args:
            language (str): Language identifier
        
        Returns:
            Dict: Language-specific configuration
        """
        language = language.lower()
        if language not in cls.LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}. Supported languages: {list(cls.LANGUAGE_CONFIGS.keys())}")
        return cls.LANGUAGE_CONFIGS[language]

class AdvancedSignalPreprocessor:
    def __init__(self, lowcut: float = 300, highcut: float = 3000):
        """
        Initialize preprocessor with custom bandpass parameters.
        
        Args:
            lowcut (float): Lower frequency cutoff
            highcut (float): Higher frequency cutoff
        """
        self.lowcut = lowcut
        self.highcut = highcut

    def butter_bandpass(self, fs: float, order: int = 5) -> Tuple:
        """Design Butterworth bandpass filter."""
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data: np.ndarray, fs: float, order: int = 5) -> np.ndarray:
        """Apply Butterworth bandpass filter to signal."""
        b, a = self.butter_bandpass(fs, order=order)
        return lfilter(b, a, data)

    @staticmethod
    def noise_reduction(signal: np.ndarray, noise_threshold: float = 0.02) -> np.ndarray:
        """Advanced noise reduction using spectral subtraction."""
        stft = librosa.stft(signal)
        noise_profile = np.mean(np.abs(stft[:, :5]), axis=1)
        
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        enhanced_magnitude = np.maximum(magnitude - noise_threshold * noise_profile[:, np.newaxis], 0)
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        return librosa.istft(enhanced_stft)

    def speech_enhancement(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Comprehensive speech enhancement pipeline."""
        # Ensure signal is 1D
        signal = signal.flatten()
        
        # Bandpass filtering
        filtered_signal = self.butter_bandpass_filter(signal, sample_rate)
        
        # Noise reduction
        noise_reduced = self.noise_reduction(filtered_signal)
        
        # Pitch correction and normalization
        f0, t = pw.dio(noise_reduced.astype(np.float64), sample_rate)
        f0 = pw.stonemask(noise_reduced.astype(np.float64), f0, t, sample_rate)
        
        # Normalize signal
        normalized_signal = librosa.util.normalize(noise_reduced)
        
        return normalized_signal

class AdvancedFeatureExtractor:
    @staticmethod
    def extract_multi_features(signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract advanced multi-dimensional speech features."""
        # Ensure signal is 1D
        signal = signal.flatten()
        
        # 1. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal, 
            sr=sample_rate, 
            n_mels=80,
            n_fft=2048,
            hop_length=512
        )
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Flatten to 1D for consistent input
        return log_mel_spec.flatten()

# Rest of the code remains the same, with one modification in the SpeechTranscriptionSystem.transcribe method:

def transcribe(self, 
               audio: Optional[np.ndarray] = None, 
               use_custom_model: bool = False) -> Dict[str, str]:
    """
    Transcribe audio using multiple models.
    
    Args:
        audio (np.ndarray, optional): Input audio
        use_custom_model (bool): Flag to use custom model
    
    Returns:
        Dict of transcription results
    """
    # Record audio if not provided
    if audio is None:
        audio = self.record_audio()
    
    # Ensure audio is 1D
    audio = audio.flatten()
    
    results = {}
    
    # Wav2Vec2 Transcription (primary method)
    inputs = self.wav2vec_processor(
        audio, 
        sampling_rate=self.sample_rate, 
        return_tensors="pt", 
        padding=True
    )
    
    with torch.no_grad():
        logits = self.wav2vec_model(
            inputs.input_values, 
            attention_mask=inputs.attention_mask
        ).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        wav2vec_transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0]
        results['wav2vec_model'] = wav2vec_transcription
    
    # Custom Model Transcription (optional)
    if use_custom_model:
        _, features = self.preprocess_audio(audio)
        
        with torch.no_grad():
            predictions = self.model(features)
            custom_transcription = self.decode_predictions(predictions)
            results['custom_model'] = custom_transcription
    
    return results

class AdvancedSignalPreprocessor:
    def __init__(self, lowcut: float = 300, highcut: float = 3000):
        """
        Initialize preprocessor with custom bandpass parameters.
        
        Args:
            lowcut (float): Lower frequency cutoff
            highcut (float): Higher frequency cutoff
        """
        self.lowcut = lowcut
        self.highcut = highcut

    def butter_bandpass(self, fs: float, order: int = 5) -> Tuple:
        """Design Butterworth bandpass filter."""
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data: np.ndarray, fs: float, order: int = 5) -> np.ndarray:
        """Apply Butterworth bandpass filter to signal."""
        b, a = self.butter_bandpass(fs, order=order)
        return lfilter(b, a, data)

    @staticmethod
    def noise_reduction(signal: np.ndarray, noise_threshold: float = 0.02) -> np.ndarray:
        """Advanced noise reduction using spectral subtraction."""
        stft = librosa.stft(signal)
        noise_profile = np.mean(np.abs(stft[:, :5]), axis=1)
        
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        enhanced_magnitude = np.maximum(magnitude - noise_threshold * noise_profile[:, np.newaxis], 0)
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        return librosa.istft(enhanced_stft)

    def speech_enhancement(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Comprehensive speech enhancement pipeline."""
        # Bandpass filtering
        filtered_signal = self.butter_bandpass_filter(signal, sample_rate)
        
        # Noise reduction
        noise_reduced = self.noise_reduction(filtered_signal)
        
        # Pitch correction and normalization
        f0, t = pw.dio(noise_reduced.astype(np.float64), sample_rate)
        f0 = pw.stonemask(noise_reduced.astype(np.float64), f0, t, sample_rate)
        
        # Normalize signal
        normalized_signal = librosa.util.normalize(noise_reduced)
        
        return normalized_signal

class AdvancedFeatureExtractor:
    @staticmethod
    def extract_multi_features(signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract advanced multi-dimensional speech features."""
        # 1. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal, 
            sr=sample_rate, 
            n_mels=80,
            n_fft=2048,
            hop_length=512
        )
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # 2. MFCC Features
        mfccs = librosa.feature.mfcc(
            y=signal, 
            sr=sample_rate, 
            n_mfcc=13
        )
        
        # 3. Spectral Features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=signal, 
            sr=sample_rate
        )
        
        # 4. Chroma Features
        chroma = librosa.feature.chroma_stft(
            y=signal, 
            sr=sample_rate
        )
        
        # 5. Harmonic-Percussive Source Separation
        harmonic, _ = librosa.effects.hpss(signal)
        
        # Combine features
        features = np.concatenate([
            log_mel_spec,
            mfccs,
            spectral_centroids,
            chroma,
            harmonic
        ])
        
        return features

class SelfAttention(nn.Module):
    """Self-attention mechanism for capturing long-range dependencies."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        
        self.out_projection = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.size()
        
        # Multi-head attention computation
        query = self.query(x).view(batch, seq_len, self.num_heads, dim // self.num_heads)
        key = self.key(x).view(batch, seq_len, self.num_heads, dim // self.num_heads)
        value = self.value(x).view(batch, seq_len, self.num_heads, dim // self.num_heads)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        
        return self.out_projection(context)

class AdvancedSpeechModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 512, 
                 output_dim: int = 29, 
                 num_classes: int = 29):
        super().__init__()
        
        # Multi-scale Convolutional Feature Extraction
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=k, stride=2, padding=k//2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for k in [3, 5, 7]
        ])
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128 * 3, 
            hidden_size=hidden_dim, 
            num_layers=3, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        # Self-Attention Layer
        self.attention = SelfAttention(hidden_dim * 2)
        
        # Classification Layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # CTC Layer
        self.ctc_layer = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale feature extraction
        multi_scale_features = [
            conv(x.transpose(1, 2)).transpose(1, 2)
            for conv in self.feature_extractor
        ]
        
        # Concatenate multi-scale features
        combined_features = torch.cat(multi_scale_features, dim=-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined_features)
        
        # Attention mechanism
        attended_features = self.attention(lstm_out)
        
        # CTC output
        ctc_output = self.ctc_layer(attended_features)
        
        return ctc_output

class SpeechTranscriptionSystem:
    def __init__(self, language: str = 'english'):
        """
        Initialize the speech transcription system for a specific language.
        
        Args:
            language (str): Language for transcription. Defaults to 'english'.
        """
        # Get language-specific configuration
        self.language_config = LanguageConfig.get_language_config(language)
        
        # Set labels for decoding
        self.labels = self.language_config['labels']
        
        # Sample rate (standard for most speech models)
        self.sample_rate = 16000
        
        # Initialize signal preprocessor with language-specific frequency cutoffs
        self.preprocessor = AdvancedSignalPreprocessor(
            lowcut=self.language_config['lowcut'], 
            highcut=self.language_config['highcut']
        )
        
        # Initialize feature extractor
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Initialize custom speech recognition model
        self.model = AdvancedSpeechModel(
            input_dim=256, 
            output_dim=len(self.labels), 
            num_classes=len(self.labels)
        )
        
        # Initialize Wav2Vec2 processor and model for the specific language
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            self.language_config['wav2vec_model']
        )
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(
            self.language_config['wav2vec_model']
        )
    
    def record_audio(self, duration: float = 3.0) -> np.ndarray:
        """
        Record audio from the default microphone.
        
        Args:
            duration (float): Recording duration in seconds
        
        Returns:
            np.ndarray: Recorded audio signal
        """
        print(f"Recording audio for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate), 
            samplerate=self.sample_rate, 
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return audio.flatten()
    
    def preprocess_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Preprocess the input audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
        
        Returns:
            Tuple of enhanced signal and extracted features
        """
        # Apply speech enhancement
        enhanced_signal = self.preprocessor.speech_enhancement(
            audio, 
            sample_rate=self.sample_rate
        )
        
        # Extract multi-dimensional features
        features = self.feature_extractor.extract_multi_features(
            enhanced_signal, 
            sample_rate=self.sample_rate
        )
        
        # Convert to tensor for model input
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        return enhanced_signal, features_tensor

    def decode_predictions(self, predictions: torch.Tensor) -> str:
        """
        Simple greedy decoding of model predictions.
        
        Args:
            predictions (torch.Tensor): Model output predictions
        
        Returns:
            str: Decoded transcription
        """
        # Apply softmax to get probabilities
        probs = F.softmax(predictions.squeeze(), dim=-1)
        
        # Greedy decoding (take the highest probability class)
        decoded_indices = torch.argmax(probs, dim=-1)
        
        # Remove duplicates and blanks
        unique_indices = []
        prev_idx = None
        for idx in decoded_indices:
            if idx != prev_idx and idx < len(self.labels):
                unique_indices.append(idx)
                prev_idx = idx
        
        # Convert to text
        transcription = ''.join([self.labels[p] for p in unique_indices])
        
        return transcription
    
    def transcribe(self, 
                   audio: Optional[np.ndarray] = None, 
                   use_custom_model: bool = True) -> Dict[str, str]:
        """
        Transcribe audio using multiple models.
        
        Args:
            audio (np.ndarray, optional): Input audio
            use_custom_model (bool): Flag to use custom model
        
        Returns:
            Dict of transcription results
        """
        # Record audio if not provided
        if audio is None:
            audio = self.record_audio()
        
        results = {}
        
        # Custom Model Transcription
        if use_custom_model:
            _, features = self.preprocess_audio(audio)
            
            with torch.no_grad():
                predictions = self.model(features)
                custom_transcription = self.decode_predictions(predictions)
                results['custom_model'] = custom_transcription
        
        # Wav2Vec2 Transcription
        inputs = self.wav2vec_processor(
            audio, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        with torch.no_grad():
            logits = self.wav2vec_model(
                inputs.input_values, 
                attention_mask=inputs.attention_mask
            ).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            wav2vec_transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0]
            results['wav2vec_model'] = wav2vec_transcription
        
        return results

def demonstrate_language_support():
    """
    Demonstrate transcription capabilities across multiple languages.
    """
    # Supported languages
    languages = ['english', 'hindi', 'tamil']
    
    # Transcription results collection
    all_transcriptions = {}
    
    for language in languages:
        print(f"\n--- {language.capitalize()} Transcription Demonstration ---")
        try:
            # Initialize Transcription System for each language
            transcription_system = SpeechTranscriptionSystem(language=language)
            
            # Simulate or placeholder for audio recording
            print(f"Simulating audio recording in {language}...")
            
            # For demonstration, create a placeholder audio signal
            # In real-world scenario, this would be actual recorded audio
            sample_rate = 16000
            duration = 3  # 3 seconds
            
            # Generate a simple test signal (sine wave)
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            
            # Transcribe the test audio
            transcriptions = transcription_system.transcribe(test_audio)
            
            # Store results
            all_transcriptions[language] = transcriptions
            
            # Print Results
            print("\n--- Transcription Results ---")
            for model, text in transcriptions.items():
                print(f"{language.capitalize()} - {model.replace('_', ' ').title()}: {text}")
        
        except Exception as e:
            print(f"Error with {language} transcription: {e}")
    
    return all_transcriptions

def check_dependencies():
    """
    Check and print the versions of critical dependencies.
    """
    dependencies = [
        ('numpy', np),
        ('torch', torch),
        ('librosa', librosa),
        ('sounddevice', sd),
        ('transformers', None),
        ('pyworld', pw)
    ]
    
    print("\n--- Dependency Versions ---")
    for name, module in dependencies:
        try:
            version = module.__version__ if module else 'N/A'
            print(f"{name}: {version}")
        except Exception as e:
            print(f"{name}: Unable to retrieve version - {e}")

def main():
    """
    Main entry point for the Multilingual Speech Transcription System.
    """
    print("Multilingual Speech Transcription System")
    print("-------------------------------------")
    
    # Check system dependencies
    check_dependencies()
    
    # Demonstrate language support
    results = demonstrate_language_support()
    
    # Optional: Additional analysis or processing of transcription results
    print("\n--- Transcription Summary ---")
    for language, transcriptions in results.items():
        print(f"{language.capitalize()} Transcriptions:")
        for model, text in transcriptions.items():
            print(f"  - {model}: {len(text)} characters")

if __name__ == "__main__":
    main()
