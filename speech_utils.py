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

#########################################
# Language and Signal Processing Config #
#########################################

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
                'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ',
                'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
                'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
                'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श',
                'ष', 'स', 'ह', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै',
                'ो', 'ौ', '्', '<space>', '<blank>'
            ],
            'wav2vec_model': "ai4bharat/indicwav2vec_v1_hindi",
            'lowcut': 200,
            'highcut': 3500
        },
        'tamil': {
            'labels': [
                'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ',
                'க', 'ங', 'ச', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன',
                'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', '்',
                '<space>', '<blank>'
            ],
            'wav2vec_model': "ai4bharat/indicwav2vec_v1_tamil",
            'lowcut': 250,
            'highcut': 3300
        }
    }

    @classmethod
    def get_language_config(cls, language: str) -> Dict:
        language = language.lower()
        if language not in cls.LANGUAGE_CONFIGS:
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {list(cls.LANGUAGE_CONFIGS.keys())}"
            )
        return cls.LANGUAGE_CONFIGS[language]


class AdvancedSignalPreprocessor:
    def __init__(self, lowcut: float = 300, highcut: float = 3000):
        self.lowcut = lowcut
        self.highcut = highcut

    def butter_bandpass(self, fs: float, order: int = 5) -> Tuple:
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data: np.ndarray, fs: float, order: int = 5) -> np.ndarray:
        b, a = self.butter_bandpass(fs, order=order)
        return lfilter(b, a, data)

    @staticmethod
    def noise_reduction(signal_data: np.ndarray, noise_threshold: float = 0.02) -> np.ndarray:
        stft = librosa.stft(signal_data)
        noise_profile = np.mean(np.abs(stft[:, :5]), axis=1)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        enhanced_magnitude = np.maximum(magnitude - noise_threshold * noise_profile[:, np.newaxis], 0)
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        return librosa.istft(enhanced_stft)

    def speech_enhancement(self, signal_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        signal_data = signal_data.flatten()
        filtered_signal = self.butter_bandpass_filter(signal_data, sample_rate)
        noise_reduced = self.noise_reduction(filtered_signal)
        f0, t = pw.dio(noise_reduced.astype(np.float64), sample_rate)
        f0 = pw.stonemask(noise_reduced.astype(np.float64), f0, t, sample_rate)
        normalized_signal = librosa.util.normalize(noise_reduced)
        return normalized_signal


class AdvancedFeatureExtractor:
    @staticmethod
    def extract_multi_features(signal_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        signal_data = signal_data.flatten()
        # 1. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal_data, 
            sr=sample_rate, 
            n_mels=80,
            n_fft=2048,
            hop_length=512
        )
        log_mel_spec = librosa.power_to_db(mel_spec)
        # 2. MFCC Features
        mfccs = librosa.feature.mfcc(
            y=signal_data, 
            sr=sample_rate, 
            n_mfcc=13
        )
        # 3. Spectral Centroids
        spectral_centroids = librosa.feature.spectral_centroid(
            y=signal_data, 
            sr=sample_rate
        )
        # 4. Chroma Features
        chroma = librosa.feature.chroma_stft(
            y=signal_data, 
            sr=sample_rate
        )
        # 5. Harmonic-Percussive Source Separation
        harmonic, _ = librosa.effects.hpss(signal_data)
        features = np.concatenate([
            log_mel_spec.flatten(),
            mfccs.flatten(),
            spectral_centroids.flatten(),
            chroma.flatten(),
            harmonic.flatten()
        ])
        return features

#########################################
# Deep Model Components                 #
#########################################

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
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 29, num_classes: int = 29):
        super().__init__()
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=k, stride=2, padding=k // 2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for k in [3, 5, 7]
        ])
        self.lstm = nn.LSTM(
            input_size=128 * 3,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attention = SelfAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        self.ctc_layer = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_features = [
            conv(x.transpose(1, 2)).transpose(1, 2)
            for conv in self.feature_extractor
        ]
        combined_features = torch.cat(multi_scale_features, dim=-1)
        lstm_out, _ = self.lstm(combined_features)
        attended_features = self.attention(lstm_out)
        ctc_output = self.ctc_layer(attended_features)
        return ctc_output

#########################################
# Speech Transcription System           #
#########################################

class SpeechTranscriptionSystem:
    def __init__(self, language: str = 'english'):
        self.language_config = LanguageConfig.get_language_config(language)
        self.labels = self.language_config['labels']
        self.sample_rate = 16000
        self.preprocessor = AdvancedSignalPreprocessor(
            lowcut=self.language_config['lowcut'],
            highcut=self.language_config['highcut']
        )
        self.feature_extractor = AdvancedFeatureExtractor()
        self.model = AdvancedSpeechModel(
            input_dim=256,
            output_dim=len(self.labels),
            num_classes=len(self.labels)
        )
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            self.language_config['wav2vec_model']
        )
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(
            self.language_config['wav2vec_model']
        )

    def record_audio(self, duration: float = 3.0) -> np.ndarray:
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
        enhanced_signal = self.preprocessor.speech_enhancement(audio, sample_rate=self.sample_rate)
        features = self.feature_extractor.extract_multi_features(enhanced_signal, sample_rate=self.sample_rate)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        return enhanced_signal, features_tensor

    def decode_predictions(self, predictions: torch.Tensor) -> str:
        probs = F.softmax(predictions.squeeze(), dim=-1)
        decoded_indices = torch.argmax(probs, dim=-1)
        unique_indices = []
        prev_idx = None
        for idx in decoded_indices:
            if idx != prev_idx and idx < len(self.labels):
                unique_indices.append(idx)
                prev_idx = idx
        transcription = ''.join([self.labels[p] for p in unique_indices])
        return transcription

    def transcribe(self, audio: Optional[np.ndarray] = None, use_custom_model: bool = True) -> Dict[str, str]:
        if audio is None:
            audio = self.record_audio()
        results = {}
        if use_custom_model:
            _, features = self.preprocess_audio(audio)
            with torch.no_grad():
                predictions = self.model(features)
                custom_transcription = self.decode_predictions(predictions)
                results['custom_model'] = custom_transcription
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

#########################################
# Functions for main.py Imports           #
#########################################

def load_speech_model(language: str = 'english') -> SpeechTranscriptionSystem:
    """
    Load and return the speech transcription system model.
    """
    return SpeechTranscriptionSystem(language=language)


def record_audio(duration: float = 3.0) -> np.ndarray:
    """
    Record audio using the default English speech transcription system.
    """
    stt_system = SpeechTranscriptionSystem(language='english')
    return stt_system.record_audio(duration=duration)


def speech_to_text(audio: np.ndarray, sample_rate: int, speech_model: SpeechTranscriptionSystem) -> str:
    """
    Transcribe audio using the custom speech model.
    """
    transcriptions = speech_model.transcribe(audio, use_custom_model=True)
    return transcriptions.get('custom_model', '')


def wav2vec2_speech_to_text(audio: np.ndarray, sample_rate: int) -> str:
    """
    Transcribe audio using the Wav2Vec2 model.
    """
    stt_system = SpeechTranscriptionSystem(language='english')
    transcriptions = stt_system.transcribe(audio, use_custom_model=False)
    return transcriptions.get('wav2vec_model', '')


# Optional: for testing purposes only
if __name__ == "__main__":
    print("Testing Speech Transcription System")
    model = load_speech_model()
    audio_sample = record_audio(duration=3.0)
    print("Custom Model Transcription:", speech_to_text(audio_sample, 16000, model))
    print("Wav2Vec2 Model Transcription:", wav2vec2_speech_to_text(audio_sample, 16000))
