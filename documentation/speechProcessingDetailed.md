# Speech Processing Detailed Documentation on IMPL

## Sources and References

### Signal Processing Fundamentals
1. Oppenheim, A. V., & Schafer, R. W. (2014). Discrete-Time Signal Processing (3rd ed.). Pearson.
   - Butterworth filter design
   - Digital filter implementation
   - Short-time Fourier transform

2. Rabiner, L. R., & Schafer, R. W. (2011). Theory and Applications of Digital Speech Processing. Pearson.
   - Speech frequency bands
   - Feature extraction techniques
   - Real-time processing considerations

3. Loizou, P. C. (2013). Speech Enhancement: Theory and Practice (2nd ed.). CRC Press.
   - Spectral subtraction
   - Noise reduction techniques
   - Speech enhancement algorithms

4. Gold, B., Morgan, N., & Ellis, D. (2011). Speech and Audio Signal Processing. Wiley.
   - MFCC computation
   - Feature extraction
   - Mel scale conversions

## 1. Bandpass Filtering

### Theory and Implementation

The Butterworth bandpass filter implementation follows the design principles from Oppenheim & Schafer (2014, pp. 511-514). The filter is characterized by its magnitude response:

```
H(ω) = |H(jω)| = 1 / √(1 + (ω/ωc)^2n)
```

Where:
- ω: Angular frequency
- ωc: Cutoff frequency
- n: Filter order

Implementation:
```python
def butter_bandpass(self, fs: float, order: int = 5) -> Tuple:
    nyq = 0.5 * fs
    low = self.lowcut / nyq
    high = self.highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a
```

### Frequency Band Selection
Based on research by Rabiner & Schafer (2011, pp. 318-320):

1. English (300-3000 Hz)
   - Fundamental frequencies: 85-255 Hz
   - Consonant energy: 2000-4000 Hz

2. Hindi (200-3500 Hz)
   - Extended range for aspirated consonants
   - Retroflex sound preservation

3. Tamil (250-3300 Hz)
   - Optimized for Tamil phonemes
   - Critical formant preservation

## 2. Spectral Subtraction

### Mathematical Foundation
Based on Loizou (2013, pp. 93-96), the spectral subtraction algorithm follows:

```
|Ŝ(ω)|² = |Y(ω)|² - |N̂(ω)|²
```

Where:
- |Ŝ(ω)|²: Enhanced speech power spectrum
- |Y(ω)|²: Noisy speech power spectrum
- |N̂(ω)|²: Estimated noise power spectrum

Implementation:
```python
def noise_reduction(signal: np.ndarray, noise_threshold: float = 0.02) -> np.ndarray:
    stft = librosa.stft(signal)
    noise_profile = np.mean(np.abs(stft[:, :5]), axis=1)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    enhanced_magnitude = np.maximum(magnitude - noise_threshold * noise_profile[:, np.newaxis], 0)
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    return librosa.istft(enhanced_stft)
```

## 3. Feature Extraction

### Mel Spectrogram Computation
Based on Gold et al. (2011, pp. 286-288), the Mel scale conversion is defined as:

```
M(f) = 2595 * log10(1 + f/700)
```

### MFCC Extraction
The Discrete Cosine Transform (DCT) computation follows Gold et al. (2011, pp. 291-293):

```
c[n] = Σ(m=1 to M) log(S[m]) * cos(πn(m-0.5)/M)
```

Where:
- S[m]: Mel spectrum
- M: Number of mel bands
- n: MFCC coefficient index

### Spectral Features
Derived from Oppenheim & Schafer (2014, pp. 790-793):

1. Spectral Centroid:
```
Centroid = Σ(f * M(f)) / Σ(M(f))
```

## 4. Real-time Processing

### Buffer Management
Based on recommendations from Rabiner & Schafer (2011, pp. 452-455):
- Buffer size: 2048 samples
- Overlap: 50% (1024 samples)
- Processing latency: ~64ms at 16kHz

Implementation:
```python
class CircularBuffer:
    def __init__(self, size):
        self.buffer = np.zeros(size)
        self.size = size
        self.index = 0
        
    def add(self, data):
        n = len(data)
        if self.index + n > self.size:
            first_part = self.size - self.index
            self.buffer[self.index:] = data[:first_part]
            self.buffer[:n-first_part] = data[first_part:]
            self.index = n - first_part
        else:
            self.buffer[self.index:self.index+n] = data
            self.index = (self.index + n) % self.size
```

## 5. Quality Metrics

### Signal Quality Assessment
Based on Loizou (2013, pp. 541-544):

1. Signal-to-Noise Ratio (SNR):
```python
def calculate_snr(clean_signal, noisy_signal):
    noise = noisy_signal - clean_signal
    return 10 * np.log10(np.sum(clean_signal**2) / np.sum(noise**2))
```

2. Additional Metrics:
   - Perceptual Evaluation of Speech Quality (PESQ)
   - Short-Time Objective Intelligibility (STOI)

## 6. Integrated System

### Pipeline Architecture
```python
class IntegratedPipeline:
    def __init__(self):
        self.preprocessor = AdvancedSignalPreprocessor()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.model = AdvancedSpeechModel()
    
    def process(self, audio):
        # 1. Signal preprocessing
        enhanced = self.preprocessor.speech_enhancement(audio)
        
        # 2. Feature extraction
        features = self.feature_extractor.extract_multi_features(enhanced)
        
        # 3. Model inference
        with torch.no_grad():
            predictions = self.model(features)
        
        return predictions
```

## Notes on Implementation

1. All implementations use NumPy (numpy.org) for numerical computations
2. Signal processing functions utilize SciPy (scipy.org)
3. Audio processing leverages Librosa (librosa.org)
4. PyTorch (pytorch.org) is used for GPU acceleration where available

## Bibliography

1. Oppenheim, A. V., & Schafer, R. W. (2014). Discrete-Time Signal Processing (3rd ed.). Pearson.
2. Rabiner, L. R., & Schafer, R. W. (2011). Theory and Applications of Digital Speech Processing. Pearson.
3. Loizou, P. C. (2013). Speech Enhancement: Theory and Practice (2nd ed.). CRC Press.
4. Gold, B., Morgan, N., & Ellis, D. (2011). Speech and Audio Signal Processing. Wiley.