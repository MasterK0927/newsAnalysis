# Installation Guide - News Analysis Application

## Table of Contents
- [System Requirements](#system-requirements)
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Docker Setup](#docker-setup)

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 18+), macOS 10.15+, Windows 10
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Storage**: 10GB free space
- **Internet**: Required for initial setup and news scraping

### Recommended Requirements
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (for speech processing)
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **Storage**: 20GB free space (SSD recommended)

## Prerequisites

### System Dependencies

#### Ubuntu/Debian
```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    build-essential
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install portaudio
brew install tesseract
brew install ffmpeg
brew install espeak
```

#### Fedora/CentOS/RHEL
```bash
# Install system dependencies
sudo dnf install -y \
    python3 \
    python3-pip \
    python3-devel \
    portaudio-devel \
    espeak \
    espeak-devel \
    ffmpeg \
    tesseract \
    tesseract-devel \
    leptonica-devel
```

### Python Environment Setup

#### Using pip (Recommended)
```bash
# Create virtual environment
python3 -m venv news_analysis_env

# Activate virtual environment
# Linux/macOS:
source news_analysis_env/bin/activate
# Windows:
# news_analysis_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Using conda (Alternative)
```bash
# Create conda environment
conda create -n news_analysis python=3.9
conda activate news_analysis
```

## Installation Steps

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/newsAnalysis.git
cd newsAnalysis
```

### Step 2: Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Install additional ML models
python -m spacy download en_core_web_sm
```

#### Manual Package Installation (if requirements.txt fails)
```bash
# Core packages
pip install streamlit
pip install opencv-python
pip install numpy
pip install pytesseract

# ML and NLP packages
pip install torch torchvision torchaudio
pip install transformers
pip install spacy
pip install scikit-learn

# Audio processing
pip install librosa
pip install sounddevice
pip install scipy

# Web scraping
pip install beautifulsoup4
pip install requests

# Database and vector storage
pip install pinecone-client
pip install langchain
pip install langchain-community

# Visualization
pip install matplotlib
pip install wordcloud
pip install pandas

# Utilities
pip install python-dotenv
pip install openai
```

### Step 3: Download Language Models

#### spaCy Models
```bash
# English model (required)
python -m spacy download en_core_web_sm

# Additional language models (optional)
python -m spacy download en_core_web_lg  # Larger English model
python -m spacy download de_core_news_sm  # German
python -m spacy download fr_core_news_sm  # French
```

#### Hugging Face Models
The application will automatically download required Hugging Face models on first use:
- `facebook/wav2vec2-base-960h` (Speech recognition)
- Model files will be cached in `~/.cache/huggingface/`

### Step 4: Set Up Configuration

#### Create Environment File
```bash
# Copy example environment file
cp .env.example .env

# Edit with your configurations
nano .env
```

#### Example .env Configuration
```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment

# Application Settings
DEBUG=False
LOG_LEVEL=INFO

# Speech Processing
SPEECH_MODEL_PATH=./models/custom_speech_model.pt
AUDIO_SAMPLE_RATE=16000

# News Scraping
MAX_NEWS_ARTICLES=5
SCRAPING_TIMEOUT=30
USER_AGENT=NewsAnalyzer/1.0
```

## Configuration

### API Keys Setup

#### OpenAI API Key
1. Visit [OpenAI API Platform](https://platform.openai.com/)
2. Create account and generate API key
3. Add key to `.env` file: `OPENAI_API_KEY=sk-...`

#### Pinecone Vector Database
1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Create a new project and index
3. Get API key and environment details
4. Add to `.env` file:
   ```env
   PINECONE_API_KEY=your-api-key
   PINECONE_ENVIRONMENT=your-environment
   PINECONE_INDEX_NAME=news-analysis
   ```

### Model Configuration

#### Custom Speech Model (Optional)
If you have a custom speech model:
```bash
# Create models directory
mkdir -p models

# Place your custom model file
cp your_custom_model.pt models/custom_speech_model.pt
```

#### GPU Configuration
For NVIDIA GPU support:
```bash
# Install CUDA toolkit (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvidia-smi

# Install PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification

### Step 1: Test Installation
```bash
# Run system check script
python -c "
import streamlit
import cv2
import numpy
import torch
import transformers
import spacy
import requests
import bs4
print('All core packages imported successfully!')
"
```

### Step 2: Test Individual Components

#### Test OCR
```bash
python -c "
import cv2
import pytesseract
print('Tesseract version:', pytesseract.get_tesseract_version())
"
```

#### Test Speech Processing
```bash
python -c "
import sounddevice as sd
import librosa
import scipy
print('Audio processing libraries loaded successfully')
"
```

#### Test NLP Models
```bash
python -c "
import spacy
nlp = spacy.load('en_core_web_sm')
print('spaCy model loaded successfully')
"
```

### Step 3: Run Application
```bash
# Start the application
streamlit run main.py

# Application should open in browser at http://localhost:8501
```

## Troubleshooting

### Common Issues

#### 1. Tesseract Not Found
```bash
# Error: TesseractNotFoundError
# Solution: Install tesseract and set path

# Linux
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

#### 2. PortAudio Issues (Speech Recording)
```bash
# Error: No module named '_portaudio'
# Solution: Install system audio libraries

# Ubuntu/Debian
sudo apt-get install portaudio19-dev

# macOS
brew install portaudio

# Then reinstall PyAudio
pip uninstall pyaudio
pip install pyaudio
```

#### 3. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Model Download Issues
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Re-download models
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('facebook/wav2vec2-base-960h')"
```

#### 5. Memory Issues
```bash
# Reduce model size in code or increase system memory
# Monitor memory usage:
htop  # Linux
top   # macOS
```

### Performance Optimization

#### For Low-Memory Systems
Edit main configuration to use smaller models:
```python
# In speech_utils.py, use base model instead of large
MODEL_NAME = "facebook/wav2vec2-base-960h"  # Instead of large variant
```

#### For CPU-Only Systems
```python
# Force CPU usage in model loading
device = torch.device("cpu")
```

## Docker Setup

### Using Docker (Alternative Installation)

#### Prerequisites
```bash
# Install Docker
# Linux: https://docs.docker.com/engine/install/
# macOS/Windows: https://docs.docker.com/desktop/
```

#### Build and Run
```bash
# Build Docker image
docker build -t news-analysis .

# Run container with GPU support (if available)
docker run --gpus all -p 8501:8501 -v $(pwd):/app news-analysis

# Run container (CPU only)
docker run -p 8501:8501 -v $(pwd):/app news-analysis
```

#### Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Post-Installation

### Initial Setup
1. **Test with sample image**: Upload a newspaper cutout to verify OCR
2. **Configure APIs**: Ensure OpenAI and Pinecone keys are working
3. **Test speech recognition**: Try recording and transcription
4. **Verify news scraping**: Check related news retrieval

### Updating
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Update language models
python -m spacy download en_core_web_sm --upgrade
```

### Maintenance
- **Regular Updates**: Keep dependencies updated monthly
- **Cache Cleanup**: Clear model caches periodically
- **Log Monitoring**: Check application logs for issues
- **API Usage**: Monitor OpenAI and Pinecone usage limits

---

**Need Help?** Check the [Troubleshooting Guide](troubleshooting.md) or create an issue on the project repository.