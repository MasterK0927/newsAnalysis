# Configuration Guide - News Analysis Application

## Table of Contents
- [Configuration Overview](#configuration-overview)
- [Environment Variables](#environment-variables)
- [API Configuration](#api-configuration)
- [Model Configuration](#model-configuration)
- [Application Settings](#application-settings)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Deployment Configuration](#deployment-configuration)

## Configuration Overview

The News Analysis application uses a combination of environment variables, configuration files, and runtime settings to customize behavior. Configuration is managed through:

- **Environment Variables**: API keys, external service settings
- **Configuration Files**: Application parameters, model settings
- **Runtime Configuration**: User preferences, temporary settings

### Configuration Hierarchy
```
1. Environment Variables (.env file)
2. Configuration Files (config.yaml, settings.json)
3. Command Line Arguments
4. Runtime Settings (Streamlit session state)
5. Default Values (hardcoded fallbacks)
```

## Environment Variables

### Setting Up Environment File

Create a `.env` file in the project root directory:

```bash
# Copy from example template
cp .env.example .env

# Edit with your configurations
nano .env
```

### Required Environment Variables

#### OpenAI Configuration
```env
# OpenAI API for language model integration
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.7
```

#### Pinecone Vector Database
```env
# Pinecone for vector storage and retrieval
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=news-analysis
PINECONE_DIMENSION=1536
```

### Optional Environment Variables

#### Application Settings
```env
# Application behavior
DEBUG=False
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=209715200  # 200MB in bytes
SESSION_TIMEOUT=3600       # 1 hour in seconds

# UI Configuration
APP_TITLE=Advanced News Analysis
APP_ICON=📰
PAGE_LAYOUT=wide
THEME=auto
```

#### Speech Processing
```env
# Speech recognition settings
SPEECH_MODEL_PATH=./models/custom_speech_model.pt
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
RECORDING_DURATION=5
SPEECH_THRESHOLD=0.01
```

#### Image Processing
```env
# OCR and image processing
TESSERACT_CMD=/usr/bin/tesseract
TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
OCR_LANGUAGE=eng
IMAGE_DPI=300
MAX_IMAGE_DIMENSION=2000
```

#### News Scraping
```env
# Web scraping configuration
MAX_NEWS_ARTICLES=5
SCRAPING_TIMEOUT=30
REQUEST_DELAY=1
USER_AGENT=NewsAnalyzer/1.0
RETRY_ATTEMPTS=3
CACHE_DURATION=3600
```

## API Configuration

### OpenAI API Setup

#### 1. Get API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API section
4. Generate new API key
5. Copy key to `.env` file

#### 2. Model Selection
```env
# Available models (as of 2024)
OPENAI_MODEL=gpt-3.5-turbo        # Fast, cost-effective
OPENAI_MODEL=gpt-4                # More capable, higher cost
OPENAI_MODEL=gpt-4-turbo         # Latest, balanced performance
```

#### 3. Usage Limits Configuration
```env
# Token limits and pricing control
OPENAI_MAX_TOKENS=4000           # Maximum tokens per request
OPENAI_MAX_REQUESTS_PER_MINUTE=3 # Rate limiting
DAILY_USAGE_LIMIT=10000          # Daily token limit
```

### Pinecone Configuration

#### 1. Account Setup
```bash
# Sign up at pinecone.io
# Create new project
# Note your environment (e.g., us-west1-gcp)
```

#### 2. Index Creation
```python
# Create index with proper dimensions
import pinecone

pinecone.init(
    api_key="your-api-key",
    environment="your-environment"
)

# Create index for news analysis
index_name = "news-analysis"
pinecone.create_index(
    name=index_name,
    dimension=1536,  # OpenAI embedding dimension
    metric="cosine"
)
```

#### 3. Environment Configuration
```env
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp  # Your specific environment
PINECONE_INDEX_NAME=news-analysis
PINECONE_NAMESPACE=default
```

### External API Integration

#### News APIs (Optional)
```env
# Google News API (if available)
GOOGLE_NEWS_API_KEY=your_google_news_key

# NewsAPI.org
NEWS_API_KEY=your_newsapi_key
NEWS_API_ENDPOINT=https://newsapi.org/v2/

# Custom news sources
CUSTOM_NEWS_SOURCES=reuters.com,bbc.com,cnn.com
```

## Model Configuration

### Speech Recognition Models

#### Custom Model Configuration
```yaml
# config/speech_models.yaml
speech_models:
  custom:
    model_path: "./models/custom_speech_model.pt"
    sample_rate: 16000
    chunk_duration: 30  # seconds
    language_support: ["en", "hi", "ta"]

  wav2vec2:
    model_name: "facebook/wav2vec2-base-960h"
    cache_dir: "./models/wav2vec2/"
    device: "auto"  # auto, cpu, cuda
```

#### Language-Specific Configuration
```yaml
language_configs:
  english:
    model: "facebook/wav2vec2-base-960h"
    sample_rate: 16000
    preprocessing:
      noise_reduction: true
      normalization: true

  hindi:
    model: "facebook/wav2vec2-large-xlsr-53-hindi"
    sample_rate: 16000
    preprocessing:
      bandpass_filter: [200, 3500]

  tamil:
    model: "facebook/wav2vec2-large-xlsr-53-tamil"
    sample_rate: 16000
    preprocessing:
      bandpass_filter: [250, 3300]
```

### NLP Model Configuration

#### spaCy Models
```yaml
# config/nlp_models.yaml
nlp_models:
  primary:
    name: "en_core_web_sm"
    components: ["tok2vec", "tagger", "parser", "ner"]

  large:
    name: "en_core_web_lg"
    use_for: ["similarity", "word_vectors"]

  multilingual:
    name: "xx_ent_wiki_sm"
    languages: ["en", "es", "fr", "de"]
```

#### Transformers Configuration
```yaml
transformers:
  sentiment:
    model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer: "cardiffnlp/twitter-roberta-base-sentiment-latest"

  summarization:
    model: "facebook/bart-large-cnn"
    max_length: 150
    min_length: 30

  embeddings:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dimension: 384
```

## Application Settings

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[global]
dataFrameSerialization = "legacy"

[server]
maxUploadSize = 200
maxMessageSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
serverAddress = "localhost"
serverPort = 8501
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Application Features
```yaml
# config/app_settings.yaml
features:
  image_processing:
    enabled: true
    max_file_size: 200MB
    supported_formats: [jpg, jpeg, png, tiff]
    ocr_languages: [eng, hin, spa, fra]

  speech_recognition:
    enabled: true
    models: [custom, wav2vec2]
    max_duration: 30  # seconds
    auto_stop_silence: 2  # seconds

  nlp_analysis:
    enabled: true
    entity_extraction: true
    sentiment_analysis: true
    summarization: true

  rag_system:
    enabled: true
    context_window: 4000
    max_results: 5
    similarity_threshold: 0.7

  news_scraping:
    enabled: true
    max_articles: 5
    timeout: 30
    respect_robots_txt: true

  visualization:
    word_clouds: true
    trend_analysis: true
    interactive_charts: true
```

### User Interface Settings
```yaml
ui_settings:
  layout:
    sidebar_width: 300
    main_width: 700
    columns: [2, 3]  # Ratio for entity/sentiment vs wordcloud

  colors:
    positive_sentiment: "#00ff00"
    negative_sentiment: "#ff0000"
    neutral_sentiment: "#ffff00"

  animations:
    loading_spinner: true
    progress_bars: true
    transitions: smooth
```

## Security Configuration

### API Security
```env
# API Security Settings
API_RATE_LIMIT=100              # Requests per hour
API_TIMEOUT=30                  # Seconds
API_RETRY_ATTEMPTS=3
API_BACKOFF_FACTOR=2

# Request headers
DEFAULT_USER_AGENT=NewsAnalyzer/1.0
ACCEPT_LANGUAGE=en-US,en;q=0.9
```

### Data Protection
```yaml
security:
  data_retention:
    temporary_files: 24h
    cache_duration: 1h
    session_data: 30min

  sanitization:
    input_validation: strict
    output_filtering: enabled
    xss_protection: true

  privacy:
    log_user_data: false
    analytics: disabled
    error_reporting: minimal
```

### Content Security
```yaml
content_security:
  allowed_domains:
    - "*.openai.com"
    - "*.pinecone.io"
    - "news.google.com"
    - "*.reuters.com"
    - "*.bbc.com"

  blocked_patterns:
    - "javascript:"
    - "data:"
    - "file:"

  file_validation:
    check_magic_bytes: true
    scan_malware: false  # Set true if antivirus available
    size_limits: enforced
```

## Performance Tuning

### Memory Management
```yaml
memory:
  model_cache_size: 2GB
  image_cache_size: 512MB
  text_cache_size: 256MB

  garbage_collection:
    frequency: 100  # requests
    aggressive_mode: false

  memory_limits:
    per_request: 1GB
    total_application: 4GB
```

### Processing Optimization
```yaml
processing:
  parallel_workers: 4
  batch_sizes:
    text_processing: 32
    image_processing: 1
    speech_processing: 1

  timeouts:
    ocr_processing: 60s
    speech_recognition: 30s
    api_requests: 30s

  caching:
    enable_model_cache: true
    enable_result_cache: true
    cache_ttl: 3600s
```

### GPU Configuration
```yaml
gpu:
  enable_gpu: auto  # auto, true, false
  memory_fraction: 0.7
  allow_growth: true

  device_placement:
    speech_models: gpu
    nlp_models: gpu
    embeddings: gpu

  fallback_cpu: true
```

## Deployment Configuration

### Production Settings
```env
# Production environment
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=WARNING

# Security
SECURE_COOKIES=True
FORCE_HTTPS=True
CSRF_PROTECTION=True

# Performance
ENABLE_CACHING=True
CACHE_BACKEND=redis
CACHE_URL=redis://localhost:6379/0

# Monitoring
ENABLE_METRICS=True
METRICS_ENDPOINT=/metrics
HEALTH_CHECK_ENDPOINT=/health
```

### Docker Configuration
```yaml
# docker-compose.yml environment section
environment:
  - NODE_ENV=production
  - PYTHONPATH=/app
  - STREAMLIT_SERVER_PORT=8501
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0

volumes:
  - ./models:/app/models:ro
  - ./config:/app/config:ro
  - /tmp/news_analysis:/app/temp
```

### Load Balancer Settings
```yaml
load_balancer:
  algorithm: round_robin
  health_checks:
    enabled: true
    interval: 30s
    timeout: 10s
    healthy_threshold: 2
    unhealthy_threshold: 3

  session_affinity: false
  timeout:
    connect: 5s
    server: 30s
    client: 60s
```

## Configuration Validation

### Validation Script
```python
# scripts/validate_config.py
import os
import yaml
from pathlib import Path

def validate_configuration():
    """Validate all configuration settings."""

    errors = []
    warnings = []

    # Check required environment variables
    required_env_vars = [
        'OPENAI_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_ENVIRONMENT'
    ]

    for var in required_env_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")

    # Check file paths
    required_paths = [
        'models/',
        'config/',
        '.streamlit/'
    ]

    for path in required_paths:
        if not Path(path).exists():
            warnings.append(f"Path does not exist: {path}")

    # Check model files
    model_files = [
        'models/custom_speech_model.pt'
    ]

    for model_file in model_files:
        if not Path(model_file).exists():
            warnings.append(f"Model file not found: {model_file}")

    # Return validation results
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

if __name__ == "__main__":
    result = validate_configuration()

    if result['valid']:
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
        for error in result['errors']:
            print(f"ERROR: {error}")

    for warning in result['warnings']:
        print(f"WARNING: {warning}")
```

### Environment Testing
```bash
# Test configuration
python scripts/validate_config.py

# Test API connections
python scripts/test_apis.py

# Test model loading
python scripts/test_models.py
```

## Troubleshooting Configuration

### Common Configuration Issues

#### 1. API Key Issues
```bash
# Test OpenAI API
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Test Pinecone
python -c "
import pinecone
pinecone.init(api_key='$PINECONE_API_KEY', environment='$PINECONE_ENVIRONMENT')
print('Pinecone connection successful')
"
```

#### 2. Model Loading Issues
```bash
# Check model paths
ls -la models/
ls -la ~/.cache/huggingface/

# Test model loading
python -c "
import spacy
nlp = spacy.load('en_core_web_sm')
print('spaCy model loaded successfully')
"
```

#### 3. Permission Issues
```bash
# Fix file permissions
chmod +r .env
chmod +r -R config/
chmod +r -R models/

# Check Tesseract permissions
ls -la $(which tesseract)
tesseract --version
```

---

*This configuration guide covers all aspects of setting up and customizing the News Analysis application. Update configuration files as needed for your specific deployment environment.*