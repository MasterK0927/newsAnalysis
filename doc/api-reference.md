# API Reference - News Analysis Application

## Table of Contents
- [Module Overview](#module-overview)
- [Core Functions](#core-functions)
- [Image Processing API](#image-processing-api)
- [Speech Processing API](#speech-processing-api)
- [NLP Analysis API](#nlp-analysis-api)
- [RAG System API](#rag-system-api)
- [News Scraping API](#news-scraping-api)
- [Database API](#database-api)
- [Visualization API](#visualization-api)
- [Error Handling](#error-handling)

## Module Overview

The News Analysis application is structured into several core modules, each providing specific functionality through well-defined APIs.

### Module Structure
```
news_analysis/
├── main.py                 # Main Streamlit application
├── image_processing.py     # OCR and image handling
├── speech_utils.py         # Speech recognition system
├── analysis.py             # NLP analysis engine
├── text_processing.py      # Text processing utilities
├── rag.py                  # RAG system implementation
├── news_scrapper.py        # Web scraping functionality
├── database.py             # Vector database operations
└── visualization.py        # Visualization utilities
```

## Core Functions

### main.py

#### `main()`
Main application entry point that initializes the Streamlit interface.

```python
def main() -> None:
    """
    Main application entry point.
    Initializes Streamlit configuration and coordinates all components.

    Returns:
        None
    """
```

**Features:**
- Sets up Streamlit page configuration
- Loads speech models
- Coordinates UI components
- Manages user interactions

#### `display_analysis_results(analysis_result)`
Displays comprehensive analysis results in the UI.

```python
def display_analysis_results(analysis_result: Dict[str, Any]) -> None:
    """
    Display analysis results in structured UI format.

    Args:
        analysis_result (Dict[str, Any]): Results from advanced_news_analysis()
            Required keys:
            - text: Original extracted text
            - summary: Article summary
            - entities: Named entities
            - sentiment: (label, confidence) tuple
            - related_news: List of related articles
            - trend_analysis: Trending topics

    Returns:
        None
    """
```

#### `handle_user_questions(analysis_result, speech_model)`
Manages user question input through text or speech.

```python
def handle_user_questions(
    analysis_result: Dict[str, Any],
    speech_model: Any
) -> None:
    """
    Handle user questions using text or speech input.

    Args:
        analysis_result: Analysis results containing context
        speech_model: Loaded speech recognition model

    Returns:
        None
    """
```

## Image Processing API

### image_processing.py

#### `extract_text(image)`
Extract text from images using OCR.

```python
def extract_text(image: np.ndarray) -> str:
    """
    Extract text from image using OCR with preprocessing.

    Args:
        image (np.ndarray): Input image array (OpenCV format)

    Returns:
        str: Extracted text content

    Raises:
        OCRError: If text extraction fails
        ImageProcessingError: If image preprocessing fails

    Example:
        >>> import cv2
        >>> image = cv2.imread('newspaper.jpg')
        >>> text = extract_text(image)
        >>> print(text)
        "Breaking News: Local Election Results..."
    """
```

**Implementation Details:**
- Converts image to grayscale
- Applies noise reduction
- Uses Tesseract OCR engine
- Handles multiple text orientations

## Speech Processing API

### speech_utils.py

#### `load_speech_model(model_path)`
Load and initialize speech recognition model.

```python
def load_speech_model(model_path: Optional[str] = None) -> Any:
    """
    Load speech recognition model (custom or Wav2Vec2).

    Args:
        model_path (Optional[str]): Path to custom model file
            If None, uses default Wav2Vec2 model

    Returns:
        Any: Loaded speech model object

    Raises:
        ModelLoadError: If model loading fails
        FileNotFoundError: If custom model path not found

    Example:
        >>> model = load_speech_model()
        >>> # or with custom model
        >>> model = load_speech_model('./models/custom_model.pt')
    """
```

#### `record_audio(duration, sample_rate)`
Record audio from microphone.

```python
def record_audio(
    duration: float = 5.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Record audio from default microphone.

    Args:
        duration (float): Recording duration in seconds. Default: 5.0
        sample_rate (int): Audio sample rate in Hz. Default: 16000

    Returns:
        np.ndarray: Audio data as numpy array

    Raises:
        AudioRecordingError: If recording fails
        DeviceNotFoundError: If microphone not available

    Example:
        >>> audio = record_audio(duration=10.0)
        >>> print(f"Recorded {len(audio)/16000:.1f} seconds of audio")
    """
```

#### `speech_to_text(audio, sample_rate, model)`
Convert audio to text using custom model.

```python
def speech_to_text(
    audio: np.ndarray,
    sample_rate: int,
    model: Any
) -> str:
    """
    Convert audio to text using custom speech model.

    Args:
        audio (np.ndarray): Audio data array
        sample_rate (int): Audio sample rate
        model: Loaded speech recognition model

    Returns:
        str: Transcribed text

    Raises:
        TranscriptionError: If transcription fails
        InvalidAudioError: If audio format is invalid

    Example:
        >>> model = load_speech_model()
        >>> audio = record_audio()
        >>> text = speech_to_text(audio, 16000, model)
        >>> print(text)
        "What is the main topic of this article?"
    """
```

#### `wav2vec2_speech_to_text(audio, sample_rate)`
Convert audio to text using Wav2Vec2 model.

```python
def wav2vec2_speech_to_text(
    audio: np.ndarray,
    sample_rate: int
) -> str:
    """
    Convert audio to text using Wav2Vec2 model.

    Args:
        audio (np.ndarray): Audio data array
        sample_rate (int): Audio sample rate (must be 16000)

    Returns:
        str: Transcribed text

    Raises:
        TranscriptionError: If transcription fails
        SampleRateError: If sample rate is not 16000

    Example:
        >>> audio = record_audio()
        >>> text = wav2vec2_speech_to_text(audio, 16000)
    """
```

## NLP Analysis API

### analysis.py

#### `advanced_news_analysis(image, user_query)`
Comprehensive news analysis pipeline.

```python
def advanced_news_analysis(
    image: np.ndarray,
    user_query: str
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on news image.

    Args:
        image (np.ndarray): Input image containing news content
        user_query (str): Analysis prompt/query

    Returns:
        Dict[str, Any]: Analysis results containing:
            - text (str): Extracted and preprocessed text
            - summary (str): Article summary
            - entities (Dict): Named entities by category
            - sentiment (Tuple[str, float]): Sentiment label and confidence
            - related_news (List[Dict]): Related news articles
            - trend_analysis (Tuple): Trending topics (up, down)
            - context_embedding (np.ndarray): Vector embedding for context

    Raises:
        AnalysisError: If analysis pipeline fails

    Example:
        >>> import cv2
        >>> image = cv2.imread('news_article.jpg')
        >>> result = advanced_news_analysis(image, "Analyze this article")
        >>> print(result['summary'])
        >>> print(result['sentiment'])
    """
```

#### `create_context_embedding(text, related_news)`
Generate context embeddings from text and related articles.

```python
def create_context_embedding(
    text: str,
    related_news: List[Dict[str, str]]
) -> np.ndarray:
    """
    Create combined embedding from main text and related news.

    Args:
        text (str): Main article text
        related_news (List[Dict]): Related news articles with 'title' and 'description'

    Returns:
        np.ndarray: Combined context embedding vector

    Example:
        >>> text = "Main article content..."
        >>> related = [{"title": "Related Article", "description": "..."}]
        >>> embedding = create_context_embedding(text, related)
    """
```

### text_processing.py

#### `preprocess_text(text)`
Clean and preprocess text for analysis.

```python
def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for NLP analysis.

    Args:
        text (str): Raw text input

    Returns:
        str: Cleaned and preprocessed text

    Processing Steps:
        - Remove extra whitespace
        - Fix encoding issues
        - Remove special characters
        - Normalize text format
    """
```

#### `extract_entities(text)`
Extract named entities from text.

```python
def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities using spaCy NLP model.

    Args:
        text (str): Input text for entity extraction

    Returns:
        Dict[str, List[str]]: Entities grouped by type
            Keys: 'PERSON', 'ORG', 'GPE', 'DATE', etc.

    Example:
        >>> text = "John Smith from Apple Inc. visited New York yesterday."
        >>> entities = extract_entities(text)
        >>> print(entities)
        {
            'PERSON': ['John Smith'],
            'ORG': ['Apple Inc.'],
            'GPE': ['New York'],
            'DATE': ['yesterday']
        }
    """
```

#### `get_sentiment(text)`
Analyze text sentiment.

```python
def get_sentiment(text: str) -> Tuple[str, float]:
    """
    Analyze sentiment of input text.

    Args:
        text (str): Text for sentiment analysis

    Returns:
        Tuple[str, float]: Sentiment label and confidence score
            Labels: 'positive', 'negative', 'neutral'
            Confidence: 0.0 to 1.0

    Example:
        >>> text = "This is a wonderful article about great achievements."
        >>> sentiment, confidence = get_sentiment(text)
        >>> print(f"Sentiment: {sentiment} ({confidence:.2f})")
        Sentiment: positive (0.89)
    """
```

#### `summarize_news(text, max_length)`
Generate text summary.

```python
def summarize_news(text: str, max_length: int = 150) -> str:
    """
    Generate abstractive summary of news text.

    Args:
        text (str): Input text to summarize
        max_length (int): Maximum summary length. Default: 150

    Returns:
        str: Generated summary

    Example:
        >>> long_text = "Very long news article content..."
        >>> summary = summarize_news(long_text, max_length=100)
        >>> print(summary)
    """
```

## RAG System API

### rag.py

#### `query_context(user_query, context_embedding)`
Retrieve relevant context for user queries.

```python
def query_context(
    user_query: str,
    context_embedding: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Retrieve contextually relevant information for user query.

    Args:
        user_query (str): User's question
        context_embedding (np.ndarray): Context vector embedding

    Returns:
        List[Dict[str, Any]]: Relevant context results with metadata
            Each dict contains:
            - score (float): Relevance score
            - text (str): Context text
            - metadata (Dict): Additional information

    Example:
        >>> query = "What is the main economic impact?"
        >>> results = query_context(query, context_embedding)
        >>> for result in results:
        ...     print(f"Score: {result['score']:.2f}")
        ...     print(f"Text: {result['text'][:100]}...")
    """
```

#### `generate_rag_response(user_query, original_text, context_results)`
Generate contextual responses using RAG.

```python
def generate_rag_response(
    user_query: str,
    original_text: str,
    context_results: List[Dict[str, Any]]
) -> str:
    """
    Generate response using retrieved context and original text.

    Args:
        user_query (str): User's question
        original_text (str): Original article text
        context_results (List[Dict]): Retrieved context from query_context()

    Returns:
        str: Generated response incorporating context

    Raises:
        GenerationError: If response generation fails
        TokenLimitError: If context exceeds token limits

    Example:
        >>> query = "Who are the main people mentioned?"
        >>> response = generate_rag_response(query, article_text, context)
        >>> print(response)
    """
```

## News Scraping API

### news_scrapper.py

#### `fetch_related_news(query, max_results, language)`
Fetch related news articles.

```python
def fetch_related_news(
    query: str,
    max_results: int = 5,
    language: str = 'en'
) -> List[Dict[str, str]]:
    """
    Fetch related news articles based on query.

    Args:
        query (str): Search query for related news
        max_results (int): Maximum articles to return. Default: 5
        language (str): Language code for results. Default: 'en'

    Returns:
        List[Dict[str, str]]: List of article dictionaries
            Each dict contains:
            - title (str): Article title
            - url (str): Article URL
            - description (str): Article description
            - source (str): News source
            - published_date (str): Publication date

    Raises:
        ScrapingError: If news fetching fails
        NetworkError: If network connection fails

    Example:
        >>> articles = fetch_related_news("climate change", max_results=3)
        >>> for article in articles:
        ...     print(f"Title: {article['title']}")
        ...     print(f"Source: {article['source']}")
    """
```

#### `scrape_article_content(url, max_length)`
Scrape content from news article URLs.

```python
def scrape_article_content(
    url: str,
    max_length: int = 2000
) -> str:
    """
    Scrape and extract content from news article URL.

    Args:
        url (str): Article URL to scrape
        max_length (int): Maximum content length. Default: 2000

    Returns:
        str: Extracted article content

    Raises:
        ScrapingError: If content extraction fails
        URLError: If URL is invalid or inaccessible

    Example:
        >>> url = "https://news-site.com/article"
        >>> content = scrape_article_content(url)
        >>> print(content[:200] + "...")
    """
```

#### `gather_comprehensive_info(query, aggregation_mode, max_articles)`
Comprehensive information gathering.

```python
def gather_comprehensive_info(
    query: str,
    aggregation_mode: str = 'comprehensive',
    max_articles: int = 3
) -> Dict[str, Any]:
    """
    Gather comprehensive information from multiple sources.

    Args:
        query (str): Information query
        aggregation_mode (str): Mode of aggregation
            Options: 'comprehensive', 'concise', 'detailed'
        max_articles (int): Maximum articles to process

    Returns:
        Dict[str, Any]: Comprehensive information package
            Contains:
            - articles (List): Article information
            - summary (str): Aggregated summary
            - key_points (List): Key information points
            - sources (List): Source attribution

    Example:
        >>> info = gather_comprehensive_info("renewable energy")
        >>> print(info['summary'])
        >>> print(f"Sources: {len(info['sources'])}")
    """
```

## Database API

### database.py

#### `generate_embedding(text)`
Generate vector embeddings for text.

```python
def generate_embedding(text: str) -> np.ndarray:
    """
    Generate vector embedding for input text.

    Args:
        text (str): Input text for embedding

    Returns:
        np.ndarray: Vector embedding array

    Example:
        >>> text = "This is a sample text for embedding"
        >>> embedding = generate_embedding(text)
        >>> print(f"Embedding shape: {embedding.shape}")
    """
```

#### `store_embedding(embedding, metadata)`
Store embeddings in vector database.

```python
def store_embedding(
    embedding: np.ndarray,
    metadata: Dict[str, Any]
) -> str:
    """
    Store embedding in vector database with metadata.

    Args:
        embedding (np.ndarray): Vector embedding to store
        metadata (Dict[str, Any]): Associated metadata

    Returns:
        str: Unique identifier for stored embedding

    Raises:
        DatabaseError: If storage operation fails

    Example:
        >>> embedding = generate_embedding("text")
        >>> metadata = {"source": "news_article", "date": "2024-01-01"}
        >>> id = store_embedding(embedding, metadata)
        >>> print(f"Stored with ID: {id}")
    """
```

## Visualization API

### visualization.py

#### `generate_word_cloud(text)`
Generate word cloud visualization.

```python
def generate_word_cloud(text: str) -> matplotlib.figure.Figure:
    """
    Generate word cloud visualization from text.

    Args:
        text (str): Input text for word cloud

    Returns:
        matplotlib.figure.Figure: Word cloud plot figure

    Example:
        >>> text = "news analysis word cloud visualization"
        >>> fig = generate_word_cloud(text)
        >>> fig.savefig('wordcloud.png')
    """
```

#### `format_google_news_url(url)`
Format Google News URLs for display.

```python
def format_google_news_url(url: str) -> str:
    """
    Format Google News URL to extract readable title.

    Args:
        url (str): Google News URL

    Returns:
        str: Formatted title string

    Example:
        >>> url = "https://news.google.com/articles/..."
        >>> title = format_google_news_url(url)
        >>> print(title)
    """
```

## Error Handling

### Custom Exception Classes

```python
class NewsAnalysisError(Exception):
    """Base exception for News Analysis application."""
    pass

class OCRError(NewsAnalysisError):
    """OCR processing failed."""
    pass

class ModelLoadError(NewsAnalysisError):
    """Model loading failed."""
    pass

class TranscriptionError(NewsAnalysisError):
    """Speech transcription failed."""
    pass

class AnalysisError(NewsAnalysisError):
    """NLP analysis failed."""
    pass

class ScrapingError(NewsAnalysisError):
    """Web scraping failed."""
    pass

class DatabaseError(NewsAnalysisError):
    """Database operation failed."""
    pass
```

### Error Handling Best Practices

1. **Always handle exceptions** in user-facing functions
2. **Log errors** with appropriate detail level
3. **Provide fallback mechanisms** where possible
4. **Return meaningful error messages** to users
5. **Use specific exception types** for different error categories

### Example Error Handling

```python
def safe_analysis_wrapper(image, query):
    """Example of proper error handling."""
    try:
        return advanced_news_analysis(image, query)
    except OCRError as e:
        logger.error(f"OCR failed: {e}")
        return {"error": "Text extraction failed", "details": str(e)}
    except AnalysisError as e:
        logger.error(f"Analysis failed: {e}")
        return {"error": "Analysis failed", "details": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "Unexpected error occurred", "details": "Please try again"}
```

---

*This API reference is automatically generated and maintained. For the latest updates, refer to the source code and inline documentation.*