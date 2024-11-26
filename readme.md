# Developer Documentation: News Analysis Application
## Technical Stack & Architecture

### Core Technologies
- **Frontend**: Streamlit
- **Image Processing**: OpenCV, NumPy
- **Speech Recognition**: Custom model + Wav2Vec2
- **Analysis**: Custom NLP pipeline
- **Storage**: File-based (local)
- **RAG System**: Custom implementation

### Project Structure
```
news_analysis/
├── main.py                    # Main application entry
├── image_processing.py        # Image handling utilities
├── speech_processing.py       # Speech recognition models
├── analysis.py               # News analysis logic
├── visualization.py          # Data visualization tools
├── rag.py                    # RAG implementation
└── requirements.txt          # Dependencies
```

## Implementation Guide

### 1. Application Initialization
```python
# main.py
def main():
    # Page config setup - Always runs first
    st.set_page_config(page_title="Advanced News Analysis", 
                      page_icon="📰", 
                      layout="wide")
    
    # Load speech model at startup to avoid repeated loading
    # IMPORTANT: This is cached in memory
    speech_model = load_speech_model()
```

**Developer Notes:**
- Speech model is loaded once at startup for performance
- Use `st.cache_resource` if implementing model caching
- Consider environment-based configuration for different setups

### 2. Image Processing Pipeline

```python
# image_processing.py
def process_image(uploaded_file):
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Preprocessing steps
    # 1. Resize for consistency
    # 2. Convert to grayscale
    # 3. Apply OCR
    return processed_image, extracted_text
```

**Performance Considerations:**
- Image preprocessing is CPU-intensive
- Consider implementing worker threads for large images
- Cache processed results using `st.cache_data`

### 3. Analysis Engine Implementation

#### Core Analysis Pipeline
```python
# analysis.py
def advanced_news_analysis(image, prompt):
    return {
        "text": extracted_text,
        "summary": generate_summary(text),
        "entities": extract_entities(text),
        "sentiment": analyze_sentiment(text),
        "context_embedding": generate_embeddings(text),
        "trend_analysis": analyze_trends(text),
        "related_news": fetch_related_news(text)
    }
```

**Key Points:**
- Each analysis component runs independently
- Implement error handling for each step
- Consider parallel processing for performance
- Cache expensive operations

### 4. RAG System Architecture

```python
# rag.py
def query_context(question, context_embedding):
    # Vector similarity search
    results = vector_search(question, context_embedding)
    return format_context_results(results)

def generate_rag_response(question, text, context_results):
    # Combine context with question for better responses
    prompt = construct_rag_prompt(question, context_results)
    return generate_response(prompt)
```

**Implementation Notes:**
- Use chunking for large texts
- Implement sliding window for context
- Consider token limits in prompt construction
- Cache embedding calculations

### 5. UI Component Implementation

#### Layout Management
```python
def display_analysis_results(analysis_result):
    # Use columns for responsive layout
    col1, col2 = st.columns(2)
    
    with col1:
        display_text_analysis(analysis_result)
    
    with col2:
        display_visualizations(analysis_result)
```

**Best Practices:**
- Use session state for persistent data
- Implement progressive loading
- Handle viewport resizing
- Consider mobile responsiveness

### 6. Speech Recognition Integration

```python
# speech_processing.py
def handle_user_questions(analysis_result, speech_model):
    if input_method == "Custom Speech Model":
        audio = record_audio()
        return speech_to_text(audio, 16000, speech_model)
    elif input_method == "Wav2Vec2 Model":
        return wav2vec2_speech_to_text(audio, 16000)
```

**Technical Considerations:**
- Handle different audio formats
- Implement timeout for recordings
- Consider memory usage for audio processing
- Cache model outputs when possible

## Error Handling Patterns

### 1. File Upload Validation
```python
def validate_upload(uploaded_file):
    try:
        if uploaded_file.type not in ["image/jpeg", "image/png"]:
            raise ValueError("Unsupported file type")
        if uploaded_file.size > MAX_FILE_SIZE:
            raise ValueError("File too large")
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return False
    return True
```

### 2. Analysis Error Management
```python
def safe_analysis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"error": str(e)}
    return wrapper
```