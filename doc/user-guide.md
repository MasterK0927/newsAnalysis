# User Guide - News Analysis Application

## Table of Contents
- [Getting Started](#getting-started)
- [Interface Overview](#interface-overview)
- [Step-by-Step Usage](#step-by-step-usage)
- [Features Guide](#features-guide)
- [Tips and Best Practices](#tips-and-best-practices)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Getting Started

### Prerequisites
Before using the News Analysis application, ensure you have:
- ✅ Completed installation (see [Installation Guide](installation.md))
- ✅ Configured API keys (OpenAI, Pinecone)
- ✅ Test image ready (newspaper cutout or article screenshot)

### Launching the Application
1. Open terminal/command prompt
2. Navigate to project directory: `cd newsAnalysis`
3. Activate virtual environment: `source news_analysis_env/bin/activate`
4. Start application: `streamlit run main.py`
5. Open browser to `http://localhost:8501`

## Interface Overview

### Main Components

```
┌─────────────────────────────────────────────────────────┐
│                    📰 Advanced Context                   │
│                    Aware News Analysis                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Sidebar               │  Main Content Area             │
│  ┌─────────────────┐   │  ┌─────────────────────────┐    │
│  │ Upload Section  │   │  │                         │    │
│  │ - File Upload   │   │  │    Analysis Results     │    │
│  │ - Image Preview │   │  │                         │    │
│  │                 │   │  │  - Summary              │    │
│  │ How to Use      │   │  │  - Entities             │    │
│  │ - Instructions  │   │  │  - Sentiment            │    │
│  │ - Tips          │   │  │  - Word Cloud           │    │
│  └─────────────────┘   │  │  - Trend Analysis       │    │
│                         │  │  - Related News         │    │
│                         │  │  - Q&A Section         │    │
│                         │  └─────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Sidebar Elements

#### 1. Upload Newspaper Cutout
- **File uploader** for JPG, JPEG, PNG images
- **Image preview** shows uploaded file
- **Automatic processing** starts after upload

#### 2. How to Use Section
- Step-by-step instructions
- Usage tips and best practices
- Creator information

### Main Content Area

#### Analysis Results Display
- **News Summary**: AI-generated article summary
- **Named Entities**: People, organizations, locations mentioned
- **Sentiment Analysis**: Positive/negative sentiment with confidence
- **Word Cloud**: Visual representation of key terms
- **Trend Analysis**: Trending up/down topics
- **Related News**: Similar articles from web sources
- **Interactive Q&A**: Ask questions about the article

## Step-by-Step Usage

### Step 1: Upload News Image

1. **Click "Browse files"** in the sidebar
2. **Select image file** (JPG, PNG, max 200MB)
3. **Wait for upload** - image appears in sidebar
4. **Processing begins** automatically (spinner shows progress)

**Supported Image Types:**
- Newspaper cutouts
- Screenshot of online articles
- Magazine articles
- Printed news content
- Any text-containing image

**Image Quality Tips:**
- Use high-resolution images (300+ DPI)
- Ensure good lighting and contrast
- Avoid blurry or skewed images
- Clean background helps OCR accuracy

### Step 2: Review Analysis Results

The application automatically displays comprehensive analysis:

#### A. News Summary
- **Location**: Top of main content area
- **Content**: AI-generated summary of article
- **Purpose**: Quick overview of key points

#### B. Named Entities & Sentiment (Left Column)
- **Named Entities**:
  - People (PERSON): Names of individuals
  - Organizations (ORG): Companies, institutions
  - Locations (GPE): Cities, countries, places
  - Dates (DATE): Time references
- **Sentiment Analysis**:
  - Label: Positive, Negative, or Neutral
  - Confidence score: 0.0 to 1.0

#### C. Word Cloud (Right Column)
- **Visual representation** of most frequent terms
- **Size indicates frequency** - larger words appear more often
- **Interactive display** updates with content

### Step 3: Explore Trend Analysis

- **Trending Up**: Topics gaining attention
- **Trending Down**: Topics losing relevance
- **Based on**: Comparison with related articles
- **Helps understand**: Current relevance and context

### Step 4: Check Related News

- **Three-column layout** of related articles
- **Each card shows**:
  - Article title
  - "Open article →" link
  - Publication source
- **Purpose**: Provide broader context and verification sources

### Step 5: Ask Questions (Interactive Q&A)

#### Choose Input Method:
1. **Text Input**: Type questions directly
2. **Custom Speech Model**: Use built-in speech recognition
3. **Wav2Vec2 Model**: Alternative speech recognition

#### Text Input Process:
1. **Select "Text"** radio button
2. **Type question** in text box
3. **Press Enter** or click outside box
4. **Wait for response** (2-5 seconds)

#### Speech Input Process:
1. **Select speech model** (Custom or Wav2Vec2)
2. **Click "Start Recording"** button
3. **Speak clearly** into microphone (5-second recording)
4. **Wait for transcription** - shows recognized text
5. **Review transcribed text** before processing
6. **Response generated** automatically

### Step 6: Review Q&A Results

#### Answer Display:
- **Main answer**: AI-generated response
- **Context-aware**: Based on article content and related news
- **Factual basis**: Uses RAG (Retrieval-Augmented Generation)

#### Source Attribution:
- **Check "Show sources"** to see references
- **Source list** shows text snippets used for answer
- **Transparency** in information sources

## Features Guide

### 🖼️ Image Processing Features

#### OCR (Optical Character Recognition)
- **Automatic text extraction** from images
- **Multi-language support** (English primary)
- **Handles various layouts**: single/multi-column
- **Orientation correction**: Automatically rotates skewed text

#### Image Preprocessing
- **Noise reduction**: Removes image artifacts
- **Contrast enhancement**: Improves text clarity
- **Resolution optimization**: Adjusts for best OCR results

### 📝 Natural Language Processing

#### Text Analysis Pipeline
- **Text cleaning**: Removes OCR artifacts
- **Entity recognition**: Identifies key entities
- **Sentiment scoring**: Determines emotional tone
- **Summarization**: Creates concise overviews

#### Advanced NLP Features
- **Contextual understanding**: Maintains meaning across text
- **Multi-entity extraction**: Comprehensive entity identification
- **Confidence scoring**: Reliability indicators for results

### 🎤 Speech Recognition

#### Dual Model System
- **Custom Model**: Optimized for news-related vocabulary
- **Wav2Vec2**: Facebook's state-of-the-art model
- **Real-time processing**: Live transcription
- **Multi-language support**: English, Hindi, Tamil, others

#### Audio Processing
- **Noise reduction**: Filters background noise
- **Quality enhancement**: Improves audio clarity
- **Format handling**: Various audio input formats

### 🔍 Information Retrieval

#### Related News Discovery
- **Intelligent searching**: Finds contextually relevant articles
- **Multi-source aggregation**: Various news outlets
- **Real-time retrieval**: Current and recent articles
- **Quality filtering**: Relevant, high-quality sources only

#### Web Scraping
- **Respectful crawling**: Follows robots.txt and rate limits
- **Content extraction**: Clean article text
- **Source attribution**: Proper crediting of sources

### 💬 Interactive Q&A System

#### RAG (Retrieval-Augmented Generation)
- **Context retrieval**: Finds relevant information
- **Vector similarity**: Semantic matching
- **Response generation**: Contextual answers
- **Source tracking**: Maintains answer provenance

#### Question Types Supported
- **Factual questions**: "Who is mentioned?"
- **Analytical questions**: "What is the main impact?"
- **Comparative questions**: "How does this compare?"
- **Explanatory questions**: "Why did this happen?"

### 📊 Visualization Features

#### Word Clouds
- **Dynamic generation**: Updates with content
- **Customizable appearance**: Colors and layouts
- **Export capabilities**: Save visualizations
- **Interactive elements**: Responsive design

#### Trend Analysis
- **Comparative analysis**: Against related articles
- **Visual indicators**: Up/down trending topics
- **Temporal context**: Recent vs. historical trends

## Tips and Best Practices

### 🖼️ Image Upload Tips

#### For Best OCR Results:
- **High resolution**: 300 DPI or higher preferred
- **Good lighting**: Even, bright illumination
- **Straight orientation**: Avoid skewed or tilted text
- **Clean background**: Minimal visual noise
- **Sharp focus**: Avoid blurry or out-of-focus images

#### Image Formats:
- **Recommended**: PNG (lossless), high-quality JPEG
- **Avoid**: Heavily compressed JPEG, GIF
- **Size limit**: Up to 200MB supported

### 🎤 Speech Input Tips

#### For Clear Recognition:
- **Quiet environment**: Minimize background noise
- **Clear pronunciation**: Speak distinctly
- **Normal pace**: Not too fast or slow
- **Close to microphone**: 6-12 inches optimal distance
- **Natural speech**: Use normal conversational tone

#### Question Formulation:
- **Specific questions**: "Who is the CEO mentioned?" vs. "Who?"
- **Context clues**: Reference article content directly
- **Clear intent**: State what you want to know

### 💬 Q&A Best Practices

#### Effective Questions:
```
✅ Good Examples:
"What are the main economic impacts mentioned?"
"Who are the key people involved in this story?"
"When did these events take place?"
"How does this relate to previous similar incidents?"

❌ Avoid:
"What?" (too vague)
"Tell me everything" (too broad)
"Is this good or bad?" (subjective without context)
```

#### Using Context:
- **Reference specific content**: Mention names, dates, events
- **Build on previous answers**: Ask follow-up questions
- **Use article terminology**: Employ words from the original text

### 🔍 Understanding Results

#### Interpreting Sentiment:
- **Positive (0.7-1.0)**: Strong positive sentiment
- **Positive (0.5-0.7)**: Mild positive sentiment
- **Neutral (0.3-0.7)**: Balanced or objective tone
- **Negative (0.5-0.7)**: Mild negative sentiment
- **Negative (0.7-1.0)**: Strong negative sentiment

#### Entity Reliability:
- **PERSON**: Usually highly accurate
- **ORG**: Good accuracy for well-known organizations
- **GPE**: Excellent for major cities/countries
- **DATE**: Very reliable for standard date formats

#### Related News Relevance:
- **Check publication dates**: Recent articles more relevant
- **Verify sources**: Reputable news outlets preferred
- **Cross-reference**: Compare information across sources

## Troubleshooting

### Common Issues and Solutions

#### 1. Image Upload Problems

**Issue**: Image won't upload or shows error
**Solutions**:
- Check file size (under 200MB)
- Verify file format (JPG, PNG, JPEG)
- Try different browser
- Clear browser cache
- Restart application

**Issue**: Poor OCR results
**Solutions**:
- Use higher resolution image
- Ensure good lighting in photo
- Rotate image if text is sideways
- Try different image of same content
- Check for image blur or focus issues

#### 2. Speech Recognition Issues

**Issue**: Microphone not working
**Solutions**:
- Check browser permissions
- Allow microphone access
- Test microphone with other applications
- Restart browser
- Check system audio settings

**Issue**: Poor speech recognition
**Solutions**:
- Speak more clearly and slowly
- Reduce background noise
- Move closer to microphone
- Try different speech model (Custom vs Wav2Vec2)
- Use text input as alternative

#### 3. Analysis Problems

**Issue**: No analysis results shown
**Solutions**:
- Wait for processing to complete
- Check internet connection (for related news)
- Verify API keys are configured
- Restart application
- Try with different image

**Issue**: Related news not loading
**Solutions**:
- Check internet connection
- Verify no firewall blocking
- Try again later (rate limiting)
- Check article content has entities for search

#### 4. Q&A System Issues

**Issue**: No response to questions
**Solutions**:
- Ensure question is clear and specific
- Check if analysis completed successfully
- Verify OpenAI API key configured
- Try shorter, simpler questions
- Check internet connectivity

**Issue**: Poor answer quality
**Solutions**:
- Ask more specific questions
- Reference content from article directly
- Try different question phrasing
- Ensure article content is relevant to question

### Performance Issues

#### Slow Processing
- **Check system resources**: CPU, memory usage
- **Close other applications**: Free up system resources
- **Use smaller images**: Resize if very large
- **Check internet speed**: For API calls

#### Memory Errors
- **Restart application**: Clear memory cache
- **Use smaller images**: Reduce memory usage
- **Close browser tabs**: Free up RAM
- **Check system memory**: Ensure sufficient available

## FAQ

### General Questions

**Q: What file formats are supported?**
A: JPG, JPEG, PNG, and TIFF images are supported. PNG is recommended for best quality.

**Q: Is there a file size limit?**
A: Yes, maximum file size is 200MB per upload.

**Q: Does the app work offline?**
A: Partially. OCR and basic analysis work offline, but related news fetching and Q&A require internet connection.

**Q: Can I analyze multiple articles at once?**
A: Currently, the application processes one article at a time. Upload each article separately.

### Technical Questions

**Q: Which languages are supported?**
A: Text analysis primarily supports English. Speech recognition supports English, Hindi, Tamil, and several other languages through Wav2Vec2.

**Q: How accurate is the OCR?**
A: OCR accuracy depends on image quality. With high-quality images, accuracy is typically 90-95%.

**Q: What happens to my uploaded images?**
A: Images are processed locally and temporarily. They are not permanently stored or shared.

**Q: Can I save or export results?**
A: Currently, results are displayed in the interface. You can copy text or take screenshots. Export functionality may be added in future updates.

### Privacy & Security

**Q: Is my data secure?**
A: Yes. Processing happens locally, and only minimal data is sent to external APIs for specific features (related news, Q&A).

**Q: Are my questions logged?**
A: Questions are processed through OpenAI's API per their privacy policy. No local logging of questions occurs.

**Q: Can I use this for commercial purposes?**
A: Check the project license for commercial usage terms. Some integrated services have usage limitations.

### Feature Requests

**Q: Can you add support for [specific language]?**
A: Language support depends on available models. Check the GitHub repository for feature requests.

**Q: Will you add video analysis?**
A: Video analysis is being considered for future releases.

**Q: Can I contribute to the project?**
A: Yes! Check the GitHub repository for contribution guidelines.

---

*This user guide is regularly updated. For the latest information and support, visit the project repository or documentation site.*