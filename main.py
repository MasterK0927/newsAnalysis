import streamlit as st
import cv2
import numpy as np
import pytesseract
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, pipeline, GPT2Tokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
from pinecone import Pinecone
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from datetime import datetime
from typing import List, Dict
import re
from collections import Counter
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import uuid
import random
import time
import librosa
import sounddevice as sd
from scipy.io import wavfile
import torch.nn.functional as F
from ctcdecode import CTCBeamDecoder

load_dotenv()

# tokenizer
sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer_2 = GPT2Tokenizer.from_pretrained("gpt2")

# cuda for gpu acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing PineconeDB
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Initializing OpenAI API
llm = OpenAI(temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))

# Initializing BERT model and bert tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

# Loading spaCy model as nlp
nlp = spacy.load("en_core_web_sm")

# Sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", 
                             model="distilbert-base-uncased-finetuned-sst-2-english", 
                             tokenizer=sentiment_tokenizer,
                             max_length=512,
                             truncation=True,
                             device=device)

def format_google_news_url(url: str) -> str:
    """Clean and format Google News URLs for better display"""
    try:
        article_id = url.split('read/')[-1].split('?')[0]
        shortened_id = article_id[:8]
        return f"Article {shortened_id}..."
    except Exception:
        return "News article"

def extract_text(image):
    """Extract text from image using OCR"""
    return pytesseract.image_to_string(image)

class AdvancedSpeechModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdvancedSpeechModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=4, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

def load_speech_model(input_dim=80, hidden_dim=512, output_dim=29):
    model = AdvancedSpeechModel(input_dim, hidden_dim, output_dim)
    
    # Load pre-trained weights if available
    if os.path.exists("advanced_speech_model.pth"):
        model.load_state_dict(torch.load("advanced_speech_model.pth"))
    model.eval()
    return model

def record_audio(duration=5, sample_rate=16000):
    """Record audio from the microphone"""
    st.write("Recording... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def preprocess_audio(audio, sample_rate):
    """Preprocess the audio signal"""
    # Extract log mel spectrogram features
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80)
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Normalize
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
    return log_mel_spec

def decode_predictions(predictions, labels):
    decoder = CTCBeamDecoder(labels, beam_width=100, blank_id=len(labels) - 1)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(predictions)
    return ''.join([labels[p] for p in beam_results[0][0][:out_lens[0][0]]])

def speech_to_text(audio, sample_rate, model):
    """Convert speech to text using our advanced model"""
    # Preprocess audio
    log_mel_spec = preprocess_audio(audio, sample_rate)
    
    # Convert to torch tensor
    log_mel_spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0)  # Add batch dimension
    
    # Get model prediction
    with torch.no_grad():
        output = model(log_mel_spec_tensor)
        predictions = F.softmax(output, dim=-1)
    
    # Decode predictions to text
    labels = [chr(i + 96) for i in range(1, 27)] + ['<space>', '<blank>']
    decoded_text = decode_predictions(predictions, labels)
    
    return decoded_text

# Initialize Wav2Vec2 model for transfer learning
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def wav2vec2_speech_to_text(audio, sample_rate):
    """Convert speech to text using Wav2Vec2 model"""
    inputs = wav2vec2_processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = wav2vec2_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = wav2vec2_processor.batch_decode(predicted_ids)
    
    return transcription[0]

def preprocess_text(text):
    """Clean and preprocess text"""
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    return cleaned_text

def generate_embedding(text):
    """Generate BERT embedding for text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def create_context_embedding(text: str, related_news: List[Dict[str, str]]) -> np.ndarray:
    """Create a combined embedding from the main text and related news"""
    related_texts = [article['title'] + ' ' + article.get('description', '') for article in related_news]
    combined_text = text + ' ' + ' '.join(related_texts)
    return generate_embedding(combined_text)

def store_embedding(embedding, metadata):
    """Store embedding and metadata in Pinecone"""
    unique_id = str(uuid.uuid4())
    if isinstance(embedding, np.ndarray):
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        embedding_list = embedding.tolist()
    elif isinstance(embedding, list):
        if any(isinstance(i, list) for i in embedding):
            embedding_list = [item for sublist in embedding for item in sublist]
        else:
            embedding_list = embedding
    else:
        raise ValueError("Embedding must be a numpy array or a list")
    
    embedding_list = [float(val) for val in embedding_list]
    index.upsert(vectors=[(unique_id, embedding_list, metadata)])

def query_context(user_query: str, context_embedding: np.ndarray) -> List[Dict]:
    """Query Pinecone for relevant context using the user's question"""
    query_embedding = generate_embedding(user_query)
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=3,
        include_metadata=True
    )
    return results.matches

# text summarizer for reducing the context window length in rag_

def summarize_text(text: str, max_tokens: int) -> str:
    """Summarize the given text to fit within the specified token limit"""
    tokens = tokenizer_2.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer_2.decode(tokens[:max_tokens], skip_special_tokens=True) + "..."

# rag

def generate_rag_response(user_query: str, original_text: str, context_results: List[Dict]) -> str:
    """Generate a response using the original text and retrieved context, with improved token limit handling"""
    MAX_MODEL_TOKENS = 4097
    MIN_COMPLETION_TOKENS = 100

    original_summary = summarize_text(original_text, max_tokens=500)

    # Preparing the context
    context_texts = [summarize_text(result.metadata.get('text', ''), max_tokens=200) for result in context_results]
    comprehensive_info = gather_comprehensive_info(user_query)
    combined_context = f"{original_text}\n\nAdditional Context:\n{comprehensive_info}"

    # Calculating the available tokens for the prompt
    query_tokens = len(tokenizer_2.encode(user_query))
    # Approximating tokens for the template
    template_tokens = 100 
    available_prompt_tokens = MAX_MODEL_TOKENS-query_tokens-template_tokens-MIN_COMPLETION_TOKENS

    # Truncating the combined context if necessary to keep it within the context window
    context_tokens = tokenizer_2.encode(combined_context)
    if len(context_tokens) > available_prompt_tokens:
        truncated_context = tokenizer_2.decode(context_tokens[:available_prompt_tokens], skip_special_tokens=True)
    else:
        truncated_context = combined_context

    # Creating the prompt template and chain
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="""
        Based on the following context, please answer the question concisely. If the context doesn't 
        contain enough information, use your general knowledge but indicate this in your response.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(query=user_query, context=truncated_context)
    
    return response

# scrapping related news from google news

def fetch_related_news(query: str) -> List[Dict[str, str]]:
    """Fetch related news articles from Google News"""
    url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    max_retries = 3
    articles = []
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            selectors = ['div[jscontroller="d0DtYd"]', 'article', 'div.NiLAwe', 'h3.ipQwMb', 'div.xrnccd']
            
            for selector in selectors:
                items = soup.select(selector)
                if items:
                    for item in items[:5]:
                        title_elem = item.select_one('h3 a, h4 a, a')
                        if title_elem:
                            title = title_elem.text.strip()
                            link = title_elem.get('href', '')
                            
                            if link.startswith('./'):
                                link = 'https://news.google.com' + link[1:]
                            elif not link.startswith('http'):
                                link = 'https://news.google.com' + link
                            
                            snippet_elem = item.select_one('div[jsname="sngebd"], div.GI74Re')
                            snippet = snippet_elem.text.strip() if snippet_elem else ''
                            
                            articles.append({
                                'title': title,
                                'url': link,
                                'description': snippet
                            })
                    
                    if articles:
                        break
            
            if articles:
                break
            else:
                time.sleep(random.uniform(1, 3))
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch news: {str(e)}")
            time.sleep(random.uniform(1, 3))
    
    return articles

# scrapping the fecthed news articles for rag context (max 3)

def scrape_article_content(url: str) -> str:
    """Scrape the main content of an article"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # simple extraction and needs to be adjusted based on the structure of the websites
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        
        return content
    except Exception as e:
        print(f"Failed to scrape article content: {str(e)}")
        return ""

def gather_comprehensive_info(query: str) -> str:
    """Gather comprehensive information about a topic"""
    articles = fetch_related_news(query)
    selected_articles = articles[:3]

    comprehensive_info = f"Information about {query}:\n\n"
    
    for article in selected_articles:
        comprehensive_info += f"Title: {article['title']}\n"
        comprehensive_info += f"Description: {article['description']}\n"
        content = scrape_article_content(article['url'])
        # Truncating content to avoid token limits
        comprehensive_info += f"Content: {content[:500]}...\n\n"
    
    return comprehensive_info

# news summarizer

def summarize_news(text):
    """Generate a summary of the news article"""
    summary_prompt = f"Summarize the following news article in a concise paragraph:\n\n{text}"
    result = llm.generate([summary_prompt])
    return result.generations[0][0].text

def extract_entities(text):
    """Extract named entities from text"""
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}

def get_sentiment(text):
    """Perform sentiment analysis on text"""
    tokens = sentiment_tokenizer.encode(text, max_length=510, truncation=True)
    tokens = [sentiment_tokenizer.cls_token_id] + tokens + [sentiment_tokenizer.sep_token_id]
    
    if len(tokens) < 512:
        tokens = tokens + [sentiment_tokenizer.pad_token_id] * (512 - len(tokens))
    elif len(tokens) > 512:
        tokens = tokens[:512]
    
    truncated_text = sentiment_tokenizer.decode(tokens)
    result = sentiment_pipeline(truncated_text)[0]
    return result['label'], result['score']

def generate_word_cloud(text):
    """Generate a word cloud visualization"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def analyze_trends(main_article, related_articles):
    """Analyze trending words"""
    main_words = Counter(main_article.lower().split())
    related_words = Counter(" ".join(related_articles).lower().split())
    
    trending_up = [word for word in main_words if main_words[word] > related_words[word]][:5]
    trending_down = [word for word in main_words if main_words[word] < related_words[word]][:5]
    
    return trending_up, trending_down

# entry point

def advanced_news_analysis(image, user_query):
    """Perform comprehensive analysis on news image"""
    text = extract_text(image)
    preprocessed_text = preprocess_text(text)
    embedding = generate_embedding(preprocessed_text)
    store_embedding(embedding, {"text": preprocessed_text})
    entities = extract_entities(preprocessed_text)
    sentiment_label, sentiment_score = get_sentiment(preprocessed_text)
    summary = summarize_news(preprocessed_text)
    
    main_entity = list(entities.values())[0] if entities else ""
    related_news = fetch_related_news(main_entity)
    related_texts = [article['title'] for article in related_news]
    
    trend_analysis = analyze_trends(preprocessed_text, related_texts)
    
    # Creating the context embedding
    context_embedding = create_context_embedding(preprocessed_text, related_news)
    
    return {
        "text": preprocessed_text,
        "summary": summary,
        "entities": entities,
        "sentiment": (sentiment_label, sentiment_score),
        "related_news": related_news,
        "trend_analysis": trend_analysis,
        "context_embedding": context_embedding
    }

# Streamlit Interface starts from here

st.set_page_config(page_title="Advanced News Analysis", page_icon="📰", layout="wide")
st.title("📰 Advanced Context Aware News Analysis")

speech_model = load_speech_model()

# Sidebar
st.sidebar.header("Upload Newspaper Cutout")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

analysis_result = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Performing advanced analysis..."):
        analysis_result = advanced_news_analysis(image, "Analyze this news article")

    st.header("News Summary")
    st.write(analysis_result["summary"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Named Entities")
        st.write(analysis_result["entities"])
        st.subheader("Sentiment Analysis")
        st.write(f"Sentiment: {analysis_result['sentiment'][0]}")
        st.write(f"Confidence: {analysis_result['sentiment'][1]:.2f}")
    
    with col2:
        st.subheader("Word Cloud")
        word_cloud_plot = generate_word_cloud(analysis_result["text"])
        st.pyplot(word_cloud_plot)
    
    st.header("Trend Analysis")
    trending_up, trending_down = analysis_result["trend_analysis"]
    st.write("Trending Up:", ", ".join(trending_up))
    st.write("Trending Down:", ", ".join(trending_down))
    
    # Related News Articles Section
    st.header("Related News Articles")
    if analysis_result["related_news"]:
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        
        for i, article in enumerate(analysis_result["related_news"]):
            if article['url']:
                with columns[i % 3]:
                    try:
                        article_title = format_google_news_url(article['url'])
                        
                        with st.container():
                            st.markdown(
                                f"""
                                <div style="border:1px solid #ccc; border-radius:5px; padding:10px; margin-bottom:10px;">
                                    <h4>{article_title}</h4>
                                    <a href="{article['url']}" target="_blank">Open article →</a>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        st.error(f"Error displaying article: {str(e)}")
    
    st.header("Ask Questions About the Article")
    input_method = st.radio("Choose input method:", ("Text", "Custom Speech Model", "Wav2Vec2 Model"))
    user_question = st.text_input("What would you like to know about this article?")
    
    if input_method == "Text":
        user_question = st.text_input("What would you like to know about this article?")
    else:
        if st.button("Start Recording"):
            with st.spinner("Listening..."):
                audio = record_audio()
                if input_method == "Custom Speech Model":
                    user_question = speech_to_text(audio, 16000, speech_model)
                else:  # Wav2Vec2 Model
                    user_question = wav2vec2_speech_to_text(audio, 16000)
            st.write(f"Recognized Text: {user_question}")

    if user_question:
        with st.spinner("Generating context-aware response..."):
            context_results = query_context(user_question, analysis_result["context_embedding"])
            rag_response = generate_rag_response(
                user_question, 
                analysis_result["text"], 
                context_results
            )
        
        st.subheader("Answer")
        st.write(rag_response)
        
        if st.checkbox("Show sources used for the answer"):
            st.subheader("Sources Referenced")
            for idx, result in enumerate(context_results, 1):
                st.markdown(f"**Source {idx}:**")
                st.write(result.metadata.get('text', '')[:200] + '...')

else:
    st.info("Please upload a newspaper cutout to get started.")

st.sidebar.markdown("""
## How to use:
1. Upload a newspaper cutout image
2. View the comprehensive analysis including summary, entities, sentiment, and more
3. Explore related news and comparisons
4. Ask questions about the article using text or speech input (custom model or Wav2Vec2)
5. Check the sources used for answers if needed
""")

st.sidebar.markdown("Created by Keshav")