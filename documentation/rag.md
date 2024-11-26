# RAG Implementation Documentation

## Overview
This code implements a Retrieval-Augmented Generation (RAG) system that combines document retrieval with language model generation. The system processes user queries by retrieving relevant context from a Pinecone vector database and generating contextualized responses using OpenAI's language models.

## Dependencies
```python
from transformers import GPT2Tokenizer
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from database import index
from news_scraper import gather_comprehensive_info
```

## Core Components

### 1. Text Summarization
```python
def summarize_text(text: str, max_tokens: int) -> str
```
**Purpose**: Truncates text to fit within a specified token limit while maintaining readability.

**Parameters**:
- `text` (str): Input text to be summarized
- `max_tokens` (int): Maximum number of tokens allowed

**Returns**:
- str: Truncated text with ellipsis if shortened

**Implementation Details**:
- Uses GPT-2 tokenizer to encode text
- Returns original text if within token limit
- Truncates to max_tokens and adds ellipsis if exceeded

### 2. Context Retrieval
```python
def query_context(user_query: str, context_embedding: np.ndarray) -> List[Dict]
```
**Purpose**: Retrieves relevant context from Pinecone vector database based on user query.

**Parameters**:
- `user_query` (str): User's input question
- `context_embedding` (np.ndarray): Embedded representation of context

**Returns**:
- List[Dict]: Top 3 most relevant matches with metadata

**Implementation Details**:
- Generates embedding for user query
- Queries Pinecone index for similar vectors
- Returns top 3 matches with metadata

### 3. RAG Response Generation
```python
def generate_rag_response(user_query: str, original_text: str, context_results: List[Dict]) -> str
```
**Purpose**: Generates a response using retrieved context and language model.

**Parameters**:
- `user_query` (str): User's input question
- `original_text` (str): Original document text
- `context_results` (List[Dict]): Retrieved context matches

**Returns**:
- str: Generated response incorporating context

**Constants**:
- `MAX_MODEL_TOKENS`: 4097 (Model context window size)
- `MIN_COMPLETION_TOKENS`: 100 (Reserved for response generation)

**Implementation Details**:
1. Token Management:
   - Summarizes original text (500 tokens max)
   - Summarizes each context result (200 tokens max)
   - Gathers additional information via `gather_comprehensive_info()`
   - Calculates available tokens for prompt

2. Context Processing:
   - Combines original text with additional context
   - Truncates combined context if exceeding token limit

3. Response Generation:
   - Uses PromptTemplate for structured query
   - Executes LLMChain with processed context
   - Returns generated response

## Prompt Template
```
Based on the following context, please answer the question concisely. If the context doesn't
contain enough information, use your general knowledge but indicate this in your response.

Context:
{context}

Question: {query}

Answer:
```

## Important Considerations

### Token Management
- Careful token budget allocation between:
  - User query
  - Context
  - Template
  - Response generation
- Dynamic context truncation to prevent token limit exceedance

### Context Integration
- Combines multiple sources:
  - Original document
  - Retrieved similar contexts
  - Additional scraped information
- Preserves most relevant information through strategic summarization

### Error Handling
- Handles oversized inputs through summarization
- Graceful truncation of context when necessary
- Maintains response coherence despite truncation

## Usage Example
```python
# Initialize components
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
llm = OpenAI(temperature=0.7)

# Process user query
query = "What are the main points discussed?"
context_results = query_context(query, context_embedding)
response = generate_rag_response(query, original_text, context_results)
```

## Performance Considerations
- Token processing overhead from multiple encoding/decoding operations
- Vector similarity search latency
- Additional information gathering latency
- Consider caching frequently accessed embeddings or contexts