# News Scraping and Content Gathering System
## Technical Documentation

### Overview
This system provides functionality for fetching and analyzing news articles from Google News and their source websites. It implements robust scraping with retry mechanisms, rate limiting, and error handling.

## Core Components

### 1. News Article Fetcher (`fetch_related_news`)

#### Purpose
Retrieves related news articles from Google News based on a search query.

#### Function Signature
```python
def fetch_related_news(query: str) -> List[Dict[str, str]]
```

#### Key Features
- Multiple CSS selector fallbacks for resilient scraping
- Retry mechanism with exponential backoff
- Rate limiting to prevent IP blocks
- Flexible parsing of different article formats

#### Implementation Details
- **Max Retries**: 3 attempts
- **Timeout**: 10 seconds per request
- **Rate Limiting**: Random delay between 1-3 seconds between retries
- **User-Agent Spoofing**: Uses Chrome browser agent string

#### Supported Article Elements
```python
selectors = [
    'div[jscontroller="d0DtYd"]',
    'article',
    'div.NiLAwe',
    'h3.ipQwMb',
    'div.xrnccd'
]
```

#### Return Format
```python
{
    'title': str,     # Article title
    'url': str,       # Full article URL
    'description': str # Article snippet/description
}
```

#### Error Handling
- Network timeout handling
- HTTP error status handling
- Invalid HTML structure handling
- URL normalization for relative paths

### 2. Article Content Scraper (`scrape_article_content`)

#### Purpose
Extracts the main content from news article URLs.

#### Function Signature
```python
def scrape_article_content(url: str) -> str
```

#### Features
- Paragraph-based content extraction
- Clean text formatting
- Timeout protection
- Error resilience

#### Implementation Details
- **Timeout**: 10 seconds per request
- **Content Selection**: Targets `<p>` tags
- **Text Cleaning**: Strips whitespace and joins paragraphs

#### Error Handling
- Network errors
- Invalid HTML structure
- Timeout handling
- Returns empty string on failure

### 3. Information Aggregator (`gather_comprehensive_info`)

#### Purpose
Combines news fetching and content scraping to provide comprehensive information about a topic.

#### Function Signature
```python
def gather_comprehensive_info(query: str) -> str
```

#### Features
- Multi-article aggregation
- Content summarization
- Structured output format

#### Implementation Details
- Processes top 3 articles
- Limits content preview to 500 characters
- Structured output with titles and descriptions

## Usage Examples

### Basic News Fetching
```python
articles = fetch_related_news("artificial intelligence")
for article in articles:
    print(f"Title: {article['title']}")
    print(f"URL: {article['url']}")
```

### Content Scraping
```python
content = scrape_article_content("https://example.com/article")
print(f"Article content: {content[:200]}...")
```

### Comprehensive Information Gathering
```python
info = gather_comprehensive_info("climate change")
print(info)
```

## Best Practices

### Rate Limiting
- Implement random delays between requests
- Use exponential backoff for retries
- Respect robots.txt guidelines

### Error Handling
1. Network Errors:
   - Implement timeouts
   - Use retry mechanism
   - Log failed attempts

2. Parsing Errors:
   - Use multiple CSS selectors
   - Implement fallback parsing
   - Validate extracted content

3. Content Validation:
   - Check for empty responses
   - Verify URL formats
   - Validate content length

### Performance Optimization
1. Request Management:
   - Limit concurrent requests
   - Implement caching when appropriate
   - Use connection pooling

2. Content Processing:
   - Limit content length
   - Implement efficient text processing
   - Use appropriate data structures

## Security Considerations

### Request Headers
```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                  (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
```

### Best Practices
- Rotate User-Agent strings
- Implement IP rotation if needed
- Respect robots.txt
- Handle sensitive data appropriately
- Validate and sanitize URLs

## Common Issues and Solutions

### 1. Rate Limiting Detection
**Problem**: Google News blocking requests
**Solution**: 
- Implement random delays
- Use rotating User-Agents
- Implement proxy rotation if needed

### 2. Content Extraction Failures
**Problem**: Unable to extract article content
**Solution**:
- Implement multiple parsing strategies
- Use fallback selectors
- Validate HTML structure

### 3. URL Handling
**Problem**: Invalid or relative URLs
**Solution**:
- Implement URL normalization
- Validate URL format
- Handle relative paths properly

## Maintenance and Updates

### Regular Tasks
1. Update User-Agent strings
2. Review and update CSS selectors
3. Monitor success rates
4. Update error handling strategies

### Code Updates
1. Keep dependencies updated
2. Monitor Google News HTML structure changes
3. Update parsing strategies as needed
4. Implement new features based on requirements

## Dependencies
- requests: HTTP requests handling
- beautifulsoup4: HTML parsing
- random: Randomization for delays
- time: Time-based operations
