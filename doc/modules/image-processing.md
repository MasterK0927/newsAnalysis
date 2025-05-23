# Image Processing Module - Documentation

## Overview
The Image Processing module handles Optical Character Recognition (OCR) and image preprocessing for extracting text from newspaper cutouts and other text-containing images.

## Module: `image_processing.py`

### Dependencies
- **OpenCV (cv2)**: Image manipulation and preprocessing
- **NumPy**: Numerical operations on image arrays
- **Tesseract**: OCR engine for text extraction
- **PyTesseract**: Python wrapper for Tesseract

### Core Functions

#### `extract_text(image: np.ndarray) -> str`
Main function for extracting text from images using OCR.

**Process Flow:**
1. **Image Validation**: Check image format and size
2. **Preprocessing**: Apply image enhancement techniques
3. **OCR Processing**: Extract text using Tesseract
4. **Post-processing**: Clean and format extracted text

**Implementation Details:**
```python
def extract_text(image):
    """
    Extract text from image using advanced OCR processing.

    Preprocessing Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur for noise reduction
    3. Threshold adjustment for better contrast
    4. Morphological operations for text enhancement

    OCR Configuration:
    - PSM (Page Segmentation Mode): 6 (Single uniform block of text)
    - OEM (OCR Engine Mode): 3 (Default, based on what is available)
    - Language: eng (English, configurable)
    """
```

### Image Preprocessing Pipeline

#### 1. Noise Reduction
```python
def reduce_noise(image):
    """Apply noise reduction techniques."""
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Median filtering
    denoised = cv2.medianBlur(blurred, 3)

    return denoised
```

#### 2. Contrast Enhancement
```python
def enhance_contrast(image):
    """Improve image contrast for better OCR results."""
    # Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    return enhanced
```

#### 3. Text Region Detection
```python
def detect_text_regions(image):
    """Detect and isolate text regions in the image."""
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours
```

### OCR Configuration Options

#### Tesseract Parameters
```python
TESSERACT_CONFIG = {
    'psm': 6,  # Page segmentation mode
    'oem': 3,  # OCR engine mode
    'lang': 'eng',  # Language
    'config': '--tessdata-dir /usr/share/tesseract-ocr/4.00/tessdata'
}

# Custom configuration string
custom_config = f'-l {TESSERACT_CONFIG["lang"]} --oem {TESSERACT_CONFIG["oem"]} --psm {TESSERACT_CONFIG["psm"]}'
```

#### Multi-language Support
```python
LANGUAGE_CONFIGS = {
    'english': 'eng',
    'hindi': 'hin',
    'spanish': 'spa',
    'french': 'fra',
    'german': 'deu',
    'chinese_simplified': 'chi_sim',
    'arabic': 'ara'
}
```

### Error Handling

#### Common Issues and Solutions
```python
def handle_ocr_errors(image, error):
    """Handle OCR processing errors with fallback strategies."""

    if "tesseract not found" in str(error).lower():
        return "Tesseract OCR engine not installed. Please install Tesseract."

    elif "image format" in str(error).lower():
        # Try different preprocessing
        try:
            processed_image = apply_aggressive_preprocessing(image)
            return pytesseract.image_to_string(processed_image)
        except Exception:
            return "Unable to process image format."

    elif "empty image" in str(error).lower():
        return "Image appears to be empty or corrupted."

    else:
        return f"OCR processing failed: {str(error)}"
```

### Performance Optimization

#### 1. Image Resizing Strategy
```python
def optimize_image_size(image, target_dpi=300):
    """
    Optimize image size for OCR processing.

    Tesseract works best with images at 300 DPI.
    For newspaper text, resize to ensure optimal character height.
    """
    height, width = image.shape[:2]

    # Calculate optimal size
    if height < 600:  # Too small, upscale
        scale_factor = 600 / height
        new_width = int(width * scale_factor)
        new_height = 600
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    elif height > 2000:  # Too large, downscale
        scale_factor = 2000 / height
        new_width = int(width * scale_factor)
        new_height = 2000
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized = image

    return resized
```

#### 2. Region-Based Processing
```python
def process_text_regions(image):
    """
    Process image by text regions for better accuracy.

    Benefits:
    - Better handling of complex layouts
    - Improved accuracy for mixed text orientations
    - Faster processing of large images
    """
    regions = detect_text_regions(image)
    extracted_texts = []

    for region in regions:
        # Extract region from image
        x, y, w, h = cv2.boundingRect(region)
        region_image = image[y:y+h, x:x+w]

        # Process region
        text = pytesseract.image_to_string(region_image, config=custom_config)
        if text.strip():  # Only add non-empty text
            extracted_texts.append(text.strip())

    return ' '.join(extracted_texts)
```

### Quality Assessment

#### Text Quality Metrics
```python
def assess_text_quality(extracted_text):
    """
    Assess the quality of extracted text.

    Returns quality score (0-1) and recommendations.
    """
    quality_score = 0.0
    issues = []

    # Check for common OCR errors
    if not extracted_text.strip():
        issues.append("No text extracted")
        return 0.0, issues

    # Character diversity check
    unique_chars = len(set(extracted_text.lower()))
    if unique_chars < 5:
        issues.append("Low character diversity")
        quality_score -= 0.2

    # Word formation check
    words = extracted_text.split()
    valid_words = sum(1 for word in words if len(word) > 1 and word.isalpha())
    word_ratio = valid_words / len(words) if words else 0

    if word_ratio < 0.6:
        issues.append("Many invalid words detected")
        quality_score -= 0.3

    # Length check
    if len(extracted_text) < 50:
        issues.append("Very short text extracted")
        quality_score -= 0.2

    # Calculate final score
    quality_score = max(0.0, 1.0 + quality_score)

    return quality_score, issues
```

### Advanced Features

#### 1. Automatic Text Orientation Detection
```python
def detect_text_orientation(image):
    """
    Detect and correct text orientation.

    Uses Tesseract's orientation detection capability.
    """
    # Get orientation info
    osd = pytesseract.image_to_osd(image)

    # Parse orientation
    angle = 0
    for line in osd.split('\n'):
        if 'Rotate:' in line:
            angle = int(line.split(':')[1].strip())
            break

    # Rotate image if needed
    if angle != 0:
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1])
        return rotated

    return image
```

#### 2. Multi-Column Text Handling
```python
def handle_multi_column_text(image):
    """
    Handle newspapers with multiple columns.

    Processes columns separately and combines in reading order.
    """
    # Detect column boundaries
    horizontal_projection = np.sum(image, axis=0)

    # Find column separators (areas with low pixel density)
    threshold = np.mean(horizontal_projection) * 0.5
    separators = []

    for i, value in enumerate(horizontal_projection):
        if value < threshold:
            separators.append(i)

    # Group consecutive separators
    column_boundaries = []
    if separators:
        start = separators[0]
        for i in range(1, len(separators)):
            if separators[i] - separators[i-1] > 10:
                column_boundaries.append((start, separators[i-1]))
                start = separators[i]
        column_boundaries.append((start, separators[-1]))

    # Process each column
    column_texts = []
    prev_end = 0

    for start, end in column_boundaries:
        # Extract column
        column = image[:, prev_end:start]
        if column.shape[1] > 50:  # Minimum column width
            text = pytesseract.image_to_string(column, config=custom_config)
            column_texts.append(text)
        prev_end = end

    # Process final column
    final_column = image[:, prev_end:]
    if final_column.shape[1] > 50:
        text = pytesseract.image_to_string(final_column, config=custom_config)
        column_texts.append(text)

    return ' '.join(column_texts)
```

### Usage Examples

#### Basic Usage
```python
import cv2
from image_processing import extract_text

# Load image
image = cv2.imread('newspaper_cutout.jpg')

# Extract text
text = extract_text(image)
print(f"Extracted text: {text}")
```

#### Advanced Usage with Quality Assessment
```python
import cv2
from image_processing import extract_text, assess_text_quality

# Load and process image
image = cv2.imread('newspaper_cutout.jpg')
text = extract_text(image)

# Assess quality
quality_score, issues = assess_text_quality(text)

print(f"Quality Score: {quality_score:.2f}")
if issues:
    print("Issues found:")
    for issue in issues:
        print(f"- {issue}")
```

### Configuration

#### Environment Variables
```bash
# Tesseract configuration
TESSERACT_CMD=/usr/bin/tesseract
TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

# Image processing settings
MAX_IMAGE_SIZE=2000
MIN_IMAGE_SIZE=600
DEFAULT_DPI=300
```

#### Module Configuration
```python
IMAGE_CONFIG = {
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'supported_formats': ['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
    'default_language': 'eng',
    'quality_threshold': 0.6,
    'preprocessing_enabled': True,
    'multi_column_detection': True,
    'orientation_correction': True
}
```

### Troubleshooting

#### Common Issues

1. **"Tesseract not found" Error**
   - **Solution**: Install Tesseract OCR engine
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

2. **Poor OCR Accuracy**
   - **Solutions**:
     - Increase image resolution
     - Apply better preprocessing
     - Check text orientation
     - Ensure good contrast

3. **Memory Issues with Large Images**
   - **Solutions**:
     - Resize images before processing
     - Process regions separately
     - Optimize preprocessing pipeline

4. **Multi-language Text Issues**
   - **Solutions**:
     - Install additional language packs
     - Configure proper language codes
     - Use language-specific preprocessing

### Performance Metrics

#### Typical Performance
- **Small images** (< 1MB): 1-2 seconds
- **Medium images** (1-5MB): 3-8 seconds
- **Large images** (5-10MB): 8-15 seconds

#### Optimization Tips
- Resize images to optimal DPI (300)
- Use grayscale conversion
- Apply region-based processing for complex layouts
- Cache processed results for repeated operations

---

*This documentation covers the complete image processing module functionality and is updated regularly to reflect improvements and new features.*