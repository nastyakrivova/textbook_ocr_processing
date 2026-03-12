# Advanced Textbook OCR Pipeline

A sophisticated Optical Character Recognition (OCR) pipeline specifically engineered for complex textbook page layouts. This system excels at processing real-world photos, handling multi-column text, curved lines, and delivering structured JSON output for downstream applications.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-green)](https://github.com/PaddlePaddle/PaddleOCR)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-red)](https://opencv.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)](https://scikit-learn.org/)

## Key Features

*   **Superior Layout Analysis**: Automatically detects and processes multi-column layouts using K-Means clustering, ensuring text is extracted in the correct reading order.
*   **Robust Text Grouping**: Implements a custom nearest-neighbor algorithm to intelligently group words into lines, effectively handling curved or skewed text lines common in photographed book pages.
*   **Structured Output**: Produces clean, machine-readable JSON output, detailing each detected line with its bounding box, text content, and associated column ID, making it ideal for integration into larger data processing pipelines.
*   **Production-Ready Design**: Clean, modular code with comprehensive error handling, type hints, and a user-friendly command-line interface suitable for batch processing.
*   **Multilingual Support**: Optimized for Russian (default), easily configurable for other languages supported by PaddleOCR.

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **OCR Engine** | PaddleOCR 2.7.3+ | High-accuracy text detection and recognition |
| **Image Processing** | OpenCV | Image loading and basic preprocessing |
| **Layout Analysis** | Scikit-learn (KMeans) | Automatic column segmentation |
| **Language** | Python 3.7+ | Core implementation |

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step-by-Step Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/nastyakrivova/textbook_ocr_processing.git
    cd textbook_ocr_processing
Create and activate a virtual environment (recommended)

bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python3 -m venv venv
source venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
If requirements.txt is not present, create it with:

bash
pip install paddleocr opencv-python scikit-learn numpy
pip freeze > requirements.txt
🚀 Usage
Basic Command
Process an image with default settings (Russian language, output to output.json):

bash
python main.py path/to/your/image.jpg
Advanced Usage
bash
python main.py path/to/your/image.jpg --output custom_output.json --lang en
Command Line Arguments
Argument	Description	Default
image_path	Path to the input image file (required)	-
-o, --output	Path for the output JSON file	output.json
--lang	Language code for OCR (e.g., ru, en, fr)	ru
Example
bash
python main.py sample_page.jpg -o result.json --lang en

## How It Works: Technical Deep Dive
The pipeline implements a sophisticated multi-stage approach to extract text from complex textbook pages:

1. Text Detection & Recognition
PaddleOCR processes the input image, detecting individual text boxes and recognizing their content.

Returns bounding polygons, recognized text, and confidence scores for each word.

2. Column Segmentation
Extracts x-coordinates of word centers and applies K-Means clustering to automatically determine the number of columns.

Dynamic thresholding: If the spread of x-coordinates exceeds 20% of image width and sufficient words exist, the page is treated as multi-column.

Words are assigned to their respective columns for proper reading order.

3. Line Reconstruction (Custom Algorithm)
Word Pairing: For each word, calculates left and right center points of its bounding polygon.

Nearest-Neighbor Grouping: Iteratively groups words into lines by:

Finding the next word to the right with minimal vertical difference (≤ 20 pixels)

Ensuring horizontal gap doesn't exceed threshold (≤ 150 pixels)

Line Sorting: Resulting lines are sorted top-to-bottom within each column.

4. Structured Output Generation
Each reconstructed line becomes a block with:

Unique block ID

Bounding box coordinates [x1, y1, x2, y2]

Full line text

Column affiliation

Word count metadata

## Example JSON Output
json
{
  "blocks": [
    {
      "block_id": 1,
      "type": "line",
      "bbox": [45, 120, 350, 145],
      "text": "This is the first line of text in column one.",
      "column_id": 1,
      "word_count": 8
    },
    {
      "block_id": 2,
      "type": "line",
      "bbox": [400, 122, 720, 148],
      "text": "This is the corresponding line in column two.",
      "column_id": 2,
      "word_count": 7
    }
  ]
}
## Configuration Parameters
The OCR processor includes tunable parameters for optimal performance:

Parameter	Description	Default
max_vertical_diff	Maximum vertical difference (pixels) for words to be considered same line	20
max_horizontal_gap	Maximum horizontal gap (pixels) between words in same line	150
column_threshold	X-coordinate spread threshold for column detection (fraction of width)	0.2
min_words_for_multicolumn	Minimum words required to trigger multi-column detection	5


