# Children's Drawstrings Safety Policy Classifier

A machine learning classifier that detects violations of safety policies related to drawstrings in children's clothing using multi-modal analysis (text + images) with iterative hypothesis-based learning.

## Overview

This classifier detects products that violate the Children's Drawstrings safety policy by analyzing both product text descriptions and images using OpenAI's GPT-4o model. It has built-in mechanisms for iterative improvement through error analysis and hypothesis generation.

## Critical Policy Interpretation

The primary factor for determining if a product falls under the Children's Drawstrings policy is whether the product is **marketed towards or intended for children**, not just if it is explicitly labeled as "children's clothing."

Key considerations for identifying products marketed towards or intended for children:

1. **Size Information**: 
   - Size ranges that include children's sizes (2T-16 or equivalents)
   - Size indicators like "kids," "youth," "junior," or children's letter sizes (XS-L for children)

2. **Visual Styling and Design**:
   - Bright colors, playful patterns, cartoon characters
   - Youth-oriented themes or graphics
   - Simplified design elements typical in children's clothing
   - Character branding or licensed imagery popular with children

3. **Marketing Context**:
   - Models who appear to be children or teenagers
   - Family settings with children
   - School or playground settings
   - Presence of toys or children's props

4. **Terminology**:
   - Words like "kids," "youth," "junior," "child," "boys," "girls"
   - References to schools, play, growth
   - Sizing terminology common in children's clothing

Products are classified as violations if:
- They have drawstrings in the hood/neck area AND could be marketed to or intended for children sizes 2T-12, OR
- They have drawstrings in the waist/bottom area that extend more than 3 inches when the garment is expanded to its fullest width AND could be marketed to or intended for children sizes 2T-16

## Technical Implementation

### Core Functionality

1. **Multi-Modal Processing**: Analyzes both text descriptions and product images to make comprehensive assessments.
2. **Hypothesis-Based Learning**: Iteratively improves by analyzing error patterns and formulating hypothesis-driven improvements.
3. **Parallel Processing**: Efficiently handles large datasets through multi-threading.

### Key Components

- **Image Analysis**: Uses the OpenAI Vision API to detect drawstrings and assess if products appear to be marketed towards children.
- **Text Analysis**: Processes product titles, descriptions, and other metadata to identify safety concerns.
- **Hypothesis Generation**: Automatically analyzes misclassifications to create hypotheses about error patterns.
- **Iterative Refinement**: Applies learned insights to improve classification accuracy over time.

## How It Works

### Classification Process

1. The classifier analyzes product information from both text and images.
2. For text analysis, it reviews the product title, description, category, and keywords to detect mentions of drawstrings and indicators of being marketed to children.
3. For image analysis, it examines product photos to:
   - Identify visible drawstrings in hood/neck or waist areas
   - Detect visual indicators that the product is marketed towards or intended for children
   - Look for size information that falls within regulated ranges
4. The classifier combines these signals to make a final determination with detailed explanation.

### Iterative Improvement

The classifier can learn and improve through iterative cycles:

1. **Error Analysis**: The system identifies and collects misclassifications (false positives and false negatives).
2. **Pattern Recognition**: It analyzes error patterns to understand systematic classification issues.
3. **Hypothesis Generation**: Based on error patterns, it formulates a hypothesis about why these errors are occurring.
4. **Guideline Creation**: The hypothesis is converted into specific classification guidelines.
5. **Refinement**: The new guidelines are incorporated into the classifier for the next iteration.

## Usage

### Installation

```bash
# Install required packages
pip install openai python-dotenv pandas tabulate requests
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

### Quick Demo

Run a quick demonstration with example products:

```bash
python vision_hypothesis_classifier.py --mode demo
```

### Process a Dataset

Classify all products in a dataset:

```bash
python vision_hypothesis_classifier.py --mode batch --dataset your_dataset.json --workers 4 --output results.json
```

### Iterative Improvement

Run the classifier with iterative learning:

```bash
python vision_hypothesis_classifier.py --mode improve --dataset your_dataset.json --iterations 3 --workers 4 --output improvement_results.json
```

### Dataset Format

The classifier expects JSON input in the following format:

```json
[
  {
    "listing_id": "123456",
    "classification": "etsy.childrens_drawstrings", // Ground truth if available
    "title": "Kids Hoodie with Drawstring - Size 8",
    "description": "Warm hoodie with adjustable hood drawstring...",
    "category": "Children's Clothing > Outerwear",
    "keywords": ["kids hoodie", "boys jacket", "children's outerwear"],
    "images": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
  },
  // Additional products...
]
```

## Performance Monitoring

After each classification run, the system reports:
- Precision (how many identified violations were actual violations)
- Recall (how many actual violations were successfully identified)
- F1 score (harmonic mean of precision and recall)
- Accuracy (overall correct classifications)

After iterative improvement cycles, it provides detailed improvement metrics and the final set of classification guidelines that achieved the best results. 