import json
import os
import time
import base64
import requests
from io import BytesIO
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate
import re
import concurrent.futures
import argparse

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def load_dataset(file_path):
    """Load the dataset from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def download_image(url):
    """Download an image from a URL and convert to base64."""
    try:
        response = requests.get(url, timeout=10)
        return url  # Return the URL to directly use it with the API
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def extract_urls(text):
    """Extract URLs from text using regex."""
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+|http?://[^\s<>"]+'
    matches = re.findall(url_pattern, text)
    return matches

def extract_classification(text):
    """Extract classification result from model response."""
    # Check for explicit classification first
    if re.search(r'\b(etsy\.childrens_drawstrings)\b', text.lower()):
        return "etsy.childrens_drawstrings"
    if re.search(r'\b(out_of_scope)\b', text.lower()):
        return "out_of_scope"
    
    # Check for decision tree criteria
    children_size_match = re.search(r'(children|child|kids|youth).{0,30}(size|age).{0,30}(up to 14|14 or under|2t-14)', text.lower())
    drawstring_match = re.search(r'(drawstring|cord|string).{0,50}(present|has|contains|visible)', text.lower())
    outerwear_match = re.search(r'(outerwear|hoodie|sweatshirt|jacket|coat)', text.lower())
    
    # Look for mentions of all three criteria being met
    all_criteria_met = re.search(r'(all three criteria|all criteria|all conditions).{0,30}(met|satisfied)', text.lower())
    
    # Look for any explicit statement that one criterion is not met
    not_children_match = re.search(r'(not|isn\'t|doesn\'t).{0,30}(child|kid|youth|size 14)', text.lower())
    no_drawstring_match = re.search(r'(no|without|lacks|absence of|doesn\'t have).{0,30}(drawstring|cord|string)', text.lower())
    not_outerwear_match = re.search(r'(not|isn\'t).{0,30}(outerwear|upper body)', text.lower())
    
    # If any criterion is clearly not met, it's out_of_scope
    if not_children_match or no_drawstring_match or not_outerwear_match:
        return "out_of_scope"
    
    # If all criteria are explicitly mentioned as met
    if children_size_match and drawstring_match and outerwear_match and all_criteria_met:
        return "etsy.childrens_drawstrings"
    
    # Default to out_of_scope when in doubt
    return "out_of_scope"

def improved_create_prompt(review_input, prompt_refinements=None):
    """
    Create a system prompt and user prompt for classification.
    Returns a tuple of (system_prompt, user_prompt).
    
    Args:
        review_input: Dictionary containing product information
        prompt_refinements: Optional dictionary of additional guidelines
    """
    # Base system prompt
    system_prompt = """You are an expert product safety analyst for Etsy, specializing in the Children's Drawstrings policy.

POLICY OVERVIEW - FOLLOW THIS DECISION TREE EXACTLY:
1. Is the item available for children size/age up to and including 14?
   - Look for explicit size indicators (2T-14, children's sizing) or marketing targeting children
   - If NO → ALLOW (out_of_scope)
   
2. Does the item contain drawstrings?
   - Look for cords, strings, or drawstrings specifically in the hood, neck, or waist area
   - If NO → ALLOW (out_of_scope)
   
3. Is the item considered upper body outerwear?
   - Examples: hoodies, sweatshirts, jackets, coats, etc.
   - NOT included: capes, ponchos, or non-outerwear items
   - If NO → ALLOW (out_of_scope)
   
4. If YES to ALL of the above → PROHIBIT (etsy.childrens_drawstrings)

CRITICAL DEFINITIONS:
- CHILDREN'S SIZING: Generally includes sizes 2T through 14 (also XS-L for children)
- DRAWSTRINGS: Functional cords or strings typically used for tightening clothing, especially around hood/neck or waist
- UPPER BODY OUTERWEAR: Garments worn over other clothing for warmth/protection (hoodies, jackets, coats, sweatshirts)

IMPORTANT CLASSIFICATION NOTES:
- Adult-sized clothing with drawstrings is ALLOWED (out_of_scope)
- Children's clothing WITHOUT drawstrings is ALLOWED (out_of_scope)
- Non-outerwear children's items with drawstrings are ALLOWED (out_of_scope)
- Drawstring items for children OVER size 14 are ALLOWED (out_of_scope)

INDICATORS TO LOOK FOR:
1. Children's Marketing Terms:
   - "Kids", "Children's", "Child", "Toddler", "Youth", "Boy's", "Girl's", "Junior"
   - Size indicators like 2T, 3T, 4T, 5, 6, 7, 8, 9, 10, 11, 12, 14
   
2. Drawstring Indicators:
   - Explicit mentions of "drawstring", "drawcord", "pull cord", "string"
   - Images showing cords coming out of hood, neck or waist areas
   
3. Outerwear Indicators:
   - Categories like "hoodies", "sweatshirts", "jackets", "coats"
   - Terms describing outer garments worn for warmth or protection

BE PRECISE IN YOUR ANALYSIS:
- Don't over-classify adult items as children's items
- Don't classify items without drawstrings as violations
- Don't classify non-outerwear as outerwear
- A product must meet ALL THREE criteria to be classified as a violation"""

    # Add any additional guidelines from prompt refinements
    if prompt_refinements:
        if isinstance(prompt_refinements, dict):
            for key, value in prompt_refinements.items():
                system_prompt += f"\n\nADDITIONAL GUIDELINE - {key.upper()}:\n{value}"
        elif isinstance(prompt_refinements, str):
            system_prompt += f"\n\nADDITIONAL GUIDELINES:\n{prompt_refinements}"

    # Extract relevant fields from review_input
    title = review_input.get("title", "No title provided")
    description = review_input.get("description", "No description provided")
    category = review_input.get("category", "No category provided")
    keywords = review_input.get("keywords", "No keywords provided")
    
    # Create user prompt
    user_prompt = f"""Please analyze this product information to determine if it violates the Children's Drawstrings policy:

TITLE: {title}

CATEGORY: {category}

DESCRIPTION: {description}

KEYWORDS: {keywords}

Remember to follow the decision tree precisely:
1. Is it for children size/age up to 14?
2. Does it contain drawstrings?
3. Is it upper body outerwear?

Only if ALL three conditions are met should this be classified as a violation.

Based on this text information ONLY, provide your initial analysis of whether this product might violate the Children's Drawstrings policy."""

    return system_prompt, user_prompt

def analyze_images(image_urls):
    """Analyze images using OpenAI's Vision model to detect drawstrings."""
    if not image_urls or image_urls == ["No images provided"]:
        return "No images available for analysis."
    
    valid_images = []
    for url in image_urls[:3]:  # Limit to first 3 images
        image_data = download_image(url)
        if image_data:
            valid_images.append({"type": "image_url", "image_url": {"url": url}})
    
    if not valid_images:
        return "Could not download any images for analysis."
    
    try:
        # Detailed prompt for vision model
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert safety analyst examining product images for potential children's drawstring violations. Follow this EXACT decision tree:

1. IS THE ITEM FOR CHILDREN UP TO SIZE 14?
   - Look for: Size tags (2T-14), proportions typical of children's clothing, child models
   - Evidence for YES: Visible children's size tags, child models, small proportions
   - Evidence for NO: Adult size tags, adult models, large proportions

2. DOES THE ITEM CONTAIN DRAWSTRINGS?
   - Look for: Visible cords/strings particularly at hood, neck, or waist
   - Evidence for YES: Visible drawstrings, cords, or strings that can be pulled to tighten
   - Evidence for NO: No visible cords or strings, decorative non-functional elements

3. IS THE ITEM UPPER BODY OUTERWEAR?
   - Look for: Hoodies, jackets, coats, sweatshirts worn over other clothing
   - Evidence for YES: Hood present, outerwear styling, heavy/insulated material
   - Evidence for NO: Lightweight material, indoor-only styling, not outerwear

PROVIDE CLEAR EVIDENCE FOR EACH DECISION POINT based ONLY on what you can see in the images.
A product must meet ALL THREE criteria to be a potential violation."""
                },
                {
                    "role": "user",
                    "content": [
                        """Analyze these product images according to the Children's Drawstrings safety decision tree. For each decision point, provide specific visual evidence:

1. IS THE ITEM FOR CHILDREN UP TO SIZE 14?
   - Describe any visible size indicators, proportions, or models that suggest target age
   
2. DOES THE ITEM CONTAIN DRAWSTRINGS?
   - Identify any visible cords, strings or drawstrings in hood, neck, or waist areas

3. IS THE ITEM UPPER BODY OUTERWEAR?
   - Determine if this is a hoodie, jacket, coat, sweatshirt or similar outerwear item

CONCLUSION: State whether the product meets ALL THREE criteria for a potential violation based ONLY on visual evidence.""",
                        *valid_images
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in image analysis API call: {e}")
        return "Error analyzing images."

def classify_product_with_vision(review_input, prompt_refinements=None, temperature=0):
    """
    Classify a product using both text and image analysis.
    Returns the classification result and supporting explanation.
    """
    # Generate text prompt
    system_prompt, user_prompt = improved_create_prompt(review_input, prompt_refinements)
    
    # Extract image URLs
    image_urls = []
    if "images" in review_input:
        # If images is a field with actual URLs
        if isinstance(review_input["images"], list):
            image_urls = review_input["images"]
        # If images is a string that might contain URLs
        elif isinstance(review_input["images"], str):
            urls = extract_urls(review_input["images"])
            if urls:
                image_urls = urls
    
    if not image_urls:
        image_urls = ["No images provided"]
    
    # Run image analysis
    image_analysis = analyze_images(image_urls)
    
    # Create a conversation that includes both text and image analysis
    try:
        # Structure the conversation to give both text information and image analysis
        messages = [
            {"role": "system", "content": system_prompt + """
            
FINAL DECISION INSTRUCTIONS:
Follow the Children's Drawstrings decision tree EXACTLY:

1. Is the item available for children size/age up to and including 14?
   - If NO → ALLOW (out_of_scope)
   
2. Does the item contain drawstrings?
   - If NO → ALLOW (out_of_scope)
   
3. Is the item considered upper body outerwear?
   - If NO → ALLOW (out_of_scope)
   
4. If YES to ALL three criteria above → PROHIBIT (etsy.childrens_drawstrings)

ALL THREE conditions must be met for a violation. If ANY condition is not met, the product is out_of_scope.
Be conservative in your assessment - only clear violations should be classified as etsy.childrens_drawstrings."""},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "I'll analyze this product based on the text description provided."},
            {"role": "user", "content": "Here's an analysis of the product images:\n\n" + image_analysis},
            {"role": "user", "content": """Based on BOTH the text description and image analysis, please provide your final classification by answering each question in the decision tree:

1. Is the item available for children size/age up to and including 14?
   [Your assessment with specific evidence]

2. Does the item contain drawstrings?
   [Your assessment with specific evidence]

3. Is the item considered upper body outerwear?
   [Your assessment with specific evidence]

FINAL CLASSIFICATION: Based on the decision tree, is this "etsy.childrens_drawstrings" or "out_of_scope"?
Remember, ALL THREE criteria must be met for a violation. If ANY condition is not met, classify as "out_of_scope".

Provide your detailed reasoning for this classification."""}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature
        )
        
        response_text = response.choices[0].message.content
        
        # Extract classification
        classification_result = extract_classification(response_text)
        
        return {
            "classification": classification_result,
            "explanation": response_text
        }
    except Exception as e:
        print(f"Error in classification API call: {e}")
        return {
            "classification": "Error during classification",
            "explanation": f"Error: {str(e)}"
        }

def process_listing(item, prompt_refinements=None):
    """Process a single listing and return its classification result."""
    try:
        start_time = time.time()
        
        # Check if this is an item with the reviewInput/expectedOutcome structure
        if "reviewInput" in item and "expectedOutcome" in item:
            # Extract the relevant fields
            review_input = item["reviewInput"]
            true_label = item["expectedOutcome"]
            listing_id = review_input.get("id", "unknown")
            
            # Run classification on the reviewInput
            result = classify_product_with_vision(review_input, prompt_refinements)
        else:
            # Assume item is already in the expected format
            listing_id = item.get("listing_id", "unknown")
            true_label = item.get("classification", "unknown")
            result = classify_product_with_vision(item, prompt_refinements)
            
        end_time = time.time()
        
        return {
            "listing_id": listing_id,
            "true_label": true_label,
            "predicted_label": result["classification"],
            "explanation": result["explanation"],
            "processing_time": end_time - start_time
        }
    except Exception as e:
        print(f"Error processing listing: {e}")
        # If we have the reviewInput/expectedOutcome structure
        if "reviewInput" in item and "expectedOutcome" in item:
            listing_id = item["reviewInput"].get("id", "unknown")
            true_label = item["expectedOutcome"]
        else:
            listing_id = item.get("listing_id", "unknown")
            true_label = item.get("classification", "unknown")
            
        return {
            "listing_id": listing_id,
            "true_label": true_label,
            "predicted_label": "Error during processing",
            "explanation": str(e),
            "processing_time": 0
        }

def classify_batch(dataset, max_workers=5, max_items=None, iteration=0, additional_guidelines=""):
    """Classify a batch of products in parallel."""
    items = dataset["data"]
    
    if max_items:
        items = items[:max_items]
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_listing, item, iteration, additional_guidelines): item for item in items}
        
        for future in future_to_item:
            try:
                result = future.result()
                results.append(result)
                print(f"Processed item {result['listing_id']}: predicted={result['predicted_label']}, expected={result['true_label']}")
            except Exception as e:
                print(f"Error processing item: {e}")
    
    return results

def get_error_examples(dataset, error_ids):
    """Get the full examples for a list of error IDs."""
    examples = []
    
    for item in dataset["data"]:
        item_id = item["reviewInput"].get("id", "unknown")
        if item_id in error_ids:
            examples.append(item)
    
    return examples

def create_hypothesis(error_set):
    """
    Analyze error patterns and generate a hypothesis to improve classification.
    Takes a list of error cases and returns a hypothesis for improvement.
    """
    if not error_set:
        return "No errors to analyze."
    
    # Prepare error cases to analyze
    error_samples = []
    for error in error_set[:10]:  # Limit to 10 examples to avoid token limits
        error_samples.append({
            "listing_id": error.get("listing_id", "unknown"),
            "true_label": error.get("true_label", "unknown"),
            "predicted_label": error.get("predicted_label", "unknown"),
            "explanation": error.get("explanation", "No explanation available")
        })
    
    # Format the error samples as text
    formatted_errors = "\n\n".join([
        f"ERROR CASE #{i+1}\n" +
        f"True Classification: {err['true_label']}\n" +
        f"Predicted Classification: {err['predicted_label']}\n" +
        f"Explanation: {err['explanation'][:1000]}..." if len(err['explanation']) > 1000 else f"Explanation: {err['explanation']}"
        for i, err in enumerate(error_samples)
    ])
    
    try:
        # Create system prompt for error analysis
        system_prompt = """You are an expert product safety analyst specializing in the Children's Drawstrings safety policy.

POLICY DETAILS:
Children's upper outerwear with drawstrings at the hood or neck area in sizes 2T to 12 violates federal safety regulations.
Children's upper outerwear with drawstrings at the waist or bottom in sizes 2T to 16 that extend more than 3 inches outside the garment's sides when expanded to full width violates federal regulations.

CRITICAL POLICY INTERPRETATION:
The key factor is whether the product is marketed towards or intended for children, not just if it is explicitly labeled as children's clothing.
Products do not need to be explicitly labeled "for children" to fall under this policy.
Any product that could reasonably be considered as marketed towards or intended for children should be evaluated under this policy.

INDICATORS OF CHILD MARKETING OR INTENT:
1. Size information suggesting children's sizing (2T-16, XS-L for children)
2. Visual styling elements suggesting child appeal:
   - Bright colors, playful patterns, cartoon characters
   - Youth-oriented themes or graphics
   - Simplified design elements typical in children's clothing
3. Marketing context:
   - Use of models who appear to be children
   - Settings suggesting children's use (school, playground)
   - Family imagery suggesting child and parent
4. Terminology suggesting child targeting:
   - Words like "kids," "youth," "junior," "child," "boys," "girls"
   - References to schools, play, growth
   - Sizing terminology common for children's clothing

YOUR TASK:
1. Analyze the error patterns in the classification results provided.
2. Identify common themes or issues that led to misclassifications.
3. Form a hypothesis about why the model might be making these errors.
4. Suggest specific improvements or guidelines that could address these issues.
5. Focus on better recognizing:
   - Implicit indicators of children's marketing
   - Subtle mentions of drawstrings in product descriptions
   - Size information that suggests children's sizing
   - Visual cues in images that indicate child-oriented products

Your response should be structured as follows:
1. ERROR PATTERNS: Describe the common patterns you observe in the misclassifications.
2. HYPOTHESIS: Provide a concise hypothesis explaining why these errors are occurring.
3. IMPROVEMENT GUIDELINES: Suggest 3-5 specific, actionable guidelines that could improve classification accuracy."""

        # Create user prompt for error analysis
        user_prompt = f"""Below are examples of misclassifications from our Children's Drawstrings safety classifier.
Please analyze these errors and provide a hypothesis for improvement.

{formatted_errors}

Based on these error cases, provide:
1. A detailed analysis of error patterns
2. A hypothesis explaining the likely cause of misclassifications
3. 3-5 specific guidelines to improve classification accuracy"""

        # Get response from OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0  # Use deterministic output
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error creating hypothesis: {e}")
        return f"Error creating hypothesis: {str(e)}"

def calculate_metrics(results):
    """Calculate precision, recall, F1, and other metrics from classification results."""
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    for result in results:
        if result["predicted_label"] == "etsy.childrens_drawstrings" and result["true_label"] == "etsy.childrens_drawstrings":
            true_positive += 1
        elif result["predicted_label"] == "etsy.childrens_drawstrings" and result["true_label"] != "etsy.childrens_drawstrings":
            false_positive += 1
        elif result["predicted_label"] != "etsy.childrens_drawstrings" and result["true_label"] == "etsy.childrens_drawstrings":
            false_negative += 1
        else:
            true_negative += 1
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / len(results) if len(results) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative
    }

def analyze_errors(results):
    """Analyze the error cases to understand misclassifications."""
    false_positives = [r for r in results if r["predicted_label"] == "etsy.childrens_drawstrings" and r["true_label"] != "etsy.childrens_drawstrings"]
    false_negatives = [r for r in results if r["predicted_label"] != "etsy.childrens_drawstrings" and r["true_label"] == "etsy.childrens_drawstrings"]
    
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def print_metrics_table(metrics_by_iteration):
    """Print a formatted table of metrics for each iteration."""
    data = []
    for i, metrics in enumerate(metrics_by_iteration):
        data.append([
            i,
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['accuracy']:.4f}",
            metrics['true_positive'],
            metrics['false_positive'],
            metrics['false_negative'],
            metrics['true_negative']
        ])
    
    headers = ["Iteration", "Precision", "Recall", "F1 Score", "Accuracy", "TP", "FP", "FN", "TN"]
    print("\n" + tabulate(data, headers=headers, tablefmt="grid"))

def print_error_ids(false_positives, false_negatives):
    """Print formatted lists of error IDs."""
    print(f"\n{'-'*50}\nFALSE POSITIVES ({len(false_positives)}):")
    if false_positives:
        for i, fp in enumerate(false_positives):
            print(f"  {i+1}. ID: {fp['id']}")
    else:
        print("  None")
    
    print(f"\nFALSE NEGATIVES ({len(false_negatives)}):")
    if false_negatives:
        for i, fn in enumerate(false_negatives):
            print(f"  {i+1}. ID: {fn['id']}")
    else:
        print("  None")
    print(f"{'-'*50}")

def vision_hypothesis_improvement(dataset, iterations=3, batch_size=20, max_workers=5):
    """Run multiple iterations of classification with hypothesis-based improvements and vision model."""
    results_by_iteration = []
    metrics_by_iteration = []
    hypotheses = []
    guidelines_by_iteration = [""] * (iterations + 1)  # +1 for the baseline
    
    # Initial run (baseline)
    print(f"\n{'='*80}")
    print(f"ITERATION 0: VISION-ENHANCED BASELINE")
    print(f"{'='*80}")
    
    results = classify_batch(dataset, max_workers=max_workers, max_items=batch_size)
    metrics = calculate_metrics(results)
    error_analysis = analyze_errors(results)
    
    print("\nVISION-ENHANCED BASELINE METRICS:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print_error_ids(error_analysis["false_positives"], error_analysis["false_negatives"])
    
    results_by_iteration.append(results)
    metrics_by_iteration.append(metrics)
    
    best_f1 = metrics['f1']
    best_iteration = 0
    
    # Iterative improvement based on hypotheses
    for i in range(1, iterations + 1):
        print(f"\n{'='*80}")
        print(f"ITERATION {i}")
        print(f"{'='*80}")
        
        # Create hypothesis about errors from previous iteration
        print("\nANALYZING ERRORS AND FORMING HYPOTHESIS...")
        hypothesis_result = create_hypothesis(
            error_analysis["false_positives"], 
            error_analysis["false_negatives"],
            dataset
        )
        
        print("\nERROR ANALYSIS AND HYPOTHESIS:")
        print(hypothesis_result["analysis"])
        
        hypotheses.append(hypothesis_result["analysis"])
        guidelines_by_iteration[i] = hypothesis_result["guidelines"]
        
        # Run with updated guidelines based on hypothesis
        print("\nRUNNING VISION-ENHANCED CLASSIFICATION WITH UPDATED GUIDELINES...")
        results = classify_batch(dataset, max_workers=max_workers, max_items=batch_size, 
                               iteration=i, additional_guidelines=hypothesis_result["guidelines"])
        
        metrics = calculate_metrics(results)
        error_analysis = analyze_errors(results)
        
        print(f"\nITERATION {i} METRICS:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        print_error_ids(error_analysis["false_positives"], error_analysis["false_negatives"])
        
        results_by_iteration.append(results)
        metrics_by_iteration.append(metrics)
        
        # Track best iteration
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_iteration = i
    
    # Print final comparison
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"Best iteration: {best_iteration} with F1 score: {best_f1:.4f}")
    
    print_metrics_table(metrics_by_iteration)
    
    # Return results from all iterations
    return {
        "best_iteration": best_iteration,
        "best_f1": best_f1,
        "results_by_iteration": results_by_iteration,
        "metrics_by_iteration": metrics_by_iteration,
        "hypotheses": hypotheses,
        "guidelines_by_iteration": guidelines_by_iteration
    }

def run_full_evaluation(batch_size=50, test_size=100):
    """Run a full evaluation with the best vision-enhanced hypothesis-based prompt."""
    dataset = load_dataset("safetykit_onsite_drawstrings_labeled_dataset.json")
    print(f"\n{'='*80}")
    print(f"VISION-ENHANCED HYPOTHESIS-BASED ITERATIVE IMPROVEMENT")
    print(f"{'='*80}")
    print(f"Running iterative improvement with {batch_size} items...")
    
    # Run iterative improvement
    iteration_results = vision_hypothesis_improvement(dataset, iterations=3, batch_size=batch_size)
    best_iteration = iteration_results["best_iteration"]
    best_guidelines = iteration_results["guidelines_by_iteration"][best_iteration]
    
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION")
    print(f"{'='*80}")
    print(f"Running final evaluation with best iteration ({best_iteration}) on {test_size} items...")
    
    # Run final evaluation with best guidelines
    results = classify_batch(
        dataset, 
        max_workers=10, 
        max_items=test_size, 
        iteration=best_iteration, 
        additional_guidelines=best_guidelines
    )
    
    metrics = calculate_metrics(results)
    error_analysis = analyze_errors(results)
    
    print("\nFINAL EVALUATION METRICS:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print_error_ids(error_analysis["false_positives"], error_analysis["false_negatives"])
    
    # Save detailed results
    output = {
        "metrics": metrics,
        "results": results,
        "error_analysis": {
            "false_positive_ids": [fp["id"] for fp in error_analysis["false_positives"]],
            "false_negative_ids": [fn["id"] for fn in error_analysis["false_negatives"]]
        },
        "iteration_history": {
            "best_iteration": best_iteration,
            "metrics_by_iteration": iteration_results["metrics_by_iteration"],
            "hypotheses": iteration_results["hypotheses"],
            "guidelines_by_iteration": iteration_results["guidelines_by_iteration"]
        }
    }
    
    with open("vision_evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nFull evaluation results saved to vision_evaluation_results.json")
    
    return output

def process_dataset(dataset, prompt_refinements=None, max_workers=4):
    """
    Process a dataset of listings with the classifier.
    
    Args:
        dataset: List of product listings
        prompt_refinements: Optional guidelines to add to the prompt
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with results and metrics
    """
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_listing, item, prompt_refinements): item for item in dataset}
        
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
                print(f"Processed item {result['listing_id']}: predicted={result['predicted_label']}, expected={result['true_label']}")
            except Exception as e:
                print(f"Error processing item: {e}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Add processing time to metrics
    metrics["processing_time"] = processing_time
    metrics["items_per_second"] = len(results) / processing_time if processing_time > 0 else 0
    
    return {
        "results": results,
        "metrics": metrics
    }

def improve_classifier(dataset, iterations=3, max_workers=4):
    """
    Iteratively improve the classifier by analyzing errors and refining prompts.
    
    Args:
        dataset: Dataset containing product listings
        iterations: Number of improvement iterations
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of metrics from each iteration
    """
    all_metrics = []
    current_refinements = None
    
    # Extract the data array from the dataset if it's in the expected format
    if isinstance(dataset, dict) and "data" in dataset:
        data_items = dataset["data"]
    else:
        # If it's already an array or has unexpected format, use as is
        data_items = dataset
    
    # Split dataset into training and validation sets
    train_size = int(len(data_items) * 0.7)
    train_dataset = data_items[:train_size]
    valid_dataset = data_items[train_size:]
    
    print(f"Training on {len(train_dataset)} examples, validating on {len(valid_dataset)} examples")
    
    for i in range(iterations):
        print(f"\n=== Iteration {i+1}/{iterations} ===")
        
        # Process training dataset with current prompt refinements
        print("Processing training dataset...")
        train_result = process_dataset(train_dataset, current_refinements, max_workers)
        train_metrics = train_result["metrics"]
        
        print("\nTraining Metrics:")
        print(f"Precision: {train_metrics['precision']:.4f}")
        print(f"Recall: {train_metrics['recall']:.4f}")
        print(f"F1 Score: {train_metrics['f1']:.4f}")
        print(f"Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Analyze errors
        error_analysis = analyze_errors(train_result["results"])
        
        # Combine false positives and false negatives for comprehensive error analysis
        all_errors = error_analysis["false_positives"] + error_analysis["false_negatives"]
        
        print(f"\nFound {len(error_analysis['false_positives'])} false positives and {len(error_analysis['false_negatives'])} false negatives")
        
        # Generate hypothesis and improvement guidelines
        print("Generating hypothesis and improvement guidelines...")
        hypothesis = create_hypothesis(all_errors)
        
        # Extract guidelines from hypothesis
        guidelines = hypothesis
        if "IMPROVEMENT GUIDELINES:" in hypothesis:
            guidelines_section = hypothesis.split("IMPROVEMENT GUIDELINES:")[1]
            guidelines = guidelines_section.strip()
        
        print("\nHypothesis:")
        print(hypothesis[:500] + "..." if len(hypothesis) > 500 else hypothesis)
        
        # Update prompt refinements
        current_refinements = guidelines if not current_refinements else current_refinements + "\n\n" + guidelines
        
        # Validate on validation dataset
        print("\nValidating with updated guidelines...")
        valid_result = process_dataset(valid_dataset, current_refinements, max_workers)
        valid_metrics = valid_result["metrics"]
        
        print("\nValidation Metrics:")
        print(f"Precision: {valid_metrics['precision']:.4f}")
        print(f"Recall: {valid_metrics['recall']:.4f}")
        print(f"F1 Score: {valid_metrics['f1']:.4f}")
        print(f"Accuracy: {valid_metrics['accuracy']:.4f}")
        
        # Save metrics for this iteration
        iteration_metrics = {
            "iteration": i+1,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "hypothesis": hypothesis,
            "guidelines": guidelines,
            "current_refinements": current_refinements
        }
        
        all_metrics.append(iteration_metrics)
    
    return all_metrics

def main():
    """Main entry point for running the classifier."""
    parser = argparse.ArgumentParser(description="Children's Drawstrings Policy Classifier with Hypothesis Learning")
    parser.add_argument('--dataset', type=str, help='Path to dataset JSON file')
    parser.add_argument('--mode', type=str, default='improve', choices=['single', 'batch', 'improve', 'demo'],
                        help='Mode of operation: single (classify one item), batch (process whole dataset), improve (iterative learning), demo (quick demo)')
    parser.add_argument('--iterations', type=int, default=3, help='Number of improvement iterations')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output', type=str, help='Path to output file for results')
    
    args = parser.parse_args()
    
    # Demo mode - run a quick example
    if args.mode == 'demo':
        print("Running demo mode with sample listings...")
        
        # Sample product that should be classified as a violation (drawstrings + children's marketing)
        violation_example = {
            "listing_id": "demo-violation-1",
            "classification": "etsy.childrens_drawstrings",  # True label for comparison
            "title": "Kids Hoodie with Drawstring - Size 8 Boys Winter Jacket",
            "description": """Warm hoodie perfect for cold weather, featuring adjustable drawstrings in the hood.
            Available in children's sizes 6-12. Bright colors kids will love!""",
            "category": "Children's Clothing > Outerwear",
            "keywords": ["kids hoodie", "boys jacket", "children's outerwear", "winter coat"],
            "images": ["https://example.com/kids_hoodie_image.jpg"]  # This URL won't work but the classifier will handle it
        }
        
        # Sample product that should not be classified as a violation (adult clothing)
        non_violation_example = {
            "listing_id": "demo-non-violation-1",
            "classification": "out_of_scope",  # True label for comparison
            "title": "Women's Wool Coat - Elegant Winter Outerwear",
            "description": """Sophisticated wool coat for women. Features side pockets and button closure.
            Available in sizes S, M, L, XL. Perfect for professional wear or evenings out.""",
            "category": "Women's Clothing > Outerwear",
            "keywords": ["women's coat", "wool coat", "winter outerwear", "elegant coat"],
            "images": ["https://example.com/womens_coat_image.jpg"]  # This URL won't work but the classifier will handle it
        }
        
        # Create mini dataset
        demo_dataset = [violation_example, non_violation_example]
        
        # Process the examples
        print("\nClassifying sample products...")
        results = process_dataset(demo_dataset)
        
        # Display detailed results
        print("\n=== CLASSIFICATION RESULTS ===")
        for result in results["results"]:
            print(f"\nItem: {result['listing_id']}")
            print(f"True label: {result['true_label']}")
            print(f"Predicted: {result['predicted_label']}")
            print("Explanation excerpt:")
            print(result['explanation'][:500] + "..." if len(result['explanation']) > 500 else result['explanation'])
            print("-" * 50)
        
        # Show metrics
        print("\n=== METRICS ===")
        metrics = results["metrics"]
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        return
    
    # All other modes require a dataset file
    if not args.dataset:
        print("Error: Dataset file is required for this mode")
        return
    
    print(f"Loading dataset from {args.dataset}")
    try:
        with open(args.dataset, 'r') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} items from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    if args.mode == 'single':
        # Classify a single item from the dataset
        if isinstance(dataset, dict) and "data" in dataset:
            if not dataset["data"]:
                print("Error: Dataset is empty")
                return
            item = dataset["data"][0]  # Use the first item in the dataset
        else:
            if not dataset:
                print("Error: Dataset is empty")
                return
            item = dataset[0]  # Use the first item if not in expected format
            
        # For items in the expected format with reviewInput/expectedOutcome structure
        if "reviewInput" in item and "expectedOutcome" in item:
            # Convert to the format expected by process_listing
            processed_item = {
                "listing_id": item["reviewInput"].get("id", "unknown"),
                "classification": item["expectedOutcome"],
                **item["reviewInput"]
            }
            result = process_listing(processed_item)
        else:
            # Assume the item is already in the correct format
            result = process_listing(item)
        
        print("\n=== CLASSIFICATION RESULT ===")
        print(f"Item: {result['listing_id']}")
        print(f"True label: {result['true_label']}")
        print(f"Predicted: {result['predicted_label']}")
        print("Explanation:")
        print(result['explanation'])
        
    elif args.mode == 'batch':
        # Process the entire dataset without improvement
        print(f"Processing dataset...")
        
        # Extract data items if the dataset is in expected format
        if isinstance(dataset, dict) and "data" in dataset:
            data_items = dataset["data"]
            print(f"Found {len(data_items)} items in dataset")
        else:
            data_items = dataset
            print(f"Processing {len(data_items)} items")
            
        results = process_dataset(data_items, max_workers=args.workers)
        
        # Display metrics
        metrics = results["metrics"]
        print("\n=== METRICS ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"True Positives: {metrics['true_positive']}")
        print(f"False Positives: {metrics['false_positive']}")
        print(f"False Negatives: {metrics['false_negative']}")
        print(f"True Negatives: {metrics['true_negative']}")
        
        # Save results if output file is specified
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
        
    elif args.mode == 'improve':
        # Run the iterative improvement process
        print(f"Running {args.iterations} improvement iterations...")
        results = improve_classifier(dataset, iterations=args.iterations, max_workers=args.workers)
        
        # Display final metrics
        final_metrics = results[-1]["valid_metrics"]
        print("\n=== FINAL METRICS ===")
        print(f"Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Precision: {final_metrics['precision']:.4f}")
        print(f"Recall: {final_metrics['recall']:.4f}")
        print(f"F1 Score: {final_metrics['f1']:.4f}")
        
        # Display final guidelines
        print("\n=== FINAL GUIDELINES ===")
        print(results[-1]["current_refinements"])
        
        # Save results if output file is specified
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")

if __name__ == "__main__":
    main() 