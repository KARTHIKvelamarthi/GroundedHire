import os
import re
import sys

# Ensure core directory is in python search path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.join(_script_dir, "core")
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

import json
import random
import unicodedata
from collections import Counter
import fitz  # PyMuPDF
import tiktoken
import config

# Ensure standard output and error use UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Try to load tiktoken encoding for token count estimations
try:
    _encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoding = None


def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in the given text using tiktoken,
    falling back to a word count proxy (1.3 tokens per word) if unavailable.
    
    Args:
        text (str): The input text to tokenise.
        
    Returns:
        int: The estimated token count.
    """
    if _encoding is not None:
        return len(_encoding.encode(text))
    else:
        return int(len(text.split()) * 1.3)


def clean_page_text(text: str) -> str:
    """
    Cleans text by decomposing Unicode ligatures and normalising multiple whitespace characters.
    
    Args:
        text (str): The raw text span or line.
        
    Returns:
        str: The cleaned, normalised text.
    """
    # Decompose ligatures (e.g., '\ufb01' -> 'fi')
    normalized = unicodedata.normalize("NFKD", text)
    # Convert fancy quotes to standard quotes
    normalized = normalized.replace("’", "'").replace("“", '"').replace("”", '"')
    # Normalise multiple consecutive spaces/tabs to a single space
    return " ".join(normalized.split())


def normalize_for_header_footer(text: str) -> str:
    """
    Prepares text for header/footer comparison by converting to lowercase
    and stripping all non-alphabetic characters.
    
    Args:
        text (str): The raw line text.
        
    Returns:
        str: The normalised comparison key.
    """
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return " ".join(text_clean.split())


def extract_visual_lines_from_page(page: fitz.Page) -> list[dict]:
    """
    Extracts text blocks from the page and groups lines that lie on the same 
    vertical baseline (within 3 points) to form visual lines. Groups are sorted
    top-to-bottom and left-to-right.
    
    Args:
        page (fitz.Page): The PyMuPDF page object.
        
    Returns:
        list[dict]: A list of visual lines, each containing text, bbox, and average size.
    """
    blocks = page.get_text("dict").get("blocks", [])
    raw_lines = []
    
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            raw_lines.append(line)
            
    if not raw_lines:
        return []
        
    # Sort raw lines by y0 coordinate first, then x0
    raw_lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    
    grouped_lines = []
    current_group = [raw_lines[0]]
    
    for line in raw_lines[1:]:
        # If the vertical baseline difference is small, group them as the same visual line
        if abs(line["bbox"][1] - current_group[-1]["bbox"][1]) < 3.0:
            current_group.append(line)
        else:
            grouped_lines.append(current_group)
            current_group = [line]
    if current_group:
        grouped_lines.append(current_group)
        
    visual_lines = []
    for group in grouped_lines:
        # Sort left-to-right within the same visual baseline group
        group.sort(key=lambda l: l["bbox"][0])
        
        merged_spans = []
        for line in group:
            merged_spans.extend(line.get("spans", []))
            
        span_texts = []
        for span in merged_spans:
            t = span.get("text", "")
            if t:
                span_texts.append(t)
        line_text = " ".join(span_texts).strip()
        line_text = clean_page_text(line_text)
        
        if not line_text:
            continue
            
        # Compute character-weighted average font size
        total_len = 0
        weighted_sum = 0.0
        for span in merged_spans:
            t = span.get("text", "").strip()
            if t:
                weighted_sum += span.get("size", 0.0) * len(t)
                total_len += len(t)
        avg_size = weighted_sum / total_len if total_len > 0 else 0.0
        
        # Calculate visual bounding box bounding all grouped lines
        x0 = min(l["bbox"][0] for l in group)
        y0 = min(l["bbox"][1] for l in group)
        x1 = max(l["bbox"][2] for l in group)
        y1 = max(l["bbox"][3] for l in group)
        
        visual_lines.append({
            "text": line_text,
            "bbox": (x0, y0, x1, y1),
            "avg_size": avg_size
        })
        
    return visual_lines


def is_non_content_page(lines: list[dict], page_num: int) -> bool:
    """
    Applies heuristics to check if a page is a non-content page (e.g., Table
    of Contents, Index, or blank/cover pages).
    
    Args:
        lines (list[dict]): Visual lines extracted from the page.
        page_num (int): 0-based page index.
        
    Returns:
        bool: True if the page is classified as non-content.
    """
    total_text = " ".join([l["text"] for l in lines]).strip()
    if len(total_text) < 100:
        return True
        
    # Table of Contents dot leaders check (e.g. "Linear Regression ... 42")
    dot_lines = 0
    for l in lines:
        text = l["text"]
        if "..." in text or ". ." in text or re.search(r'\.\s*\.\s*\.\s*\.', text):
            dot_lines += 1
            
    if len(lines) > 0 and (dot_lines / len(lines) > 0.15):
        return True
        
    # Explicit TOC or Index header keywords check
    is_index_or_toc_header = False
    for l in lines[:5]:
        text_lower = l["text"].lower().strip()
        if text_lower in ["contents", "table of contents", "index", "subject index", "author index"]:
            is_index_or_toc_header = True
            break
            
    if is_index_or_toc_header:
        digit_count = sum(1 for c in total_text if c.isdigit())
        if len(total_text) > 0 and (digit_count / len(total_text) > 0.05 or len(lines) < 40):
            return True
            
    return False


def extract_pages(pdf_path: str) -> tuple[list[dict], float]:
    """
    Processes all pages of a PDF in a single layout-extraction pass. Pre-extracts
    and groups lines, filters non-content pages, computes median font size,
    detects running headers/footers, and returns a list of filtered visual lines.
    
    Args:
        pdf_path (str): File path to the PDF.
        
    Returns:
        tuple[list[dict], float]: A list of content line dictionaries, and the document's median font size.
    """
    doc = fitz.open(pdf_path)
    
    # 1. Single Pass: Group visual lines and gather body font sizes
    all_pages_lines = []
    font_sizes = []
    index_started = False
    
    for p_idx in range(len(doc)):
        page = doc[p_idx]
        height = page.rect.height
        raw_lines = extract_visual_lines_from_page(page)
        
        # Stateful index detection: once we hit the index page, we stop processing all subsequent pages
        for l in raw_lines[:15]:
            text_lower = l["text"].lower().strip()
            text_clean = re.sub(r'[^a-z\s]', '', text_lower).strip()
            if text_clean in ["index", "indexes", "subject index", "author index", "index of terms"]:
                print(f"[INFO] Index detected starting at page {p_idx+1}. Skipping this and all remaining pages.")
                index_started = True
                break
        if index_started:
            break
            
        # Skip TOC/Index and blank cover pages
        if is_non_content_page(raw_lines, p_idx):
            continue
            
        all_pages_lines.append((p_idx + 1, height, raw_lines))
        
        # Accumulate char-weighted line sizes for document median calculation
        for line in raw_lines:
            text_len = len(line["text"])
            if text_len > 0:
                font_sizes.append((line["avg_size"], text_len))
                
    # Compute median font size
    if not font_sizes:
        median_size = 10.0
    else:
        font_sizes.sort(key=lambda x: x[0])
        total_weight = sum(w for _, w in font_sizes)
        cumulative_weight = 0
        median_size = font_sizes[-1][0]
        for size, w in font_sizes:
            cumulative_weight += w
            if cumulative_weight >= total_weight / 2:
                median_size = size
                break
                
    # 2. Detect repeated header/footer patterns from top/bottom 12% regions
    line_counts = Counter()
    for _, height, lines in all_pages_lines:
        for line in lines:
            y0, y1 = line["bbox"][1], line["bbox"][3]
            is_top = y1 < height * 0.12
            is_bottom = y0 > height * 0.88
            
            if is_top or is_bottom:
                norm = normalize_for_header_footer(line["text"])
                words = norm.split()
                if len(words) < 12 and norm:
                    line_counts[norm] += 1
                    
    threshold = max(5, int(len(doc) * 0.05))
    repeated_patterns = {k for k, v in line_counts.items() if v >= threshold}
    
    # 3. Filter page numbers and repeated headers/footers
    page_number_pattern = re.compile(r'^[\s\-\[\]\(\)]*\d+[\s\-\[\]\(\)]*$')
    filtered_lines = []
    
    for page_num, height, lines in all_pages_lines:
        for line in lines:
            y0, y1 = line["bbox"][1], line["bbox"][3]
            is_top = y1 < height * 0.12
            is_bottom = y0 > height * 0.88
            
            if is_top or is_bottom:
                norm = normalize_for_header_footer(line["text"])
                if norm in repeated_patterns:
                    continue
                if page_number_pattern.match(line["text"]):
                    continue
                    
            filtered_lines.append({
                "text": line["text"],
                "avg_size": line["avg_size"],
                "page": page_num,
                "bbox": line["bbox"],
                "page_height": height
            })
            
    return filtered_lines, median_size


def detect_headings(line_text: str, line_avg_size: float, doc_median_size: float) -> bool:
    """
    Determines if a line is a section heading.
    
    A line is treated as a heading if it matches a numbered section pattern (e.g., "3.2 Decision Trees")
    or is noticeably larger than the median font size (+2pt) and short (< 12 words).
    
    Args:
        line_text (str): Text of the line.
        line_avg_size (float): Character-weighted average size of the line.
        doc_median_size (float): Median body font size for the document.
        
    Returns:
        bool: True if the line is a heading.
    """
    # Rule 1: Numbered-section pattern via regex (e.g., 3.2 Decision Trees, 10 Neural Networks)
    # Exclude numbers starting with 0 and ensure the line is short (not a list item/sentence)
    if re.match(r'^[1-9]\d*(\.\d+)*\.?\s+[A-Z]', line_text):
        word_count = len(line_text.split())
        if word_count < 12:
            return True
        
    # Rule 2: Noticeably larger than median body font size (+2pt or more) and short (< 12 words)
    word_count = len(line_text.split())
    if word_count < 12 and line_avg_size >= doc_median_size + 2.0:
        # Avoid single-character, symbol-only, or very short noise lines in diagrams being treated as headings
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', line_text)
        if len(clean_text) >= 3:
            return True
            
    # Rule 3: Explicit standard section keywords (case-insensitive) to ensure references/appendices are split
    clean_upper = re.sub(r'[^A-Z\s]', '', line_text.upper()).strip()
    if clean_upper in ["REFERENCES", "BIBLIOGRAPHY", "FURTHER READING", "APPENDIX", "APPENDICES", "SUMMARY AND FURTHER READING", "REFERENCES AND FURTHER READING"]:
        return True
        
    return False


def group_lines_into_elements(doc_lines: list[dict], doc_median_size: float) -> list[dict]:
    """
    Groups visual lines into headings and paragraphs, preserving their source page ranges
    and identifying paragraph breaks.
    
    Args:
        doc_lines (list[dict]): List of visual lines from the document.
        doc_median_size (float): Document's median body text font size.
        
    Returns:
        list[dict]: List of elements (headings or paragraphs).
    """
    elements = []
    current_paragraph_lines = []
    
    for idx, line in enumerate(doc_lines):
        text = line["text"]
        avg_size = line["avg_size"]
        page = line["page"]
        y0 = line["bbox"][1]
        page_height = line.get("page_height", 1000.0)
        
        # Define footnote/footer region (bottom ~15-18% of the page)
        is_bottom_region = y0 > page_height * 0.82
        
        # Check if this line is footnote/footer content (bottom region and smaller font size)
        is_footnote_content = is_bottom_region and (avg_size < doc_median_size - 0.8)
        
        # If it is footnote/footer content, strip it from the chunk entirely (preferred)
        if is_footnote_content:
            continue
            
        # Heading boundary check
        is_heading = detect_headings(text, avg_size, doc_median_size)
        
        # If a heading candidate falls within the bottom region, do NOT treat it as a heading boundary
        if is_heading and is_bottom_region:
            is_heading = False
            
        if is_heading:
            if current_paragraph_lines:
                elements.append(make_paragraph_element(current_paragraph_lines))
                current_paragraph_lines = []
            elements.append({
                "type": "heading",
                "text": text,
                "page": page
            })
            continue
            
        if not current_paragraph_lines:
            current_paragraph_lines.append(line)
            continue
            
        prev_line = current_paragraph_lines[-1]
        is_break = False
        
        if prev_line["page"] == line["page"]:
            # Same page: check vertical spacing gap
            gap = line["bbox"][1] - prev_line["bbox"][3]
            prev_line_height = prev_line["bbox"][3] - prev_line["bbox"][1]
            
            # Standard paragraph spacing gap check
            if gap > 0.5 * prev_line_height or gap > 5.0:
                is_break = True
            # Check for punctuation end + line indent
            elif prev_line["text"].strip()[-1] in [".", "?", "!", "\"", "”"] and line["bbox"][0] > prev_line["bbox"][0] + 10:
                is_break = True
        else:
            # Page transition: break if the previous line ended with sentence punctuation
            last_char = prev_line["text"].strip()[-1]
            if last_char in [".", "?", "!", "\"", "”"]:
                is_break = True
                
        if is_break:
            elements.append(make_paragraph_element(current_paragraph_lines))
            current_paragraph_lines = [line]
        else:
            current_paragraph_lines.append(line)
            
    if current_paragraph_lines:
        elements.append(make_paragraph_element(current_paragraph_lines))
        
    return elements


def make_paragraph_element(lines: list[dict]) -> dict:
    """
    Assembles a list of lines into a single paragraph element dict.
    """
    text = " ".join([l["text"] for l in lines]).strip()
    text = " ".join(text.split())
    page_start = min(l["page"] for l in lines)
    page_end = max(l["page"] for l in lines)
    return {
        "type": "paragraph",
        "text": text,
        "page_start": page_start,
        "page_end": page_end
    }


def group_elements_into_sections(elements: list[dict]) -> list[dict]:
    """
    Groups elements into sections based on heading boundaries.
    
    Args:
        elements (list[dict]): List of headings and paragraphs.
        
    Returns:
        list[dict]: List of sections with title and paragraphs.
    """
    sections = []
    current_title = None
    current_paragraphs = []
    
    for el in elements:
        if el["type"] == "heading":
            if current_paragraphs or current_title is not None:
                sections.append({
                    "title": current_title,
                    "paragraphs": current_paragraphs
                })
            current_title = el["text"]
            current_paragraphs = []
        else:
            current_paragraphs.append(el)
            
    if current_paragraphs or current_title is not None:
        sections.append({
            "title": current_title,
            "paragraphs": current_paragraphs
        })
    return sections


def split_paragraph_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences based on punctuation followed by a capital letter.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z\d])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_long_paragraph(text: str, target_size: int = 500) -> list[str]:
    """
    Helper to split a single paragraph that exceeds 900 tokens into smaller sub-paragraphs
    at sentence boundaries.
    """
    sentences = split_paragraph_into_sentences(text)
    sub_paragraphs = []
    current_sentences = []
    current_tokens = 0
    
    for s in sentences:
        s_tokens = count_tokens(s)
        if current_tokens + s_tokens > target_size and current_sentences:
            sub_paragraphs.append(" ".join(current_sentences))
            current_sentences = [s]
            current_tokens = s_tokens
        else:
            current_sentences.append(s)
            current_tokens += s_tokens
            
    if current_sentences:
        sub_paragraphs.append(" ".join(current_sentences))
    return sub_paragraphs


def chunk_section(
    section_title: str | None,
    section_paragraphs: list[dict],
    target_size: int = 500,
    max_size: int = 900,
    overlap_target: int = 75
) -> list[dict]:
    """
    Packs paragraph elements of a single section into chunks, ensuring:
      - Headings are hard chunk boundaries (never carry overlap or content across).
      - No paragraph is cut mid-thought unless it alone exceeds max_size (900 tokens).
      - ~15% token overlap (75 tokens) between chunks within the same section.
      
    Args:
        section_title (str | None): Heading title.
        section_paragraphs (list[dict]): Paragraph elements belonging to the section.
        target_size (int): Target token count (~500).
        max_size (int): Max token count before forcing a split (~900).
        overlap_target (int): Target token overlap.
        
    Returns:
        list[dict]: List of chunk dicts.
    """
    # 1. Preprocess paragraphs: split any that exceed max_size (900 tokens)
    processed_paragraphs = []
    for p in section_paragraphs:
        p_tokens = count_tokens(p["text"])
        if p_tokens > max_size:
            sub_paras = split_long_paragraph(p["text"], target_size=target_size)
            for sp in sub_paras:
                processed_paragraphs.append({
                    "text": sp,
                    "page_start": p["page_start"],
                    "page_end": p["page_end"],
                    "was_force_split": True
                })
        else:
            processed_paragraphs.append({
                "text": p["text"],
                "page_start": p["page_start"],
                "page_end": p["page_end"],
                "was_force_split": False
            })
            
    # 2. Greedily pack paragraphs into chunks
    chunks = []
    i = 0
    n = len(processed_paragraphs)
    
    while i < n:
        chunk_paras = []
        chunk_tokens = 0
        was_split = False
        
        j = i
        while j < n:
            p = processed_paragraphs[j]
            p_tokens = count_tokens(p["text"])
            
            # Stop if adding this paragraph exceeds max size
            if chunk_paras and chunk_tokens + p_tokens > max_size:
                break
                
            chunk_paras.append(p)
            joined_text = " ".join([cp["text"] for cp in chunk_paras])
            chunk_tokens = count_tokens(joined_text)
            
            if p["was_force_split"]:
                was_split = True
                
            j += 1
            
            # Stop adding if we've reached or exceeded the target size
            if chunk_tokens >= target_size:
                break
                
        chunk_text = " ".join([cp["text"] for cp in chunk_paras])
        page_start = min(cp["page_start"] for cp in chunk_paras)
        page_end = max(cp["page_end"] for cp in chunk_paras)
        
        chunks.append({
            "section_title": section_title,
            "page_start": page_start,
            "page_end": page_end,
            "was_force_split": was_split,
            "text": chunk_text,
            "token_count": chunk_tokens
        })
        
        if j >= n:
            break
            
        # Calculate next starting index for overlap (only if we have more than 1 paragraph)
        if j - i > 1:
            overlap_tokens = 0
            k = j - 1
            while k > i:
                joined_overlap = " ".join([processed_paragraphs[m]["text"] for m in range(k, j)])
                o_tokens = count_tokens(joined_overlap)
                if o_tokens >= overlap_target:
                    break
                k -= 1
            next_i = max(i + 1, k)
        else:
            next_i = j
            
        i = next_i
        
    return chunks


def assign_difficulty(filename: str) -> str:
    """
    Hardcoded mapping of PDF filename to difficulty tier.
    
    Args:
        filename (str): The PDF file name.
        
    Returns:
        str: Difficulty tier string.
    """
    fn_lower = filename.lower()
    if "absolute" in fn_lower or "beginners" in fn_lower:
        return "beginner"
    elif "mitchell" in fn_lower or "python" in fn_lower:
        return "intermediate"
    elif "burkov" in fn_lower or "brownlee" in fn_lower or "hundred" in fn_lower:
        return "intermediate-dense"
    else:
        return "intermediate"


# Topic Trigger Taxonomy Keywords mapping
TOPIC_KEYWORDS = {
    "supervised_learning": [r"\bsupervised\s+learning\b", r"\bsupervised\s+classification\b", r"\bsupervised\s+regression\b"],
    "unsupervised_learning": [r"\bunsupervised\s+learning\b", r"\bunsupervised\s+clustering\b", r"\bautoencoder\b"],
    "decision_trees": [r"\bdecision\s+tree\b", r"\bdecision\s+trees\b", r"\brandom\s+forest\b", r"\bxgboost\b", r"\bgradient\s+boosting\b", r"\bcart\b", r"\binformation\s+gain\b"],
    "neural_networks": [r"\bneural\s+network\b", r"\bneural\s+networks\b", r"\bmultilayer\s+perceptron\b", r"\bmlp\b", r"\bperceptron\b", r"\bbackpropagation\b", r"\bneuron\b", r"\bneurons\b"],
    "clustering": [r"\bclustering\b", r"\bclusters\b", r"\bk-means\b", r"\bkmeans\b", r"\bhierarchical\s+clustering\b", r"\bdbscan\b"],
    "regression": [r"\bregression\b", r"\bregressions\b", r"\blinear\s+regression\b", r"\blogistic\s+regression\b", r"\bleast\s+squares\b", r"\bridge\b", r"\blasso\b"],
    "classification": [r"\bclassification\b", r"\bclassifier\b", r"\bclassifiers\b", r"\bclassify\b", r"\blogistic\s+regression\b", r"\bsvm\b", r"\bsupport\s+vector\b", r"\bnaive\s+bayes\b"],
    "overfitting_regularization": [r"\boverfitting\b", r"\bunderfitting\b", r"\bregularization\b", r"\bregularize\b", r"\bdropout\b", r"\bearly\s+stopping\b"],
    "evaluation_metrics": [r"\bevaluation\s+metrics?\b", r"\baccuracy\b", r"\bprecision\b", r"\brecall\b", r"\bf1\b", r"\broc\b", r"\bauc\b", r"\bconfusion\s+matrix\b", r"\bmse\b", r"\bmae\b"],
    "feature_engineering": [r"\bfeature\s+engineering\b", r"\bfeature\s+selection\b", r"\bone-hot\b", r"\bnormalization\b", r"\bscaling\b", r"\bstandardization\b"],
    "optimization": [r"\boptimization\b", r"\bgradient\s+descent\b", r"\bsgd\b", r"\badam\b", r"\boptimizer\b", r"\bloss\s+function\b"],
    "probability_statistics": [r"\bprobability\b", r"\bstatistics\b", r"\bbayesian?\b", r"\bgaussian\b", r"\bvariance\b", r"\bstandard\s+deviation\b"],
    "model_selection": [r"\bmodel\s+selection\b", r"\bcross-validation\b", r"\bcross\s+validation\b", r"\bgrid\s+search\b", r"\bhyperparameters?\b"],
    "ensemble_methods": [r"\bensemble\b", r"\bbagging\b", r"\bboosting\b", r"\brandom\s+forest\b", r"\badaboost\b"],
    "deep_learning": [r"\bdeep\s+learning\b", r"\bcnn\b", r"\brnn\b", r"\blstm\b", r"\btransformers?\b"],
    "dimensionality_reduction": [r"\bdimensionality\s+reduction\b", r"\bpca\b", r"\bprincipal\s+component\b", r"\bt-?sne\b", r"\blda\b"]
}


def assign_topics(text: str) -> list[str]:
    """
    Performs case-insensitive regex keyword matching to assign taxonomy topics.
    
    Args:
        text (str): Chunk text.
        
    Returns:
        list[str]: Matched topics list.
    """
    topics = []
    for topic, regexes in TOPIC_KEYWORDS.items():
        for regex in regexes:
            if re.search(regex, text, re.IGNORECASE):
                topics.append(topic)
                break
    return topics


def slugify(text: str) -> str:
    """
    Converts text to a lowercase alphanumeric slug with underscores.
    """
    clean = text.lower()
    clean = re.sub(r'[^a-z0-9]+', '_', clean)
    return clean.strip('_')


def process_role_folder(role_name: str, role_dir: str) -> list[dict]:
    """
    Processes all PDFs in a role's subfolder and constructs the required schema per chunk.
    
    Args:
        role_name (str): Role identifier.
        role_dir (str): Subfolder directory path.
        
    Returns:
        list[dict]: List of all processed chunks.
    """
    all_chunks = []
    if not os.path.exists(role_dir):
        print(f"[ERROR] Subfolder not found: {role_dir}")
        return []
        
    files = [f for f in os.listdir(role_dir) if f.endswith(".pdf")]
    
    for filename in files:
        pdf_path = os.path.join(role_dir, filename)
        source_book = os.path.splitext(filename)[0]
        book_slug = slugify(source_book)
        difficulty = assign_difficulty(filename)
        
        print(f"\nProcessing: {filename} ({role_name})")
        
        try:
            # Step 1: Text extraction (page-aware, single layout-extraction pass)
            doc_lines, doc_median = extract_pages(pdf_path)
            
            # Step 2: Group lines into structural elements
            elements = group_lines_into_elements(doc_lines, doc_median)
            
            # Step 3: Group into sections
            sections = group_elements_into_sections(elements)
            
            # Step 4: Chunk sections
            book_chunks = []
            for sec in sections:
                title = sec["title"]
                if title:
                    title_lower = title.lower()
                    ref_patterns = ["references", "bibliography", "further reading", "appendix", "appendices"]
                    meta_patterns = [
                        "how to read this book", "preface", "about this book", "acknowledgments",
                        "about the author", "about the authors", "foreword", "introduction to this book",
                        "who this book is for", "table of contents"
                    ]
                    if any(p in title_lower for p in ref_patterns) or any(p in title_lower for p in meta_patterns):
                        # Skip bibliography/references/further reading/appendix and generic book-meta sections
                        continue
                        
                sec_chunks = chunk_section(sec["title"], sec["paragraphs"])
                book_chunks.extend(sec_chunks)
                
            # Step 5: Wrap in metadata schema
            for idx, chunk in enumerate(book_chunks, start=1):
                chunk_id = f"{role_name}_{book_slug}_{idx:04d}"
                
                # Check for force split warnings
                if chunk["was_force_split"]:
                    print(f"[WARNING] Force split paragraph in chunk '{chunk_id}' on pages {chunk['page_start']}-{chunk['page_end']}.")
                    
                topics = assign_topics(chunk["text"])
                
                schema_chunk = {
                    "chunk_id": chunk_id,
                    "role": role_name,
                    "source_book": source_book,
                    "section_title": chunk["section_title"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "difficulty_tier": difficulty,
                    "topics": topics,
                    "token_count": chunk["token_count"],
                    "was_force_split": chunk["was_force_split"],
                    "text": chunk["text"]
                }
                all_chunks.append(schema_chunk)
                
            print(f"Done processing: {filename}. Created {len(book_chunks)} chunks.")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    return all_chunks


def main():
    """
    Main execution pipeline: processes role folders and prints evaluation stats.
    """
    roles = ["ai_ml_engineer", "data_science_applied_ml"]
    all_role_chunks = {}
    review_flagged_chunks = []
    
    # Process folders
    for role in roles:
        role_dir = os.path.join("role_pdfs", role)
        chunks = process_role_folder(role, role_dir)
        all_role_chunks[role] = chunks
        
        # Output JSON file per role
        output_filename = os.path.join(config.CHUNKS_DIR, f"chunks_{role}.json")
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(chunks)} chunks to {output_filename}")
        
        # Check for chunks with 3+ topics
        for chunk in chunks:
            if len(chunk["topics"]) >= 3:
                review_flagged_chunks.append(chunk)
                
    # Output flagged review chunks
    flagged_filename = os.path.join(config.CHUNKS_DIR, "review_flagged_chunks.json")
    with open(flagged_filename, "w", encoding="utf-8") as f:
        json.dump(review_flagged_chunks, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(review_flagged_chunks)} review flagged chunks to {flagged_filename}")
    
    # Summary & Verification Stats
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    
    total_chunks = 0
    zero_topic_chunks = 0
    force_split_chunks = 0
    
    difficulty_counts = Counter()
    role_book_counts = Counter()
    all_token_counts = []
    global_chunks = []
    
    for role, chunks in all_role_chunks.items():
        for chunk in chunks:
            total_chunks += 1
            global_chunks.append(chunk)
            role_book_counts[(role, chunk["source_book"])] += 1
            difficulty_counts[chunk["difficulty_tier"]] += 1
            all_token_counts.append(chunk["token_count"])
            
            if not chunk["topics"]:
                zero_topic_chunks += 1
            if chunk["was_force_split"]:
                force_split_chunks += 1
                
    print("\n--- Total Chunks Per Role and Book ---")
    for (role, book), count in sorted(role_book_counts.items()):
        print(f"  {role} | {book}: {count} chunks")
        
    print("\n--- Chunk Count Per Difficulty Tier ---")
    for tier, count in sorted(difficulty_counts.items()):
        print(f"  {tier}: {count} chunks")
        
    print(f"\n--- Coverage and Quality Flags ---")
    print(f"  Total Chunks Processed: {total_chunks}")
    print(f"  Chunks with Zero Topic Matches: {zero_topic_chunks}")
    print(f"  Force-split Chunks: {force_split_chunks}")
    print(f"  Chunks Flagged for Review (3+ Topics): {len(review_flagged_chunks)}")
    
    if total_chunks > 0:
        avg_tokens = sum(all_token_counts) / len(all_token_counts)
        min_tokens = min(all_token_counts)
        max_tokens = max(all_token_counts)
        print(f"\n--- Token Statistics ---")
        print(f"  Average Token Count: {avg_tokens:.1f}")
        print(f"  Min Token Count: {min_tokens}")
        print(f"  Max Token Count: {max_tokens}")
        
        # Pick 3 random sample chunks
        print("\n" + "="*50)
        print("RANDOM CHUNK SAMPLES (3)")
        print("="*50)
        samples = random.sample(global_chunks, min(3, len(global_chunks)))
        for s_idx, sample in enumerate(samples, 1):
            print(f"\n--- Sample {s_idx} [{sample['chunk_id']}] ---")
            print(f"Source Book: {sample['source_book']}")
            print(f"Difficulty: {sample['difficulty_tier']}")
            print(f"Section Title: {sample['section_title']}")
            print(f"Page Range: {sample['page_start']}-{sample['page_end']}")
            print(f"Topics: {sample['topics']}")
            print(f"Token Count: {sample['token_count']}")
            print(f"Text:\n{sample['text']}")
            print("-" * 50)


if __name__ == "__main__":
    main()
