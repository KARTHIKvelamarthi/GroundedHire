import os
import sys
# Add parent directory to sys.path so we can import from the root directory modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from question_generator import check_copy_overlap

def test_detector():
    chunk_text = """
    Regularization is an umbrella-term that encompasses methods that force the learning algorithm to build a less complex model.
    In practice, that often leads to slightly higher bias but significantly reduces the variance.
    This problem is known in the literature as the bias-variance tradeoff.
    The two most widely used types of regularization are called L1 regularization and L2 regularization.
    """
    
    # Case A: A well-phrased original question (uses keywords but no contiguous copied phrases)
    q_original = "Can you describe the bias-variance tradeoff and explain how L1 and L2 regularization can help reduce model complexity?"
    ratio_orig, warning_orig = check_copy_overlap(q_original, chunk_text, n=8)
    print(f"Original Question Check:")
    print(f"  Question: '{q_original}'")
    print(f"  Computed Word Overlap Ratio: {ratio_orig:.4f}")
    print(f"  Warning Triggered: {warning_orig} (Expected: False)")
    
    # Case B: A near-verbatim copied phrase (plagiarism/copying)
    # The phrase "methods that force the learning algorithm to build a less complex model" is 11 words verbatim
    q_copied = "What are the methods that force the learning algorithm to build a less complex model?"
    ratio_copy, warning_copy = check_copy_overlap(q_copied, chunk_text, n=8)
    print(f"\nCopied Question Check:")
    print(f"  Question: '{q_copied}'")
    print(f"  Computed Word Overlap Ratio: {ratio_copy:.4f}")
    print(f"  Warning Triggered: {warning_copy} (Expected: True)")
    
    # Assert correctness
    assert not warning_orig, "Failed: Detector false-positive on original question!"
    assert warning_copy, "Failed: Detector failed to catch verbatim copying!"
    print("\nSanity Check Successful! Detector correctly differentiates original questions from copied phrases.")

if __name__ == "__main__":
    test_detector()
