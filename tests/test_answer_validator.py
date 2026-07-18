import os
import sys
import json
import time
import warnings

# Suppress all future and deprecation warnings to keep the terminal output clean
warnings.simplefilter("ignore")

# Add parent directory to sys.path to import modules from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core")))

from answer_validator import validate_answer, summarize_session

# Ensure standard output uses UTF-8 to prevent Windows terminal printing encoding errors
sys.stdout.reconfigure(encoding='utf-8')

# Case A: Bias-Variance tradeoff (Beginner)
QUESTION_A = "Can you explain the bias-variance trade-off in machine learning and describe its significance?"

CHUNK_TEXT_A = """In Figure 3, we can see two lines moving from left to right. The line above represents the test data and the line below represents the training data. From the left, both lines begin at a point of high prediction error due to low variance and high bias. As they move from left to right they change to the opposite: high variance and low bias. This leads to low prediction error in the case of the training data and high prediction error for the test data. In the middle of the chart is an optimal balance of prediction error between the training and test data. This is a common case of bias-variance trade-off. Figure 4: Underfitting on the left and overfitting on the right Mismanaging the bias-variance trade-off can lead to poor results. As seen in Figure 4, this can result in the model becoming overly simple and inflexible (underfitting) or overly complex and flexible (overfitting). Underfitting (low variance, high bias) on the left and overfitting (high variance, low bias) on the right are shown in these two scatterplots. A natural temptation is to add complexity to the model (as shown on the right) in order to improve accuracy, but which can, in turn, lead to overfitting. An overfitted model will yield accurate predictions from the training data but prove less accurate at formulating predictions from the test data. Overfitting can also occur if the training and test data aren't randomized before they are split and patterns in the data aren't distributed across the two segments of data. Underfitting is when your model is overly simple, and again, has not scratched the surface of the underlying patterns in the dataset. Underfitting can lead to inaccurate predictions for both the training data and test data. Common causes of underfitting include insufficient training data to adequately cover all possible combinations, and situations where the training and test data were not properly randomized. To eradicate both underfitting and overfitting, you may need to modify the model's hyperparameters to ensure that they fit patterns in both the training and test data and not just one-half of the data. A suitable fit should acknowledge major trends in the data and play down or even omit minor variations. This may also mean re-randomizing the training and test data or adding new data points so as to better detect underlying patterns. However, in most instances, you will probably need to consider switching algorithms or modifying your hyperparameters based on trial and error to minimize and manage the issue of bias-variance trade-off. Specifically, this might entail switching from linear regression to non-linear regression to reduce bias by increasing variance. Or it could mean increasing " k " in k -NN to reduce variance (by averaging together more neighbors). A third example could be reducing variance by switching from a single decision tree (which is prone to overfitting) to a random forest with many decision trees. Another effective strategy to combat overfitting and underfitting is to introduce regularization . Regularization artificially amplifies bias error by penalizing an increase in a model's complexity. In effect, this add-on parameter provides a warning alert to keep high variance in check while the original parameters are being optimized. Another effective technique to contain overfitting and underfitting in your model is to perform cross validation, as covered earlier in Chapter 6, to minimize any discrepancies between the training data and the test data."""

EXPECTED_POINTS_A = [
    "The bias-variance trade-off is a fundamental concept in machine learning that deals with the balance between bias (simplicity of a model) and variance (complexity of a model).",
    "A good model should strike a balance between these two aspects to minimize both high prediction error (high bias, low variance) for training data and high prediction error (high variance, low bias) for test data. The optimal balance leads to low overall error.",
    "Underfitting occurs when the model is too simple and fails to capture important patterns in the data, resulting in high bias and high error for both training and test data. Overfitting happens when the model is excessively complex, capturing noise as well as signal, leading to high variance and low error for training data but high error for test data."
]

ANSWERS_A = {
    "strong": "The bias-variance trade-off is about balancing model simplicity (bias) and complexity (variance). High bias leads to underfitting because the model is too simple and cannot capture the patterns, causing high error on both train and test sets. High variance leads to overfitting because the model is too complex and fits noise in the training set, causing low training error but high test error. The goal is to find the optimal balance in the middle that minimizes total prediction error.",
    "partial": "Bias is when the model is underfitting and variance is when the model is overfitting. We want to balance them so that the model does not have too much error.",
    "weak": "I am not completely sure, maybe it relates to how biased the dataset is."
}


# Case B: Cross-Validation / Model Selection (Intermediate)
QUESTION_B = "Can you explain the concept of cross-validation and how it differs from using a test set for evaluating machine learning models? Additionally, describe the potential issue when using the same data for both model selection and evaluation, and provide an example of how to avoid this issue?"

CHUNK_TEXT_B = """In this chapter we discussed cross-validation, grid search, and evaluation metrics, the cornerstones of evaluating and improving machine learning algorithms. The tools described in this chapter, together with the algorithms described in Chapters 2 and 3 , are the bread and butter of every machine learning practitioner. There are two particular points that we made in this chapter that warrant repeating, because they are often overlooked by new practitioners. The first has to do with cross-validation. Cross-validation or the use of a test set allow us to evaluate a machine learning model as it will perform in the future. However, if we use the test set or cross-validation to select a model or select model parameters, we \"use up\" the test data, and using the same data to evaluate how well our model will do in the future will lead to overly optimistic estimates. We therefore need to resort to a split into training data for model building, validation data for model and parameter selection, and test data for model evaluation. Instead of a simple split, we can replace each of these splits with cross-validation. The most commonly used form (as described ear‐ lier) is a training/test split for evaluation, and using cross-validation on the training set for model and parameter selection. The second point has to do with the importance of the evaluation metric or scoring function used for model selection and model evaluation. The theory of how to make business decisions from the predictions of a machine learning model is somewhat beyond the scope of this book. 7 However, it is rarely the case that the end goal of a machine learning task is building a model with a high accuracy. Make sure that the metric you choose to evaluate and select a model for is a good stand-in for what the model will actually be used for. In reality, classification problems rarely have balanced classes, and often false positives and false negatives have very different consequences. Make sure you understand what these consequences are, and pick an evaluation met‐ ric accordingly. The model evaluation and selection techniques we have described so far are the most important tools in a data scientist's toolbox. Grid search and cross-validation as we've described them in this chapter can only be applied to a single supervised model. We have seen before, however, that many models require preprocessing, and that in some applications, like the face recognition example in Chapter 3 , extracting a different representation of the data can be useful. In the next chapter, we will introduce the Pipeline class, which allows us to use grid search and cross-validation on these com‐ plex chains of algorithms."""

EXPECTED_POINTS_B = [
    "Cross-validation is a technique used for evaluating the performance of a machine learning model by splitting the dataset into multiple subsets. One subset is used to train the model, another is used to validate it (for tuning hyperparameters or selecting models), and the remaining subset is used for testing the final model.",
    "Using the same data for both model selection and evaluation can lead to overly optimistic estimates of the model's performance. This is because the test set is 'used up' during the selection process, resulting in biased evaluations.",
    "To avoid this issue, a split into training data for model building, validation data for model and parameter selection, and test data for model evaluation can be employed. Alternatively, each of these splits can be replaced with cross-validation."
]

ANSWERS_B = {
    "strong": "Cross-validation splits the data repeatedly into train/val subsets to evaluate parameters and select models. It differs from a single test set which is only used for the final model's performance evaluation. If we use the same test set (or cross-validation) for both selecting the model/parameters and final evaluation, we 'use up' the data and get overly optimistic estimates of performance. To avoid this, we must use a three-way split: training data for model building, validation data for parameter/model selection, and a separate test set for final evaluation. Or we can replace these splits with nested cross-validation.",
    "partial": "Cross-validation is about splitting the data multiple times. If we use the same data to choose a model and test it, it will look too good. We should have a train, validation, and test set.",
    "weak": "skip"
}

def run_evaluation(case_id: str, question: str, expected_points: list[str], source_text: str, answers: dict, topic: str, tier: str) -> list[dict]:
    results = []
    print(f"\n=======================================================")
    print(f"EVALUATING CASE: {case_id} ({topic} - {tier})")
    print(f"=======================================================")
    
    for quality, candidate_ans in answers.items():
        print(f"\nEvaluating {quality.upper()} candidate answer:")
        print(f"Answer text: '{candidate_ans}'")
        
        start_time = time.time()
        try:
            eval_res = validate_answer(
                question=question,
                candidate_answer=candidate_ans,
                expected_answer_points=expected_points,
                source_chunk_text=source_text
            )
            elapsed = time.time() - start_time
            
            # Tag results for session summary
            eval_res["topic"] = topic
            eval_res["difficulty_tier"] = tier
            
            print(f"Evaluation succeeded in {elapsed:.2f} seconds.")
            print(json.dumps(eval_res, indent=2, ensure_ascii=False))
            results.append(eval_res)
        except Exception as e:
            print(f"[ERROR] Evaluation failed for {quality} answer: {e}")
            
    return results

# Case C: Discrepancy checking (Project question)
QUESTION_C = "Describe the model architecture, framework, and evaluation metrics used in your healthcare triage classification system."
PROJECT_DESC_C = "Developed a healthcare triage classification system using Random Forest in scikit-learn, achieving a 92% validation accuracy."
EXPECTED_POINTS_C = [
    "Used Random Forest classification model.",
    "Built within scikit-learn framework.",
    "Achieved 92% accuracy metric."
]
ANSWER_C = "I designed a deep convolutional neural network (CNN) model using PyTorch that achieved a 99% accuracy on healthcare triage."

def main():
    print("Starting Answer Validator Validation Suite...")
    
    all_results = []
    
    # Run Case A
    results_a = run_evaluation(
        case_id="Case A: Bias-Variance",
        question=QUESTION_A,
        expected_points=EXPECTED_POINTS_A,
        source_text=CHUNK_TEXT_A,
        answers=ANSWERS_A,
        topic="overfitting_regularization",
        tier="beginner"
    )
    all_results.extend(results_a)
    
    # Run Case B
    results_b = run_evaluation(
        case_id="Case B: Cross-Validation",
        question=QUESTION_B,
        expected_points=EXPECTED_POINTS_B,
        source_text=CHUNK_TEXT_B,
        answers=ANSWERS_B,
        topic="model_selection",
        tier="intermediate"
    )
    all_results.extend(results_b)

    # Run Case C: Discrepancy Verification
    print("\n=======================================================")
    print("EVALUATING CASE C: Resume Discrepancy Checking")
    print("=======================================================")
    print(f"Project Description: '{PROJECT_DESC_C}'")
    print(f"Candidate Answer: '{ANSWER_C}'")
    
    start_time = time.time()
    try:
        eval_res = validate_answer(
            question=QUESTION_C,
            candidate_answer=ANSWER_C,
            expected_answer_points=EXPECTED_POINTS_C,
            source_chunk_text="No textbook passage (project question).",
            project_description=PROJECT_DESC_C
        )
        elapsed = time.time() - start_time
        print(f"Evaluation succeeded in {elapsed:.2f} seconds.")
        print(json.dumps(eval_res, indent=2, ensure_ascii=False))
        all_results.append(eval_res)
    except Exception as e:
        print(f"[ERROR] Evaluation failed for Case C: {e}")
    
    # Run session summary
    print("\n=======================================================")
    print("RUNNING SESSION SUMMARY AGGREGATION")
    print("=======================================================")
    
    summary = summarize_session(all_results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

