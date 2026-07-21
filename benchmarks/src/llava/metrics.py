from typing import List, Dict

def compute_mcq_metrics(sample_results: List[Dict], dataset_name: str = None) -> Dict:
    """
    Computes MCQ metrics from a list of sample predictions.
    
    Args:
        sample_results: List of dicts, where each dict represents a single sample prediction.
                        Must contain 'question_type', 'correct_answer', 'predicted_answer', 
                        and 'caption_types'. (predicted_answer = -1 indicates parse failure)
        dataset_name: Optional prefix to add to all metric keys.
        
    Returns:
        A dictionary containing the computed metrics (total_accuracy, accuracy by type, 
        wrong answer percentages, etc.).
    """
    correct_answers_sum = 0
    correct_answers_by_type = {"positive": 0, "negative": 0, "hybrid": 0}
    total_questions_by_type = {"positive": 0, "negative": 0, "hybrid": 0}
    wrong_answer_counts_by_type = {"hybrid": 0, "positive": 0, "negative": 0}
    predictions_by_type = {"positive": 0, "negative": 0, "hybrid": 0}
    wrong_answers_by_question_type = {
        "positive": {"positive": 0, "negative": 0, "hybrid": 0},
        "negative": {"positive": 0, "negative": 0, "hybrid": 0},
        "hybrid":   {"positive": 0, "negative": 0, "hybrid": 0},
    }

    total_questions = len(sample_results)
    if total_questions == 0:
        return {}

    for sample in sample_results:
        q_type = sample.get("question_type")
        if not q_type:
            continue
            
        pred_idx = sample.get("predicted_answer", -1)
        corr_idx = sample.get("correct_answer", -1)
        caption_types = sample.get("caption_types", [])

        total_questions_by_type[q_type] = total_questions_by_type.get(q_type, 0) + 1

        if pred_idx == corr_idx:
            correct_answers_sum += 1
            correct_answers_by_type[q_type] = correct_answers_by_type.get(q_type, 0) + 1
            predictions_by_type[q_type] = predictions_by_type.get(q_type, 0) + 1
        elif pred_idx == -1:
            wrong_answer_counts_by_type["parse_failure"] = wrong_answer_counts_by_type.get("parse_failure", 0) + 1
        else:
            if pred_idx < len(caption_types):
                predicted_type = caption_types[pred_idx]
            else:
                predicted_type = "unknown"
                
            wrong_answer_counts_by_type[predicted_type] = wrong_answer_counts_by_type.get(predicted_type, 0) + 1
            predictions_by_type[predicted_type] = predictions_by_type.get(predicted_type, 0) + 1
            
            if q_type not in wrong_answers_by_question_type:
                wrong_answers_by_question_type[q_type] = {}
            wrong_answers_by_question_type[q_type][predicted_type] = wrong_answers_by_question_type[q_type].get(predicted_type, 0) + 1

    def safe_div(a, b): return a / b if b > 0 else float('nan')

    total_wrong = sum(wrong_answer_counts_by_type.values())
    wrong_answer_percentages = {
        k: (v / total_wrong) * 100 if total_wrong > 0 else 0.0
        for k, v in wrong_answer_counts_by_type.items()
    }
    
    try:
        most_common_wrong = max(wrong_answer_counts_by_type, key=wrong_answer_counts_by_type.get)
    except ValueError:
        most_common_wrong = None

    metrics = {
        'total_accuracy': safe_div(correct_answers_sum, total_questions),
        'positive_accuracy': safe_div(correct_answers_by_type.get('positive', 0), total_questions_by_type.get('positive', 0)),
        'negative_accuracy': safe_div(correct_answers_by_type.get('negative', 0), total_questions_by_type.get('negative', 0)),
        'hybrid_accuracy': safe_div(correct_answers_by_type.get('hybrid', 0), total_questions_by_type.get('hybrid', 0)),
        'most_common_wrong_answer_type': most_common_wrong,
        'wrong_answer_percentages': wrong_answer_percentages,
        'predictions_by_type': predictions_by_type,
        'wrong_answers_by_question_type': wrong_answers_by_question_type,
        'sample_results': sample_results
    }

    if dataset_name:
        metrics = {f"{dataset_name}-{k}": v for k, v in metrics.items()}
        
    return metrics
