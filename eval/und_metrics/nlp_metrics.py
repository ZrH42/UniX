import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

class Tokenizer:
    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)

def calculate_bleu_metrics(output, label):
    """Calculate BLEU metric for a single text"""
    smoothing_function = SmoothingFunction().method4
    
    output_tokens = word_tokenize(output)
    label_tokens = word_tokenize(label)

    if not output_tokens:
        return None

    bleu1 = sentence_bleu([label_tokens], output_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    bleu2 = sentence_bleu([label_tokens], output_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    bleu3 = sentence_bleu([label_tokens], output_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
    bleu4 = sentence_bleu([label_tokens], output_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

    return {
        'BLEU1': bleu1,
        'BLEU2': bleu2,
        'BLEU3': bleu3,
        'BLEU4': bleu4,
    }

def compute_rouge(
    predictions,
    references,
    rouge_types=None,
    use_aggregator=True,
    use_stemmer=False,
    tokenizer=None,
    confidence_interval=0.95,
    n_samples=500
):
    """Calculate ROUGE metric"""
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    multi_ref = isinstance(references[0], list)

    if tokenizer is not None:
        tokenizer = Tokenizer(tokenizer)

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer)
    if use_aggregator:
        aggregator = scoring.BootstrapAggregator(
            confidence_interval=confidence_interval, n_samples=n_samples
        )
    else:
        scores = []

    for ref, pred in zip(references, predictions):
        if multi_ref:
            score = scorer.score_multi(ref, pred)
        else:
            score = scorer.score(ref, pred)
        if use_aggregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)

    if use_aggregator:
        result = aggregator.aggregate()
        for key in result:
            result[key] = {
                'median': result[key].mid.fmeasure,
                'ci_l': result[key].low.fmeasure,
                'ci_h': result[key].high.fmeasure,
            }
    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key].fmeasure for score in scores)

    return result

def calculate_nlp_metrics(predictions, references):
    """Calculate all NLP metrics including BLEU and ROUGE"""
    all_metrics = {
        'BLEU1': [],
        'BLEU2': [],
        'BLEU3': [],
        'BLEU4': [],
    }

    # Calculate BLEU metrics
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Calculating BLEU metrics"):
        metrics = calculate_bleu_metrics(pred, ref)
        if metrics:
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

    # Calculate average BLEU
    avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items() if values}

    # Calculate ROUGE-L
    rouge_scores = compute_rouge(
        predictions=predictions,
        references=references,
        rouge_types=["rougeL"],
        use_aggregator=True,
        use_stemmer=False
    )
    
    avg_metrics['ROUGE_L'] = rouge_scores['rougeL']['median']
    return avg_metrics 