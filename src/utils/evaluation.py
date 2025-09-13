from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

def evaluar_respuesta(query, respuesta, referencia=""):
    # BLEU
    bleu = sentence_bleu([referencia.split()], respuesta.split(), smoothing_function=SmoothingFunction().method1)
    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(referencia, respuesta)
    # METEOR
    meteor = meteor_score([referencia.split()], respuesta.split())

    return {"BLEU": bleu, "ROUGE-L": rouge["rougeL"].fmeasure, "METEOR": meteor}
