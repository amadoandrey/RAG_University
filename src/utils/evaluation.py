from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score


def evaluar_respuesta(query: str, respuesta: str, referencia: str = "") -> dict:
    """
    Evalúa la calidad de una respuesta generada por el modelo comparándola con una referencia esperada.

    Se calculan tres métricas automáticas de evaluación:
      - BLEU: mide la coincidencia de palabras/n-gramas entre respuesta y referencia.
      - ROUGE-L: mide la similitud basada en la subsecuencia común más larga.
      - METEOR: evalúa coincidencias considerando sinónimos y orden flexible.

    Args:
        query (str): Pregunta original formulada por el usuario. (No se usa en el cálculo, 
                     pero se deja para posibles ampliaciones futuras).
        respuesta (str): Texto generado por el modelo que se desea evaluar.
        referencia (str, opcional): Texto de referencia esperado contra el cual comparar la respuesta.
                                    Por defecto: cadena vacía.

    Returns:
        dict: Diccionario con las métricas calculadas:
              {
                  "BLEU": float,
                  "ROUGE-L": float,
                  "METEOR": float
              }
    """
    # --- BLEU ---
    # Mide la similitud basada en n-gramas (palabras consecutivas).
    # Se aplica un suavizado (SmoothingFunction) para evitar valores 0 en frases cortas.
    bleu = sentence_bleu(
        [referencia.split()],
        respuesta.split(),
        smoothing_function=SmoothingFunction().method1
    )

    # --- ROUGE ---
    # ROUGE-L compara la subsecuencia común más larga entre referencia y respuesta.
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(referencia, respuesta)

    # --- METEOR ---
    # Evalúa considerando coincidencias exactas, sinónimos y variaciones de palabras.
    meteor = meteor_score([referencia.split()], respuesta.split())

    return {
        "BLEU": bleu,
        "ROUGE-L": rouge["rougeL"].fmeasure,
        "METEOR": meteor
    }
