import sacrebleu

class EvaluationModule:
    """
    Handles evaluation of translation quality using BLEU score.
    """
    def __init__(self):
        pass

    def calculate_bleu(self, hypothesis: str, reference: str) -> float:
        """
        Calculates BLEU score for a single hypothesis and reference.
        """
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return bleu.score

    def evaluate_model(self, model, dataset: list) -> dict:
        """
        Evaluates model on a small dataset of (source, reference) pairs.
        """
        results = []
        total_bleu = 0
        
        for src, ref, src_lang, tgt_lang in dataset:
            hyp = model.translate(src, src_lang, tgt_lang)
            score = self.calculate_bleu(hyp, ref)
            results.append({
                "source": src,
                "reference": ref,
                "hypothesis": hyp,
                "bleu": score
            })
            total_bleu += score
            
        avg_bleu = total_bleu / len(dataset) if dataset else 0
        return {
            "average_bleu": avg_bleu,
            "detailed_results": results
        }

def get_sample_dataset():
    return [
        ("How are you?", "तिमी कस्तो छौ?", "eng_Latn", "npi_Deva"),
        ("Good morning", "शुभ प्रभात", "eng_Latn", "npi_Deva"),
        ("Good evening", "शुभ सन्ध्या", "eng_Latn", "npi_Deva"),
        ("Thank you", "धन्यवाद", "eng_Latn", "npi_Deva"),
        ("I am fine", "म ठिक छु", "eng_Latn", "npi_Deva"),
        ("What is your name?", "तिम्रो नाम के हो?", "eng_Latn", "npi_Deva"),
        ("Where are you going?", "तिमी कहाँ जाँदै छौ?", "eng_Latn", "npi_Deva"),
        ("I love Nepal", "म नेपाललाई माया गर्छु", "eng_Latn", "npi_Deva"),
        ("Today is sunny", "आज घाम लागेको छ", "eng_Latn", "npi_Deva"),

        ("The weather is nice", "मौसम राम्रो छ", "eng_Latn", "npi_Deva"),
        ("I am learning Nepali", "म नेपाली सिक्दैछु", "eng_Latn", "npi_Deva"),
        ("This is my friend", "यो मेरो साथी हो", "eng_Latn", "npi_Deva"),
        ("Please help me", "कृपया मलाई मद्दत गर्नुहोस्", "eng_Latn", "npi_Deva"),
        ("I am hungry", "म भोको छु", "eng_Latn", "npi_Deva"),
        ("Open the door", "ढोका खोल्नुहोस्", "eng_Latn", "npi_Deva"),
        ("Close the window", "झ्याल बन्द गर्नुहोस्", "eng_Latn", "npi_Deva"),
        ("What time is it?", "अहिले कति बजे?", "eng_Latn", "npi_Deva"),
        ("I like music", "मलाई संगीत मन पर्छ", "eng_Latn", "npi_Deva"),
        ("This is important", "यो महत्त्वपूर्ण छ", "eng_Latn", "npi_Deva"),

        ("Where do you live?", "तिमी कहाँ बस्छौ?", "eng_Latn", "npi_Deva"),
        ("I am a student", "म विद्यार्थी हुँ", "eng_Latn", "npi_Deva"),
        ("Let’s go", "जाऔं", "eng_Latn", "npi_Deva"),
        ("Be careful", "सावधान रहनुहोस्", "eng_Latn", "npi_Deva"),
        ("I am tired", "म थाकेको छु", "eng_Latn", "npi_Deva"),
    ]
