import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TranslationModel:
    """
    Handles model loading and translation inference using NLLB-200.
    """
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("Model loaded successfully.")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translates text from source language to target language.
        """
        self.tokenizer.src_lang = src_lang
        # Prepare inputs
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate translation
        # forced_bos_token_id is used to specify the target language
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
            max_length=128
        )
        
        # Decode output
        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translation
