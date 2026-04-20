import re

class TextPreprocessor:
    """
    Handles text cleaning and normalization for Nepali and English.
    """
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning: remove extra whitespaces, strip.
        """
        if not text:
            return ""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def normalize_nepali(self, text: str) -> str:
        """
        Specific normalization for Nepali text if needed.
        (e.g., handling different Unicode representations of similar characters)
        """
        # For this implementation, we keep it simple but extensible
        return text

    def preprocess(self, text: str, lang: str = "eng_Latn") -> str:
        """
        Main preprocessing pipeline.
        """
        text = self.clean_text(text)
        if lang == "ne_NP":
            text = self.normalize_nepali(text)
        return text
