from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import local modules
try:
    from backend.inference import TranslationModel
    from backend.preprocess import TextPreprocessor
    from backend.evaluation import EvaluationModule, get_sample_dataset
    from backend.finetune import get_fine_tuning_info
except ImportError:
    # Fallback for local execution if running from within the backend folder
    from inference import TranslationModel
    from preprocess import TextPreprocessor
    from evaluation import EvaluationModule, get_sample_dataset
    from finetune import get_fine_tuning_info

app = FastAPI(title="Nepali-English Translation API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
# Note: In a real production environment, we'd use dependency injection or a singleton
# Here we initialize them at startup for simplicity
preprocessor = TextPreprocessor()
model = TranslationModel()
evaluator = EvaluationModule()

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "eng_Latn"
    target_lang: str = "ne_NP"

class TranslationResponse(BaseModel):
    translation: str

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Endpoint for translating text between English and Nepali.
    """
    try:
        # Preprocess input text
        cleaned_text = preprocessor.preprocess(request.text, request.source_lang)
        
        # Perform translation
        translation = model.translate(cleaned_text, request.source_lang, request.target_lang)
        
        return TranslationResponse(translation=translation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate")
async def evaluate_model():
    """
    Endpoint for evaluating the model on a sample dataset.
    """
    try:
        dataset = get_sample_dataset()
        results = evaluator.evaluate_model(model, dataset)
        return results
    except Exception as e:
        print("ERROR OCCURRED:", e)   # 👈 ADD THIS LINE
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/finetune/info")
async def finetune_info():
    """
    Returns the LoRA fine-tuning configuration and simulated training
    results for display in the frontend analytics dashboard.
    """
    try:
        return get_fine_tuning_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy", "model": "facebook/nllb-200-distilled-600M"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
