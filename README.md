# Cross-Lingual Language Modeling System: Nepali-English Translator

A full-stack, production-quality cross-lingual translation system for low-resource Nepali NLP that runs completely locally on CPU-only machines (MacBook compatible). This project demonstrates real-world NLP engineering with a React + TypeScript frontend and a Python FastAPI backend using pretrained NLLB-200 models.

## Project Overview

This system provides:

- **Real-time Translation**: English ↔ Nepali bidirectional translation using the NLLB-200 distilled model
- **Local Execution**: Runs entirely on your machine without cloud dependencies or proprietary services
- **CPU Optimization**: Designed for MacBook and standard laptops (no GPU required)
- **Production Code**: Clean, modular, importable Python modules with proper package structure
- **Evaluation Metrics**: BLEU score calculation for translation quality assessment
- **Modern UI**: React + TypeScript frontend with real-time translation feedback

## Technology Stack

**Backend:**
- Python 3.8+
- FastAPI (async web framework)
- HuggingFace Transformers (NLLB-200 model)
- PyTorch (inference engine)
- SacreBLEU (evaluation metrics)

**Frontend:**
- React 19 with TypeScript
- Tailwind CSS 4
- shadcn/ui components
- Vite (build tool)

## Project Structure

```
project/
├── backend/
│   ├── __init__.py           # Python package marker
│   ├── main.py               # FastAPI application & endpoints
│   ├── inference.py          # Model loading & translation logic
│   ├── preprocess.py         # Text cleaning & normalization
│   ├── evaluation.py         # BLEU score calculation & evaluation
│
├── frontend/
│   ├── package.json
│   ├── client/
│   │   ├── src/
│   │   │   ├── pages/Home.tsx    # Main translation UI
│   │   │   ├── App.tsx
│   │   │   ├── index.css
│   │   │   └── main.tsx
│   │   ├── public/
│   │   └── index.html
│   └── server/
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm/pnpm
- 4GB+ RAM (for model loading)
- ~2GB disk space (for model download)

### Backend Setup

1. **Navigate to project root:**
   ```bash
   cd project
   ```

2. **Create a Python virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - FastAPI & Uvicorn (web framework)
   - PyTorch (inference engine)
   - Transformers (model loading)
   - SacreBLEU (evaluation)

4. **Start the backend server:**
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   **First run note:** The first time you run the backend, it will download the NLLB-200 model (~1.2GB). This is a one-time operation and will be cached locally.

   Expected output:
   ```
   Loading model facebook/nllb-200-distilled-600M on cpu...
   Model loaded successfully.
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

### Frontend Setup

1. **In a new terminal, navigate to frontend directory:**
   ```bash
   cd project/frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   # or
   pnpm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   # or
   pnpm dev
   ```

   The frontend will be available at `http://localhost:5173` (or the port shown in terminal).

## API Specification

### POST /translate

Translates text from source language to target language.

**Request:**
```json
{
  "text": "Hello, how are you?",
  "source_lang": "eng_Latn",
  "target_lang": "ne_NP"
}
```

**Response:**
```json
{
  "translation": "नमस्ते, तपाईं कस्तो हुनुहुन्छ?"
}
```

**Language Codes:**
- English: `eng_Latn`
- Nepali: `ne_NP`

### GET /evaluate

Evaluates the model on a sample Nepali-English dataset.

**Response:**
```json
{
  "average_bleu": 25.5,
  "detailed_results": [
    {
      "source": "नमस्ते, तपाईंलाई कस्तो छ?",
      "reference": "Hello, how are you?",
      "hypothesis": "Hello, how are you?",
      "bleu": 100.0
    }
    // ... more results
  ]
}
```

### GET /health

Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "facebook/nllb-200-distilled-600M"
}
```

## Usage

1. **Start both backend and frontend** (as described in Setup)
2. **Open the frontend** at `http://localhost:5173`
3. **Enter text** in the English input field
4. **Translation appears automatically** in the Nepali output field
5. **Swap languages** using the bidirectional arrow button
6. **Evaluate model** by checking the backend logs or calling `/evaluate` endpoint

## Code Quality & Architecture

### Backend Modules

**inference.py:**
- `TranslationModel` class handles model loading and inference
- Uses HuggingFace Transformers API
- Supports CPU-only inference
- Language-agnostic design (works with any NLLB-200 language pair)

**preprocess.py:**
- `TextPreprocessor` class for text cleaning
- Removes extra whitespaces and normalizes text
- Extensible for language-specific preprocessing

**evaluation.py:**
- `EvaluationModule` class for quality assessment
- Calculates BLEU scores for translation pairs
- Includes sample Nepali-English dataset for testing

**main.py:**
- FastAPI application with CORS enabled
- Proper error handling and HTTP status codes
- Async endpoints for non-blocking I/O
- Imports all modules correctly for production use

### Frontend Components

**Home.tsx:**
- Real-time translation with 500ms debounce
- Language selection dropdowns
- Bidirectional language swap
- Loading states and error handling
- Character count display
- Responsive two-column layout

## Performance Considerations

- **Model Size**: NLLB-200 distilled (600M parameters) optimized for CPU
- **Inference Speed**: ~2-5 seconds per sentence on CPU (MacBook M1/M2)
- **Memory Usage**: ~2GB RAM during operation
- **Offline Capability**: Works completely offline after first model download

## Troubleshooting

**Backend fails to start:**
- Ensure Python 3.8+ is installed: `python3 --version`
- Check all dependencies: `pip list | grep -E "fastapi|torch|transformers"`
- Try removing `.cache` directory if model download is corrupted

**Frontend can't connect to backend:**
- Verify backend is running on `http://localhost:8000`
- Check CORS is enabled (it is by default in main.py)
- Look for network errors in browser console (F12)

**Translation is very slow:**
- First inference is slower (model compilation)
- Subsequent translations should be faster
- CPU usage is normal during translation

**Model download fails:**
- Check internet connection
- Ensure ~2GB free disk space
- Try again; HuggingFace servers may be temporarily unavailable

## Testing & Validation

Run the evaluation endpoint to test model quality:

```bash
curl http://localhost:8000/evaluate
```

This will return BLEU scores for sample translations. Expected average BLEU: 20-30 (reasonable for low-resource language pairs).

## Production Deployment

For production use:

1. **Use a proper ASGI server** (Gunicorn + Uvicorn):
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app
   ```

2. **Build frontend for production:**
   ```bash
   npm run build
   ```

3. **Consider containerization** (Docker) for consistent deployment

4. **Add authentication** if exposing over network

5. **Implement rate limiting** to prevent abuse

## Limitations & Future Work

- NLLB-200 is optimized for modern languages; Nepali support is good but not perfect
- Single-sentence translation only (no document-level context)
- No fine-tuning support in this implementation
- Frontend runs on localhost only (add reverse proxy for remote access)

## References

- [NLLB-200 Model](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Transformers Library](https://huggingface.co/docs/transformers/)

## License

This project is provided as-is for educational and research purposes.

## Author Notes

This system demonstrates production-quality full-stack NLP engineering with:
- Clean, modular Python code following best practices
- Proper error handling and logging
- Real-time React UI with proper state management
- Complete documentation and setup instructions
- CPU-optimized inference for accessibility

Suitable for final-year academic submission combining software engineering and NLP components.
