# detection/api_views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os, joblib, re, requests
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from .models import NewsQuery
import os


import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Lazy-load PIPELINE (tfidf + classifier)
# ------------------------------------------------------------------
model = None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "detection", "ml_model", "fake_news_model.pkl"
)

def get_model():
    global model
    if model is None:
        logger.info("Loading ML pipeline...")
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model type: {type(model)}")
        #safety validation
        if not isinstance(model, Pipeline):
            logger.error(f"Invalid artifact loaded: {type(model)}")
            raise TypeError(f"Expected Pipeline, got {type(model)}")
    return model

# ------------------------------------------------------------------
# Minimal text cleaning
# ------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.strip()

# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
@api_view(["POST"])
def detect_news(request):
    logger.info("Prediction request received")

    text = request.data.get("text", "")
    url = request.data.get("url", "")

    # Fetch text from URL if needed
    if url and not text:
        try:
            logger.info(f"Fetching article from URL: {url}")

            r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()

            soup = BeautifulSoup(r.content, "html.parser")
            text = " ".join(p.text for p in soup.find_all("p"))
        except Exception as e:
            logger.error(f"URL fetch failed: {str(e)}")
            return Response({"error": "Cannot fetch URL"}, status=400)

    if not text:
        logger.warning("No input text provided")
        return Response({"error": "No input provided"}, status=400)
    
    #Prediction
    try:
        model = get_model()
        logger.info(f"Raw input length: {len(text)}")

        cleaned = clean_text(text)
        logger.info(f"Cleaned text length: {len(cleaned)}")

        if not cleaned.strip():
            logger.warning("Cleaned text is empty after preprocessing")
            return Response({"error": "Text empty after cleaning"}, status=400)

        # 🚀 Pipeline handles tfidf internally
        logger.info("Before prediction")
        pred = model.predict([cleaned])[0]
        logger.info("After prediction")

        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba([cleaned])[0].max())
        else:
            confidence = 0.0

        result = "FAKE" if pred == 1 else "REAL"

        logger.info(
            f"Prediction success → {result} "
            f"(confidence={confidence:.3f})"
        )

    except Exception as e:
        logger.exception("Prediction failed")
        return Response(
            {"error": "Prediction failed"},
            status=500
        )

    # Save to DB
    NewsQuery.objects.create(
        text_input=text[:1000],
        url_input=url,
        prediction=result,
        confidence=confidence
    )

    # Response
    return Response({
        "prediction": result,
        "confidence": confidence
    })


#/health endpoint to check if service alive and ready
@api_view(["GET"])
def health_check(request):
    try:
        if not os.path.exists(MODEL_PATH):
            return Response(
                {"status": "error", "detail": "Model file missing"},
                status=500
            )

        return Response(
            {
                "status": "ok",
                "model_loaded": model is not None
            },
            status=200
        )

    except Exception as e:
        return Response(
            {"status": "error", "detail": str(e)},
            status=500
        )




