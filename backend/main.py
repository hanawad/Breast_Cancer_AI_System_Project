"""
Breast Cancer AI Assessment - FastAPI backend.
DISCLAIMER: This is a decision-support prototype. It does not diagnose disease and
must not replace evaluation by a qualified clinician.
"""
import io
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train_clinical import FEATURES, train_and_save  # noqa: E402
from models.train_vision import (  # noqa: E402
    CLASS_NAMES,
    build_model,
    create_demo_checkpoint,
    get_eval_transform,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bc-api")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
DISCLAIMER = (
    "This AI output is decision support only and is not a medical diagnosis. "
    "Please consult a qualified physician for evaluation and next steps."
)

STATE: Dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_models() -> None:
    """Load (and, if missing, create demo) model artifacts."""
    clinical_files = [MODELS_DIR / n for n in ("clinical_model.pkl", "scaler.pkl", "model_meta.json")]
    if not all(p.exists() for p in clinical_files):
        log.warning("Clinical artifacts missing - training demo model on synthetic data.")
        train_and_save(MODELS_DIR, verbose=False)

    vision_path = MODELS_DIR / "vision_model.pth"
    if not vision_path.exists():
        log.warning("vision_model.pth missing - writing untrained DEMO checkpoint.")
        create_demo_checkpoint(vision_path)

    STATE["clinical_model"] = joblib.load(MODELS_DIR / "clinical_model.pkl")
    STATE["scaler"] = joblib.load(MODELS_DIR / "scaler.pkl")
    STATE["clinical_meta"] = json.loads((MODELS_DIR / "model_meta.json").read_text())

    ckpt = torch.load(vision_path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    STATE["vision_model"] = model
    STATE["vision_classes"] = list(ckpt["classes"])
    STATE["vision_meta"] = {
        "arch": ckpt["arch"],
        "source": ckpt.get("source", "unknown"),
        "val_acc": ckpt.get("val_acc"),
        "is_demo": ckpt.get("source") != "trained",
    }
    STATE["transform"] = get_eval_transform()
    log.info("Models loaded: clinical=%s vision=%s", STATE["clinical_meta"]["source"], STATE["vision_meta"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    STATE.clear()


app = FastAPI(
    title="Breast Cancer AI Assessment API",
    version="1.0.0",
    description="Clinical + ultrasound decision-support API (prototype).",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def complete_clinical_features(values: Dict[str, float], medians: Dict[str, float]) -> Dict[str, float]:
    """
    Intelligently complete all 30 features from the provided inputs (e.g. 5 primary features).
    Calculates proportional standard errors (_se) and worst-case values (_worst) for provided means,
    and fills unprovided features using dataset baseline medians to prevent dimension mismatch.
    """
    completed = dict(values)
    primary_bases = ["radius", "texture", "perimeter", "area", "smoothness"]

    for base in primary_bases:
        mean_key = f"{base}_mean"
        se_key = f"{base}_se"
        worst_key = f"{base}_worst"

        if mean_key in completed and completed[mean_key] is not None:
            m_val = float(completed[mean_key])
            base_median_mean = float(medians.get(mean_key, 1.0) or 1.0)

            # Proportionally scale Standard Error (_se) if not explicitly provided
            if se_key not in completed and se_key in medians:
                se_ratio = float(medians[se_key]) / base_median_mean
                completed[se_key] = round(m_val * se_ratio, 6)

            # Proportionally scale Worst Value (_worst) if not explicitly provided
            if worst_key not in completed and worst_key in medians:
                worst_ratio = float(medians[worst_key]) / base_median_mean
                completed[worst_key] = round(m_val * worst_ratio, 4)

    # Impute any remaining missing features using dataset medians
    for k, v in medians.items():
        if k not in completed or completed[k] is None:
            completed[k] = float(v)

    return completed


def predict_clinical(values: Dict[str, float]) -> Dict[str, Any]:
    """values: {feature_name: number}. Intelligently expands and imputes features to 30 dimensions."""
    meta = STATE["clinical_meta"]
    feats = meta["features"]                      # exact order used in training
    medians = meta.get("impute_medians", {})

    full_values = complete_clinical_features(values, medians)
    row = [float(full_values.get(f, medians.get(f, 0.0))) for f in feats]
    x = pd.DataFrame([row], columns=feats)        # DataFrame keeps feature names (no sklearn warning)
    prob = float(STATE["clinical_model"].predict_proba(STATE["scaler"].transform(x))[0, 1])
    return {
        "risk_probability": prob,
        "risk_score": round(prob * 100, 1),
        "feature_importances": meta["feature_importances"],
        "patient_values": dict(zip(feats, row)),
    }


def predict_vision(image_bytes: bytes) -> Dict[str, Any]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    tensor = STATE["transform"](img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(STATE["vision_model"](tensor), dim=1)[0].numpy()
    classes = STATE["vision_classes"]
    top = int(np.argmax(probs))
    return {
        "prediction": classes[top],
        "confidence": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
    }


def risk_level(score: float) -> str:
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Moderate"
    return "High"


def build_summary(vision: Dict[str, Any], clinical: Dict[str, Any]) -> Dict[str, Any]:
    mal = vision["probabilities"].get("Malignant", 0.0)
    integrated = 0.6 * mal + 0.4 * clinical["risk_probability"]
    level = risk_level(integrated)
    actions = {
        "Low": "No high-risk signals detected. Continue routine screening per clinical guidelines.",
        "Moderate": "Some concerning signals. Recommend clinician review and possible short-interval follow-up imaging.",
        "High": "Multiple high-risk signals. Recommend prompt specialist referral and tissue diagnosis (biopsy) as clinically indicated.",
    }
    text = (
        f"Ultrasound model suggests '{vision['prediction']}' ({vision['confidence']*100:.1f}% confidence). "
        f"Clinical risk model estimates {clinical['risk_score']}% risk. "
        f"Integrated risk: {level} ({integrated*100:.1f}%). {actions[level]}"
    )
    return {
        "integrated_score": round(integrated * 100, 1),
        "risk_level": level,
        "text": text,
        "weights": {"ultrasound_malignant_probability": 0.6, "clinical_risk_probability": 0.4},
    }


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    answer: str
    engine: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/clinical-schema")
def clinical_schema():
    """Feature names + default (median) values, plus metadata for simplified primary features."""
    meta = STATE["clinical_meta"]
    medians = meta.get("impute_medians", {})
    primary = [
        {
            "id": "radius_mean",
            "name_ar": "متوسط حجم الورم (نصف القطر)",
            "name_en": "Tumor Radius",
            "default": round(float(medians.get("radius_mean", 13.34)), 2),
            "min": 5.0,
            "max": 35.0,
            "step": 0.1,
            "unit": "mm",
            "tooltip": "متوسط المسافة من المركز إلى نقاط المحيط (نصف القطر). القيمة المرجعية الطبيعية حوالي 13.34 مم.",
        },
        {
            "id": "texture_mean",
            "name_ar": "حدة الملمس / الكثافة",
            "name_en": "Tumor Texture",
            "default": round(float(medians.get("texture_mean", 18.90)), 2),
            "min": 5.0,
            "max": 45.0,
            "step": 0.1,
            "unit": "grayscale std",
            "tooltip": "الانحراف المعياري لقيم التدرج الرمادي (كثافة وتباين الملمس). القيمة المرجعية حوالي 18.90.",
        },
        {
            "id": "perimeter_mean",
            "name_ar": "المحيط الخارجي للورم",
            "name_en": "Tumor Perimeter",
            "default": round(float(medians.get("perimeter_mean", 86.18)), 2),
            "min": 30.0,
            "max": 200.0,
            "step": 0.5,
            "unit": "mm",
            "tooltip": "المحيط الإجمالي لكتلة الورم. القيمة المرجعية الطبيعية حوالي 86.18 مم.",
        },
        {
            "id": "area_mean",
            "name_ar": "مساحة الورم",
            "name_en": "Tumor Area",
            "default": round(float(medians.get("area_mean", 546.4)), 1),
            "min": 100.0,
            "max": 2600.0,
            "step": 1.0,
            "unit": "mm²",
            "tooltip": "المساحة الإجمالية المقاسة للورم. القيمة المرجعية حوالي 546.4 مم².",
        },
        {
            "id": "smoothness_mean",
            "name_ar": "درجة انسيابية / نعومة الحدود",
            "name_en": "Tumor Smoothness",
            "default": round(float(medians.get("smoothness_mean", 0.09524)), 4),
            "min": 0.04,
            "max": 0.20,
            "step": 0.001,
            "unit": "local variation",
            "tooltip": "التباين المحلي في أطوال أنصاف الأقطار (نعومة الحدود). القيمة المرجعية حوالي 0.0952.",
        },
    ]
    return {
        "features": meta["features"],
        "defaults": medians,
        "primary_features": primary,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "clinical_model": {
            "source": STATE["clinical_meta"]["source"],
            "is_demo": STATE["clinical_meta"]["is_demo"],
            "roc_auc_test": STATE["clinical_meta"]["roc_auc_test"],
        },
        "vision_model": STATE["vision_meta"],
        "llm_enabled": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


@app.post("/analyze")
def analyze(
    clinical_json: str = Form(..., description="JSON object {feature_name: value}"),
    file: UploadFile = File(...),
):
    try:
        clinical_values = json.loads(clinical_json)
        if not isinstance(clinical_values, dict):
            raise ValueError
        clinical_values = {str(k): float(v) for k, v in clinical_values.items()}
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="clinical_json must be a JSON object of numeric values.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    vision = predict_vision(data)
    clinical = predict_clinical(clinical_values)
    summary = build_summary(vision, clinical)

    warnings = []
    if STATE["vision_meta"]["is_demo"]:
        warnings.append("Ultrasound model is an UNTRAINED demo checkpoint - its output is meaningless. Run models/train_vision.py.")
    if STATE["clinical_meta"]["is_demo"]:
        warnings.append("Clinical model was trained on SYNTHETIC data - not clinically validated.")

    return {
        "ultrasound": vision,
        "clinical": clinical,
        "summary": summary,
        "model_status": {
            "vision": STATE["vision_meta"],
            "clinical": {"source": STATE["clinical_meta"]["source"], "is_demo": STATE["clinical_meta"]["is_demo"]},
        },
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }


def _clinical_intelligence_answer(question: str, ctx: Optional[Dict[str, Any]]) -> str:
    """Intelligent context-grounded medical explanation generator (fallback Transformer emulator)."""
    q = question.lower()
    parts = []

    us_pred = "غير محدد"
    us_conf = 0.0
    clin_score = 0.0
    risk_lvl = "معتدل"
    int_score = 0.0

    if ctx:
        if "ultrasound" in ctx:
            us_pred = ctx["ultrasound"].get("prediction", "غير محدد")
            us_conf = round(ctx["ultrasound"].get("confidence", 0.0) * 100, 1)
        if "clinical" in ctx:
            clin_score = round(float(ctx["clinical"].get("risk_score", 0.0)), 1)
        if "summary" in ctx:
            risk_lvl = ctx["summary"].get("risk_level", "معتدل")
            int_score = round(float(ctx["summary"].get("integrated_score", 0.0)), 1)

    # 1. Questions regarding results or diagnosis
    if any(k in q for k in ("نتيجة", "نتائج", "result", "diagnos", "معنى", "what does", "فحص", "تقييم")):
        parts.append(
            f"بناءً على التقييم المتكامل للنظام: تصنيف فحص السونار يشير إلى حالة '{us_pred}' بدقة {us_conf}%، "
            f"بينما يُقدّر نموذج التحليل السريري نسبة خطورة تبلغ {clin_score}%. "
            f"النتيجة المتكاملة المجمعة هي: مستوى خطورة {risk_lvl} ({int_score}%)."
        )
        if us_pred == "Malignant" or int_score >= 60:
            parts.append("تظهر المؤشرات وجود علامات تستدعي اهتماماً طبياً فورياً من قبل أخصائي الأورام لإجراء فحوصات تأكيدية.")
        elif us_pred == "Benign":
            parts.append("المؤشرات الحالية مطمئنة وترجح طبيعة حميدة للكتلة، ولكن يُنصح دوماً بالمتابعة الدورية.")
        else:
            parts.append("لم تُظهر الأشعة علامات اشتباه مقلقة، ويُنصح بالاستمرار في برامج الكشف المبكر الروتينية.")

    # 2. Questions regarding next steps or biopsy
    elif any(k in q for k in ("خطوة", "خطوات", "next step", "biopsy", "خزعة", "طبيب", "ماذا أفعل", "فحوصات")):
        parts.append("الخطوات الطبية الموصى بها في مثل هذه الحالات تشمل:")
        parts.append("1. استشارة استشاري جراحة أورام الثدي لمراجعة الفحوصات يدوياً.")
        parts.append("2. إجراء فحص الماموجرام الشعاعي (Mammogram) أو الرنين المغناطيسي (MRI) لتقييم أدق.")
        if int_score >= 50 or us_pred == "Malignant":
            parts.append("3. أخذ خزعة نسيجية (Core Needle Biopsy) لتأكيد طبيعة الخلايا مجهرياً.")
        else:
            parts.append("3. تحديد موعد للمتابعة بالموجات فوق الصوتية بعد 3 إلى 6 أشهر للتأكد من ثبات الحجم.")

    # 3. Questions regarding features or risk factors
    elif any(k in q for k in ("عوامل", "أسباب", "ميزات", "factor", "feature", "حجم", "كثافة", "أهمية")):
        parts.append(
            "يعتمد التقييم السريري بشكل أساسي على الخصائص المورفولوجية للخلية: حجم الورم (Radius)، المحيط (Perimeter)، "
            "والكثافة السطحية (Texture). الزيادة في حجم ومحيط الورم مع عدم انتظام الحواف تُعد من أبرز المؤشرات التي ترفع درجة الاشتباه."
        )

    # 4. General / Greeting / Other
    else:
        parts.append(
            f"مرحباً بك. أنا المساعد الطبي الذكي للنظام. بناءً على بيانات الحالة الحالية: مستوى الخطورة المتكامل هو {risk_lvl} ({int_score}%) "
            f"ونتائج فحص الموجات فوق الصوتية هي '{us_pred}'. "
            f"يمكنك سؤالي عن تفسير النتائج، الخطوات الطبية القادمة، أو العوامل المؤثرة في التشخيص."
        )

    parts.append(f"\n\n*(تنويه طبي: {DISCLAIMER})*")
    return "\n\n".join(parts)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1. Check Gemini API
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            prompt_text = f"You are an empathetic, clinical AI decision-support assistant for breast cancer. Answer in the same language as the user query (Arabic or English). Keep answers concise and grounded in the patient context.\n\nContext:\n{json.dumps(req.context, default=str)}\n\nQuestion: {req.question}"
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                json={"contents": [{"parts": [{"text": prompt_text}]}]},
                timeout=30,
            )
            if resp.status_code == 200:
                candidates = resp.json().get("candidates", [])
                if candidates:
                    answer = candidates[0]["content"]["parts"][0]["text"]
                    return ChatResponse(answer=answer, engine="Gemini-1.5-Flash (Transformer)")
        except Exception as exc:
            log.warning("Gemini chat failed, falling back: %s", exc)

    # 2. Check Anthropic API
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            system = (
                "You are a careful, empathetic medical information assistant embedded in a breast-cancer "
                "screening decision-support tool. Answer in the user's language. Keep concise (under 200 words)."
            )
            if req.context:
                system += "\n\nContext:\n" + json.dumps(req.context, default=str)[:4000]
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={
                    "model": os.getenv("LLM_MODEL", "claude-3-haiku-20240307"),
                    "max_tokens": 500,
                    "system": system,
                    "messages": [{"role": "user", "content": req.question}],
                },
                timeout=30,
            )
            if resp.status_code == 200:
                text = "".join(b.get("text", "") for b in resp.json().get("content", []) if b.get("type") == "text")
                if text:
                    return ChatResponse(answer=text, engine="Anthropic Claude (Transformer)")
        except Exception as exc:
            log.warning("Anthropic chat failed: %s", exc)

    # 3. Contextual Clinical Engine
    return ChatResponse(
        answer=_clinical_intelligence_answer(req.question, req.context),
        engine="Clinical Intelligence Engine (Context-Aware)",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)