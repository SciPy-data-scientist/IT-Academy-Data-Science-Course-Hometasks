from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
import torch

from predictor.api.json_predictor import json_router

device = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    local_path = Path(__file__).resolve().parent.parent.parent.parent
    keyword_model_path = local_path / "models" / "keyword_model"
    app.state.keyword_model = BartForConditionalGeneration.from_pretrained(
        keyword_model_path
        ).to(device)
    app.state.keyword_tokenizer = BartTokenizer.from_pretrained(
        keyword_model_path
        )

    app.state.translation_model = pipeline(
        "translation_ru_to_en",
        model="Helsinki-NLP/opus-mt-ru-en",
        device=0 if device == "cuda" else -1
        )

    yield

app = FastAPI(lifespan=lifespan)

app.include_router(json_router)
