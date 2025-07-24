from fastapi import APIRouter, Request
from predictor.schemas.api_schemas import PredictionResponse, Question
from openai import OpenAI
from pydantic import ValidationError
import requests
import httpx

API_key = "YOUR_API_KEY"
CX_key = "YOUR_CX_KEY"
DEEP_SEEK_API_key = "YOUR_DEEP_SEEK_API_KEY"


json_router = APIRouter()


@json_router.post("/", response_model=PredictionResponse)
async def predict_json(request: Request, data: Question):

    async with httpx.AsyncClient(timeout=120.0) as client:

            # 1. Translate the question
        try:
            translator = request.app.state.translation_model
            translated = translator(data.question)[0]["translation_text"]

            # 2. Extract keywords
            keyword_model = request.app.state.keyword_model
            tokenizer = request.app.state.keyword_tokenizer

            inputs = tokenizer(
                translated,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(keyword_model.device)
            output = keyword_model.generate(
                inputs["input_ids"],
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
            keywords = tokenizer.decode(output[0], skip_special_tokens=True)

            # 3. Google search with keywords
            search_terms = keywords.replace(", ", "+AND+")
            api_key = API_key
            cx = CX_key
            url = (
                "https://www.googleapis.com/customsearch/v1?"
                f"q={search_terms}&key={api_key}&cx={cx}"
            )

            google_response = requests.get(url)
            if google_response.status_code != 200:
                return PredictionResponse(
                    prediction="Error while searching Google."
                )

            items = google_response.json().get("items", [])
            top_3_links = [item['link'] for item in items[:3]]

            if not top_3_links:
                return PredictionResponse(
                    prediction="No relevant sources found for the query."
                )

            # 4. Call DeepSeek for summarization
            deep_seek_api_key = DEEP_SEEK_API_key
            client = OpenAI(
                api_key=deep_seek_api_key,
                base_url="https://api.deepseek.com/v1"
            )

            chat_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — строгий эксперт по анализу и структурированию информации. "
                            "Тебе будут даны три статьи. Твоя задача: "
                            "1. Проанализировать каждую статью и определить, релевантна ли она запросу. "
                            "2. Если статья релевантна — изложить её ключевые идеи (только факты, без добавлений). "
                            "3. Если не релевантна — пропустить её. "
                            "4. Ранжировать обзор по релевантности (начиная с самой полезной статьи). "
                            "Твой ответ должен быть на 100% основан на предоставленных статьях. "
                            "Если информация не найдена в статьях, ответ должен быть пустым для этой части. "
                            "Не пытайся 'угадать' или 'дополнить' факты — это критическая ошибка. "
                            "Формат вывода: "
                            "- В тексте указывай ссылки в формате [1], [2]. "
                            "- В конце приведи полный список источников с нумерацией. "
                            "- Сохраняй нейтральный тон и объективность. "
                            "- Объём: 400-500 слов."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Запрос: {data.question}\n\n"
                            f"Ссылки:\n{top_3_links}\n\n"
                            "Проанализируй статьи. Если информация релевантна запросу, изложи её основные идеи. "
                            "Важно: используй ТОЛЬКО информацию из предоставленных статей. "
                            "Если в статье нет данных по запросу — не пиши ничего. "
                            "Не добавляй обобщений, предположений или информации из твоих знаний. Объём ответа: 400-500 слов. "
                            "При ссылке на статью указывай ее как [1], [2], [3]. "
                            "В конце приведи список источников в формате: "
                            "1. [ссылка] 2. [ссылка] 3. [ссылка]"
                        )
                    }
                ],
                temperature=0,
                stream=False
            )

            answer_summary = chat_response.choices[0].message.content

            return PredictionResponse(prediction=answer_summary)

        except ValidationError as e:
            print(f"Validation error: {e.errors()}")
            return PredictionResponse(
                prediction=f"Data validation error: {str(e)}"
            )
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return PredictionResponse(
                prediction=f"An error occurred: {str(e)}"
            )
