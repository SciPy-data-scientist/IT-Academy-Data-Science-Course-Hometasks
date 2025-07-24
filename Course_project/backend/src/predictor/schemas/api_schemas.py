from pydantic import BaseModel, Field
from pydantic.types import StringConstraints
from typing import Annotated


RussianText = Annotated[
    str,
    StringConstraints(pattern=r'^[а-яА-ЯёЁa-zA-Z0-9\s\.,!?()-]+$')
]


class Question(BaseModel):
    question: RussianText = Field(
        ...,
        description="Вопрос должен быть текстовой строкой на русском языке,"
        "может содержать цифры и пунктуацию",
        examples=["Можно ли детям использовать зубную пасту со фтором?"]
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(
        ...,
        description="Ответ должен быть текстовой строкой на русском языке",
        examples=["Да, детям нужно использовать зубную пасту со фтором."]
    )
