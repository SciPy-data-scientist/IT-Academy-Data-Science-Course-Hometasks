FROM python:3.12

ENV PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry install --no-interaction --no-root

COPY . .

CMD ["poetry", "run", "python", "src/predictor/main.py"]