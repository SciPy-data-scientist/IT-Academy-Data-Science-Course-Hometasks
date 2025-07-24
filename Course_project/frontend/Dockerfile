FROM python:3.12

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry install --no-interaction --no-root

RUN poetry config virtualenvs.create false

COPY . .

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]