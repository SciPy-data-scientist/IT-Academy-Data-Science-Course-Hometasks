import streamlit as st
import requests
from typing import Dict, Any
import os
import time


class QuestionAnsweringApp:
    """
    Streamlit app for processing Russian questions through a FastAPI backend.

    Provides an interface to:
    1. Submit questions in Russian
    2. Get processed answers with keyword extraction and summarization
    """

    def __init__(self, api_base_url: str = "http://0.0.0.0:8000"):
        """
        Initialize the QuestionAnsweringApp.

        Args:
            api_base_url: Base URL of the FastAPI backend (default: localhost)
        """
        self.timeout = 120
        self.api_base_url = api_base_url
        st.set_page_config(
            page_title="Russian Question Answering System",
            layout="centered"
        )

    def run(self) -> None:
        """Run the main app interface."""
        st.title("❓ HealthQuery")
        st.markdown(
            "Данное приложение позволяет задать на русском языке вопрос, " \
            "касающийся здоровья, и получить подробный ответ и ссылки " \
            "на использованные источники"
        )
        self.render_question_form()

    def render_question_form(self) -> None:
        """Render the question input form."""
        st.header("📝 Вопрос:")
        question = st.text_area(
            "Введите свой вопрос на русском языке:",
            placeholder="Можно ли детям использовать зубную пасту со фтором?",
            height=150
        )

        if st.button("Получить ответ"):
            if not question.strip():
                st.warning("Пожалуйста, введите вопрос")
                return

            with st.spinner("Обработка вопроса (это может занять 1-2 минуты)..."):
                try:
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.5)
                        progress_bar.progress(i + 1)
                    response = self._send_question(question)
                    self._handle_response(response)
                except Exception as e:
                    st.error(f"Ошибка обработки запроса: {str(e)}")

    def _send_question(self, question: str) -> requests.Response:
        """
        Send question to the FastAPI backend.

        Args:
            question: Russian text question to process

        Returns:
            Response from the API endpoint
        """
        data = {"question": question}
        return requests.post(
            f"{self.api_base_url}/",
            json=data,
            timeout=120
        )

    def _handle_response(self, response: requests.Response) -> None:
        """
        Process and display the API response.

        Args:
            response: HTTP response from the FastAPI backend
        """
        if response.status_code == 422:
            st.error("Validation error in your question:")
            st.json(response.json())
        elif response.status_code != 200:
            st.error(f"API Error: {response.status_code}")
            st.json(response.json())
        else:
            result = response.json()
            st.success("✅ Ответ успешно сгенерирован")
            st.subheader("Ответ:")
            st.markdown(result["prediction"])

            if "keywords" in result:
                with st.expander("🔍 Extracted Keywords"):
                    st.write(result["keywords"])

            if "sources" in result:
                with st.expander("📚 Reference Sources"):
                    for source in result["sources"]:
                        st.markdown(f"- [{source['title']}]({source['url']})")


if __name__ == "__main__":
    api_url = os.getenv("API_BASE_URL", "http://0.0.0.0:8000")
    app = QuestionAnsweringApp(api_base_url=api_url)
    app.run()