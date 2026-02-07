"""
FAQ + Pinecone + LLM

Скрипт містить:
1) Архітектуру RAG AI Agent
2) Цикл взаємодії (multi-turn conversation)
3) Пошук в Pinecone + генерацію через LLM
4) Обробку помилок та управління виходом
"""

from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import sys

# =========================
# КОНФІГУРАЦІЯ
# =========================
PINECONE_INDEX = "faq-index"
TOP_K = 5
THRESHOLD = 0.40

# Використання запасної моделі
LLM_MODEL_PRIMARY = "gpt-5-mini"
LLM_MODEL_FALLBACK = "gpt-4o-mini"

# =========================
# ІНІЦІАЛІЗАЦІЯ КЛІЄНТІВ
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

try:
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("API-ключі не знайдені у змінних середовища або .env")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

    index = pinecone_client.Index(PINECONE_INDEX)

except Exception as e:
    print(f"❌ Помилка при ініціалізації клієнтів: {str(e)}")
    print("Переконайтеся, що API ключі встановлені правильно (.env або змінні середовища).")
    sys.exit(1)

# =========================
# ФУНКЦІЇ
# =========================
def generate_embedding(text: str):
    """Генеруємо embedding для тексту"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Помилка при генерації embedding: {str(e)}")
        return None


def search_faq(question: str, top_k: int = TOP_K, threshold: float = THRESHOLD) -> list:
    """Повертає ВСІ релевантні FAQ зі score >= threshold"""
    embedding = generate_embedding(question)
    if embedding is None:
        return []

    try:
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        print(f"❌ Помилка Pinecone: {e}")
        return []

    matches = results.get("matches", [])
    if not matches:
        print("❌ Результатів немає")
        return []

    relevant_results = []

    # (можеш прибрати цей лог, але на демо інколи корисно)
    print("\n📊 Результати Pinecone:")
    for match in matches:
        score = match.get("score", 0)
        metadata = match.get("metadata", {}) or {}

        print(f" • score={score:.4f} | {metadata.get('question', 'N/A')}")

        if score >= threshold:
            relevant_results.append({
                "score": score,
                "question": metadata.get("question", ""),
                "answer": metadata.get("answer", "")
            })

    return relevant_results


def build_context(relevant_results: list, max_items: int = 3) -> str:
    """
    Формуємо контекст для LLM з кількох релевантних FAQ,
    щоб LLM краще “розумів” і не відповідав 1:1.
    """
    chunks = []
    for item in relevant_results[:max_items]:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        s = item.get("score", 0)
        chunks.append(f"[score={s:.3f}] Q: {q}\nA: {a}")
    return "\n\n".join(chunks)


def generate_response_with_llm(question: str, context: str) -> str:
    """Генеруємо відповідь через LLM"""
    system_prompt = (
        "Ти є дружелюбним асистентом FAQ-бота інтернет-магазину харчових товарів в Україні. "
        "Відповідай українською, коротко (2–4 речення), по суті та дружелюбно. "
        "ВАЖЛИВО: не копіюй текст з контексту дослівно — перефразуй людською мовою. "
        "Якщо інформації в контексті недостатньо або питання не про магазин — скажи це чесно."
    )

    user_prompt = f"""На основі цієї інформації (контекст з бази знань):
{context}

Дай відповідь на запитання користувача:
"{question}"

Відповідь (одним текстом):
"""

    last_error = None
    for model in (LLM_MODEL_PRIMARY, LLM_MODEL_FALLBACK):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=220
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = str(e)

    return f"Помилка при генерації відповіді: {last_error}"


def process_question(question: str) -> str:
    """
    Основна логіка:
    1) Пошук у Pinecone
    2) Якщо є релевантні — формуємо контекст
    3) Генеруємо відповідь через LLM або кажемо “не знаю”
    """
    relevant = search_faq(question)

    if relevant:
        context = build_context(relevant, max_items=3)
        return generate_response_with_llm(question, context)
    else:
        return "Вибачте, я не знаю відповіді на це запитання."


def print_welcome():
    print("🤖 ЛАСКАВО ПРОСИМО ДО FAQ BOT!")
    print("Це AI-асистент, який відповідає на питання про наш магазин.")
    print("Можна питати про товари, замовлення, доставку, оплату, сертифікати тощо.")
    print("Для виходу введіть: exit, quit або натисніть Ctrl+C\n")


def get_user_input() -> str:
    """Отримуємо введення від користувача"""
    try:
        return input("Вас цікавить: ").strip()
    except EOFError:
        return "exit"
    except KeyboardInterrupt:
        return "exit"


def main_loop():
    """Основний цикл програми"""
    print_welcome()

    while True:
        question = get_user_input()

        if question.lower() in ["exit", "quit"]:
            print("\nДякуємо за використання FAQ Bot! 👋")
            break

        if question == "":
            print("💭 Напишіть, будь ласка, запитання.\n")
            continue

        print()
        response = process_question(question)
        print(f"💭 {response}\n")


# =========================
# ЗАПУСК
# =========================
try:
    main_loop()
except KeyboardInterrupt:
    print("\n\nДякуємо за використання FAQ Bot! 👋")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Неочікувана помилка: {str(e)}")
    sys.exit(1)
