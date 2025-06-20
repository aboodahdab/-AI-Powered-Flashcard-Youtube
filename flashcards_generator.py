# Flashcard Generator with Chunk Context + CSV Export
import traceback as tb
import pandas as pd
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "ddt9WT0iXH4"
transcript = YouTubeTranscriptApi.get_transcript(video_id)
full_text = " ".join([entry['text'] for entry in transcript])


def split_text(text, max_words=225):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks


def get_qa_pipeline():
    # i used this ai module deepset/roberta-base-squad because it's light i know it's bad and it's answers makes no sense
    # change the module if you want to ,but remeber that i used this cause it's light and small (only 500mb)
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/roberta-base-squad2")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)


def generate_flashcards(chunks):
    qa = get_qa_pipeline()
    questions = [
        "What is this part talking about?",
        "What are the key ideas here?",
        "What examples are mentioned?",
        "What can we learn from this?"
    ]
    flashcards = []
    for i, chunk in enumerate(chunks):
        print(f"\n🧩 Chunk {i+1}: {chunk[:500]}...")
        print("-" * 60)
        for question in questions:
            try:
                result = qa(question=question, context=chunk)
                answer = result["answer"]
                flashcards.append({
                    "chunk": chunk,
                    "question": question,
                    "answer": answer
                })
                print(f"❓ {question}")
                print(f"✅ {answer}")
                print("-" * 40)
            except Exception as e:
                print(f"⚠️ Error in chunk {i+1} for question '{question}':", e)
                tb.print_exc()
    return flashcards


def save_to_csv(flashcards, filename="flashcards.csv"):
    df = pd.DataFrame(flashcards)
    df.to_csv(filename, index=False, encoding="utf-8")


# def main:


save_to_csv(generate_flashcards(split_text(full_text)))
