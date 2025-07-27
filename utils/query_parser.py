import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_query_from_text(query, context):
    try:
        if len(context) > 10000:
            context = context[:10000]
        
        result = qa_model(question=query, context=context)
        return result["answer"]
    except Exception as e:
        return f"Error processing query: {str(e)}"