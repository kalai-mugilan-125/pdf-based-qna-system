from utils.document_loader import load_text
from utils.query_parser import answer_query_from_text

def run_query_mode(file_path, user_query):
    try:
        text = load_text(file_path)
        answer = answer_query_from_text(user_query, text)
        return {
            "query": user_query,
            "answer": answer,
            "file": file_path
        }
    except Exception as e:
        return {
            "query": user_query,
            "answer": f"Error processing query: {str(e)}",
            "file": file_path
        }