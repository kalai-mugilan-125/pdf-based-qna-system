import os
import json
from utils.document_loader import load_text
from utils.chunker import chunk_text
from utils.ner_helper import find_answer_candidate
from utils.qa_qg_interface import generate_and_answer

def run_qna_pipeline(file_path, output_dir, max_tokens=300):
    try:
        raw_text = load_text(file_path)
        chunks = chunk_text(raw_text, max_tokens=max_tokens)
        results = []

        for i, chunk in enumerate(chunks):
            answer = find_answer_candidate(chunk)
            if not answer:
                continue
                
            question, predicted_answer = generate_and_answer(chunk, answer)
            if question and predicted_answer:
                results.append({
                    "chunk_id": i,
                    "chunk": chunk,
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "original_answer": answer
                })

        file_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"Processed {len(results)} Q&A pairs and saved to {output_path}")
        return results
        
    except Exception as e:
        print(f"Error in QNA pipeline: {str(e)}")
        return []