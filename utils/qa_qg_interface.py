import torch
import os
from dotenv import load_dotenv
from model.hybrid_qg_qa import HybridQGQAModel
from transformers import T5Tokenizer

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = HybridQGQAModel().to(device)
model.eval()

def generate_and_answer(chunk, answer_text):
    if answer_text not in chunk:
        return None, None

    try:
        highlighted = chunk.replace(answer_text, f"<hl> {answer_text} <hl>")
        qg_input = tokenizer.encode("generate question: " + highlighted, return_tensors="pt", truncation=True, max_length=512).to(device)
        decoder_input = tokenizer.encode("<pad>", return_tensors="pt").to(device)
        qa_input = tokenizer.encode(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)

        decoder_input_expanded = decoder_input.expand(1, qg_input.shape[1], -1)

        with torch.no_grad():
            qg_logits, start_logits, end_logits = model(
                input_ids=qg_input,
                attention_mask=qg_input.ne(tokenizer.pad_token_id),
                decoder_input_ids=decoder_input_expanded,
                qa_input_ids=qa_input,
                qa_attention_mask=qa_input.ne(tokenizer.pad_token_id)
            )

        generated_ids = qg_logits.argmax(-1)
        question = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        start_index = start_logits.argmax(-1).item()
        end_index = end_logits.argmax(-1).item()
        
        if start_index <= end_index and end_index < qa_input.shape[1]:
            tokens = qa_input[0][start_index:end_index + 1]
            predicted_answer = tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            predicted_answer = answer_text

        return question, predicted_answer
    
    except Exception as e:
        return f"Error generating question: {str(e)}", answer_text