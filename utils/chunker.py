from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-base")

def chunk_text(text, max_tokens=300):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        tokenized = tokenizer(" ".join(current_chunk), return_tensors="pt")
        if tokenized.input_ids.shape[1] > max_tokens:
            if len(current_chunk) > 1:
                chunks.append(" ".join(current_chunk[:-1]))
                current_chunk = [word]
            else:
                chunks.append(word)
                current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks