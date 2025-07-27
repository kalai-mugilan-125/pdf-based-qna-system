import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text, label_filter=None):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if label_filter:
        entities = [ent for ent in entities if ent[1] in label_filter]
    return entities

def find_answer_candidate(text, preferred_labels=["ORG", "DATE", "PERSON", "GPE"]):
    entities = extract_named_entities(text, label_filter=preferred_labels)
    if entities:
        return entities[0][0]
    
    words = text.split()
    if len(words) > 10:
        return " ".join(words[:5])
    return None