from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model = "emilyalsentzer/Bio_ClinicalBERT"
nlp = pipeline("ner", model=model, tokenizer=model, aggregation_strategy="simple")

text = open("sample_note.txt", "r", encoding="utf-8").read()
entities = nlp(text)

symptoms = [e["word"] for e in entities if e["entity_group"] == "PROBLEM"]
medications = [e["word"] for e in entities if e["entity_group"] == "TREATMENT"]

diagnosis = "diabetes" in text.lower() or "hypertension" in text.lower()

print("Extracted Symptoms:", symptoms)
print("Medications:", medications)
print("Possible Diagnosis:", diagnosis)
