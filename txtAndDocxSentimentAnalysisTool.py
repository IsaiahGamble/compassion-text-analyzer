import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
from docx import Document

# Folder containing files to analyze (both .docx and/or .txt)
FOLDER_NAME_WHERE_TEXT_TO_BE_ANALYZED_IS = "Texts Used"
CSV_OUTPUT_FILENAME = "text_analysis_output.csv"

vader = SentimentIntensityAnalyzer()
lexicon = Empath()
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

# Categories for compassion proxies
empath_categories = ["affection", "nurture", "love", "help", "pain", "emotional", "family", "health", "care"]

results = []

def read_docx(path: Path) -> str:
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_file(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return read_docx(path)
    elif path.suffix.lower() == ".txt":
        return read_txt(path)
    else:
        print(f"Skipping unsupported file type: {path.name}")
        return None

base_dir = Path(__file__).resolve().parent
folder_path = base_dir / FOLDER_NAME_WHERE_TEXT_TO_BE_ANALYZED_IS
files = [f for f in folder_path.iterdir() if f.is_file()]
num_files = len(files)

for file_path in tqdm(files):
    try:
        text = read_file(file_path)
        if text is None:
            continue
        filename = file_path.name

        # VADER Sentiment
        vader_scores = vader.polarity_scores(text)

        # Empath categories
        empath_scores = lexicon.analyze(text, categories=empath_categories, normalize=True)

        # Emotion classification (HuggingFace)
        emotion_scores = emotion_model(text[:512])

        top_emotions = {f"emotion_{e['label'].lower()}": e['score'] for e in emotion_scores[0]}

        # Combine all into one row
        row = {
            "filename": filename,
            "vader_pos": vader_scores["pos"],
            "vader_neg": vader_scores["neg"],
            "vader_neu": vader_scores["neu"],
            "vader_compound": vader_scores["compound"],
        }
        row.update(empath_scores)
        row.update(top_emotions)

        results.append(row)
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

os.makedirs(base_dir, exist_ok=True)
df = pd.DataFrame(results)
output_path = os.path.join(base_dir, CSV_OUTPUT_FILENAME)
df.to_csv(output_path, index=False)

print("File saved to:", output_path)
