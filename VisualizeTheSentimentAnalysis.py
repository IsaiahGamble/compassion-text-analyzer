import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

CSV_TO_VISUALIZE = "text_analysis_output.csv"

# Load CSV
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / CSV_TO_VISUALIZE
df = pd.read_csv(csv_path)

# Columns of interest
columns_of_interest = [
    "vader_pos",
    "vader_neg",
    "vader_neu",
    "vader_compound",
    "affection",
    "nurture",
    "love",
    "help",
    "pain",
    "emotional",
    "family",
    "health",
    "care",
    "emotion_joy",
    "emotion_sadness",
    "emotion_neutral",
    "emotion_disgust",
    "emotion_fear",
    "emotion_surprise",
    "emotion_anger"

]
# others
# "vader_pos", "vader_neg", "vader_neu", "emotion_neutral", "emotion_anger"


# Summary statistics dataframe
summary_stats = df[columns_of_interest].agg(["mean", "max", "min", "median", "std"]).T
print(summary_stats)

means = summary_stats["mean"]
stds = summary_stats["std"]
labels = summary_stats.index

x = np.arange(len(labels))

medians = df[columns_of_interest].median()
q1 = df[columns_of_interest].quantile(0.25)
q3 = df[columns_of_interest].quantile(0.75)
iqr = q3 - q1

x = np.arange(len(columns_of_interest))

plt.figure(figsize=(12, 7))
plt.bar(x, medians, yerr=iqr, capsize=5, color="skyblue")
plt.xticks(x, columns_of_interest, rotation=45, ha="right")
plt.ylabel("Score")
plt.title("Median Scores with IQR as Error Bars")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(data=df[columns_of_interest], orient="h")
plt.title("Inter-quartile Range with Highest & Lowest Non-Outliers as Error Bars, Marking the Median and Outliers")
plt.xlabel("Score")
plt.tight_layout()
plt.show()


