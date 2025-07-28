# Compassion Text Analyzer

This project contains tools to analyze emotional and compassionate content in text files.

## Scripts

- `txtAndDocxSentimentAnalysisTool.py`: Analyzes `.docx` documents and `.txt` files.
- `VisualizeTheSentimentAnalysis.py`: Visualizes results from CSV output.

## Usage

1. Put your `.docx` and `.txt` files in `Texts Used/` folder.
2. Run analysis scripts:

```bash
python txtAndDocxSentimentAnalysisTool.py

## visualize with. NOTE: Not all emotions are generated every time, you may have to edit the columns_of_interest list
python VisualizeTheSentimentAnalysis.py
