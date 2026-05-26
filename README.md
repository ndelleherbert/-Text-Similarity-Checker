# Text Similarity Checker — Project Reader Guide

## Overview

The Text Similarity Checker is a machine learning-powered web application built with Streamlit that compares two pieces of text and measures their semantic similarity using sentence embeddings and cosine similarity.

Unlike traditional keyword matching systems, this application understands contextual meaning between sentences using transformer-based NLP models.

---

# Project Goal

The goal of this project is to:

- Compare semantic meaning between texts
- Demonstrate NLP embedding techniques
- Provide an interactive UI for users
- Teach cosine similarity concepts in AI applications

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| Sentence Transformers | Text embedding generation |
| Scikit-learn | Cosine similarity calculation |
| Torch | Deep learning backend |

---

# How the Project Works

The application follows these steps:

1. User enters two sentences
2. Text is converted into embeddings
3. Cosine similarity is calculated
4. Similarity score is displayed
5. Application classifies the similarity level

---

# Semantic Embeddings

The model converts text into numerical vectors called embeddings.

Example:

```text
"My pet name is cherry"

becomes:

[0.231, -0.441, 0.992, ...]

These vectors capture semantic meaning rather than exact words.


---

Cosine Similarity

Cosine similarity measures how close two vectors are.

The formula used is:

Similarity = (A · B) / (||A|| × ||B||)

Where:

A = Vector 1

B = Vector 2

· = Dot product

|| || = Vector magnitude



---

Similarity Interpretation

Score Range	Meaning

0.85 – 1.00	Very similar
0.65 – 0.84	Somewhat similar
Below 0.65	Not very similar



---

File Explanation

1. app.py

Main Streamlit application.

Responsibilities:

User interface

Text input handling

Button interaction

Displaying similarity results



---

2. cosine_similarity.py

Handles NLP processing.

Responsibilities:

Loading transformer model

Generating embeddings

Computing cosine similarity



---

3. requirements.txt

Contains all project dependencies.

Used for:

pip install -r requirements.txt


---

4. README.md

Main public documentation file.

Contains:

Installation guide

Usage instructions

Features

Examples



---

Machine Learning Model

The project uses:

all-MiniLM-L6-v2

This lightweight transformer model is optimized for:

Semantic similarity

Fast inference

Low memory usage



---

Example Workflow

Input:

Text 1: My pet name is cherry
Text 2: My dog is called cherry

Processing:

1. Generate embeddings


2. Compare vector similarity


3. Return score



Output:

Cosine Similarity: 0.89
✅ Very similar


---

Running the Project

Step 1 — Install Dependencies

pip install -r requirements.txt

Step 2 — Start Streamlit Server

streamlit run app.py


---

Local Server

After running the application, Streamlit provides a local URL:

http://localhost:8501

Open it in your browser.


---

Possible Improvements

Future enhancements may include:

Multiple language support

PDF similarity checking

Voice input

Text summarization

Similarity heatmaps

AI-powered plagiarism detection



---

Learning Concepts Covered

This project helps understand:

Natural Language Processing (NLP)

Sentence embeddings

Vector similarity

Cosine similarity

Transformer models

Streamlit web apps



---

Common Errors

ModuleNotFoundError

Solution:

pip install -r requirements.txt


---

Torch Installation Issues

Upgrade pip:

python -m pip install --upgrade pip


---

Streamlit Not Found

Install Streamlit:

pip install streamlit


---

Recommended Python Version

Python 3.10+


---

Author

Developed as an NLP semantic similarity project using Python and Machine Learning.


---

License

MIT License