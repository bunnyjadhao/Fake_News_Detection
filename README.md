# Fake News Detection System

A Python-based machine learning project that detects fake news from textual statements using NLP techniques, feature extraction, and word embeddings.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
Fake news is a major challenge in today’s information-driven world.  
This project implements a **Fake News Detection system** using NLP and machine learning models. It analyzes text statements and classifies them as **real** or **fake**.  

Key techniques used:
- Text preprocessing with NLTK
- Feature extraction using **CountVectorizer**
- Word embeddings using **GloVe**
- Machine learning models for classification  

---

## Features
- Clean and preprocess text data
- Extract features using bag-of-words and embeddings
- Train and test ML models for fake news detection
- Visualize dataset distribution using Seaborn
- Easily extendable for new datasets

---

## Tech Stack
- **Programming Language:** Python 3.x  
- **Libraries:**  
  - `pandas` – data handling  
  - `numpy` – numerical computations  
  - `scikit-learn` – machine learning models  
  - `nltk` – natural language processing  
  - `gensim` – word embeddings  
  - `matplotlib` & `seaborn` – visualizations  

---

## Dataset
- Public dataset containing statements and labels (Real/Fake)  
- Example: 10,240 statements with corresponding labels  
- Dataset can be loaded in `DataPrep.py`

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Fake_News_Detection.git
cd Fake_News_Detection-master



pip install pandas numpy scikit-learn nltk gensim matplotlib seaborn


import nltk

nltk.download('treebank')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')



python FeatureSelection.py


Roshan Jadhao
Email: roshanjadhao267@gmail.com

LinkedIn: https://linkedin.com/in/roshanjadhao
GitHub

Check out the repository: https://github.com/bunnyjadhao/Fake_News_Detection
