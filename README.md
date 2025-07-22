# 🧠 English to French Translator using NLP and RNN (Seq2Seq)

This project implements a sequence-to-sequence (Seq2Seq) deep learning model using LSTM layers to translate English sentences into French. It uses standard NLP preprocessing, tokenization, and embedding techniques to train a neural translation model.

---

## 📌 Overview

- **Model Type:** Encoder-Decoder with LSTM
- **Goal:** Translate English text to French
- **Techniques Used:** NLP, Tokenization, Padding, Embedding, LSTM
- **Libraries:** TensorFlow, Keras, NLTK, Gensim, WordCloud, Plotly, Seaborn

---

## 📂 Dataset

- Two text files with aligned English and French sentences.
- Files used:
  - `small_vocab_en.csv`
  - `small_vocab_fr.csv`
- Loaded from Google Drive via Colab.

---

## 🧹 Preprocessing Steps

1. Removed punctuation
2. Tokenized using NLTK
3. Counted word frequency (visualized with WordCloud & Plotly)
4. Found max sentence length
5. Tokenized and padded both English and French datasets
6. Vocabulary size determined

---

## 🧠 Model Architecture

The model is a simple **Encoder-Decoder RNN** using LSTM layers.

```python
# Encoder
Input → Embedding → LSTM → RepeatVector

# Decoder
RepeatVector → LSTM (return_sequences=True) → TimeDistributed(Dense)
# requirments
tensorflow
keras
numpy
pandas
matplotlib
seaborn
nltk
gensim
plotly
wordcloud
scikit-learn
