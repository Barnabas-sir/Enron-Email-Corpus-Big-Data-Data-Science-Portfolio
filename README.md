# Enron-Email-Corpus-Big-Data-Data-Science-Portfolio
This portfolio analyses the Enron email corpus — one of the most significant real-world datasets in data science history. Released during the 2001 fraud investigation, it contains the internal communications of Enron employees in the years surrounding one of the largest corporate scandals ever recorded.


# Enron Email Corpus — Big Data & Data Science Portfolio

> **7 end-to-end analytical projects** on 517,401 real corporate emails using PySpark on Databricks

---

## 📌 Overview

This portfolio analyses the **Enron email corpus** — one of the most significant real-world datasets in data science history. Released during the 2001 fraud investigation, it contains the internal communications of Enron employees in the years surrounding one of the largest corporate scandals ever recorded.

| Property | Value |
|---|---|
| Dataset | Enron Email Corpus |
| Size | 1.33 GB / 517,401 emails |
| Date Range | 1999 – 2002 |
| Platform | Databricks (PySpark) |
| Projects | 7 |

---

## 🛠️ Tech Stack

`PySpark` · `Databricks` · `Spark MLlib` · `TextBlob` · `Matplotlib` · `Seaborn` · `Python`

---

## 📁 Project Structure

```
enron-portfolio/
│
├── 01_eda_volume_analysis.ipynb
├── 02_network_analysis.ipynb
├── 03_sentiment_analysis.ipynb
├── 04_tfidf_keyword_extraction.ipynb
├── 05_spam_classifier.ipynb
├── 06_author_attribution.ipynb
└── 07_anomaly_detection.ipynb
```

---

## 🗂️ Projects

### Project 1 — EDA & Email Volume Analysis
**Techniques:** PySpark aggregations, Matplotlib, Seaborn

Explored the full shape of the dataset — email volume by hour, day of week, year, and top senders.

**Key Finding:**
- 2001 had the **highest email volume** in Enron's history despite the company collapsing
- Peak sending hours: **2pm–4pm**; peak day: **Tuesday/Wednesday**
- **~20,000 weekend emails** reveal the high-pressure culture Enron was known for
- **Kay Mann** (lead attorney) sent 16,735 emails — most active sender in the corpus

---

### Project 2 — Communication Network Analysis
**Techniques:** Edge list construction, weighted graph aggregation, pie charts

Mapped 760,214 sender-recipient pairs into a weighted communication network.

**Key Finding:**
- **Kenneth Lay** (CEO) was contacted by more unique senders than anyone else (1,295)
- **70.2%** of all communication was internal — only 11.3% went external
- **Vince Kaminski** forwarded 4,316 emails to a personal AOL account — flagged by investigators
- **Pete Davis** emailed himself 9,141 times — automated report generation

---

### Project 3 — Sentiment Analysis
**Techniques:** TextBlob, Python UDF, stopword removal, broadcasting, time series

Scored the emotional polarity of every email body (-1 negative → +1 positive) after removing 127 custom stopwords.

**Key Finding:**
- Sentiment **did not drop** during the scandal — it slightly rose (0.08 → 0.12)
- Suggests employees used **deliberately neutral and positive language** anticipating scrutiny
- Highlights a key limitation of lexicon-based sentiment analysis on corporate communications
- **Most negative sender:** joanne.rozycki@enron.com (-0.46 avg polarity)

---

### Project 4 — TF-IDF Keyword Extraction
**Techniques:** Spark MLlib (Tokenizer, StopWordsRemover, HashingTF, IDF), word frequency ranking, Window functions

Extracted the most important vocabulary per year and tracked crisis-related word growth over time.

**Key Finding:**

| Word | 1999 Count | 2001 Count | Growth |
|---|---|---|---|
| `investigation` | 52 | 5,937 | **+11,317%** |
| `original` | 657 | 125,193 | **+18,955%** |
| `audit` | 85 | 3,614 | **+4,152%** |
| `power` | 2,408 | 219,202 | **+9,003%** |

The word `"original"` growing 18,955% reflects mass email forwarding for **legal discovery purposes**.

---

### Project 5 — Spam Detection Classifier
**Techniques:** Imbalanced data handling (undersampling), Spark MLlib Pipeline, Logistic Regression, Naive Bayes, AUC-ROC evaluation

Trained two classifiers on 52,295 balanced emails (50% spam / 50% legitimate).

**Results:**

| Metric | Logistic Regression | Naive Bayes |
|---|---|---|
| Accuracy | **93.5%** | 83.6% |
| Precision | **0.935** | 0.848 |
| Recall | **0.935** | 0.836 |
| AUC-ROC | **0.972** | 0.860* |

*Naive Bayes AUC required correction via probability extraction UDF due to log-probability output format.

**Confusion Matrix (Logistic Regression):**
```
                 Predicted Legit    Predicted Spam
Actual Legit          4,848               269
Actual Spam             402             4,820
```

---

### Project 6 — Author Attribution (Stylometry)
**Techniques:** Custom style feature UDFs, StringIndexer, VectorAssembler, Random Forest, feature importance

Predicted which of 10 Enron employees wrote an email using **writing style features only** — no sender information used.

**Style Features Engineered:**
- Average word length
- Average sentence length
- Punctuation ratio
- Uppercase ratio
- Unique word ratio (vocabulary richness)
- Email length
- Word count

**Results:**
- Accuracy: **51.5%** on 10-class classification
- Baseline (random): **10.0%**
- Improvement: **5.2x better than random**
- Most discriminative feature: **Uppercase Ratio (0.245 importance)**

---

### Project 7 — Anomaly Detection
**Techniques:** Z-score statistical analysis, weekly behavioural profiling, baseline modelling, bubble chart visualisation

Applied Z-score analysis to weekly email volumes across 1,321 active senders, flagging weeks that deviated more than 3 standard deviations from each sender's personal baseline.

**Results:**
- **488 anomalous weeks** detected across 1,321 senders
- Three distinct **company-wide spike clusters** identified:

| Event | Date | Anomalies |
|---|---|---|
| Stock price peak | December 2000 (Week 50) | 4 employees spike simultaneously |
| Internal accounting concerns | June 2001 (Week 23) | **6 employees spike simultaneously** |
| SEC investigation announced | October 2001 (Week 43) | 3 employees spike simultaneously |

The simultaneous spike of 6 independent employees in Week 23, 2001 is particularly significant — coordinated false positives are statistically near-impossible, making this a strong signal of a real organisational event.

---

## 💡 Key Technical Challenges Solved

| Challenge | Solution |
|---|---|
| CSV with multi-line quoted fields | `multiLine=True` + `quote` + `escape` options |
| PySpark `max()` overriding Python built-in inside UDF | `import builtins; builtins.max()` |
| Naive Bayes AUC reading as 0.235 (inverted) | Extracted calibrated probabilities via `DoubleType` UDF |
| Slow aggregations without caching | `.cache()` + `.count()` after every derived DataFrame |
| Imbalanced spam dataset (95%/5%) | Undersampling majority class to achieve 50/50 balance |
| Timestamp parsing failures (Enron date format) | `spark.sql.legacy.timeParserPolicy = LEGACY` |

---

## 📊 Results Summary

| Project | Technique | Best Result |
|---|---|---|
| EDA | Aggregation + Visualisation | 3 chart types, 4 key findings |
| Network | Graph analysis | 760K pairs, 115K unique relationships |
| Sentiment | TextBlob NLP | Scored 515K emails |
| TF-IDF | Spark MLlib | "investigation" +11,317% growth |
| Spam | Logistic Regression | 93.5% accuracy, AUC 0.972 |
| Attribution | Random Forest | 51.5% on 10-class (5.2x random) |
| Anomaly | Z-score statistics | 488 anomalies, 3 event clusters |

---

## 🚀 How to Run

1. Upload `emails.csv` and `stopwords.txt` to your Databricks volume
2. Update the file path in Cell 1 of each notebook
3. Run notebooks in order (01 → 07)
4. Each notebook is self-contained with markdown explanations

**Requirements:**
```
Databricks Runtime 12.0+ (includes PySpark 3.3+)
textblob (pip install textblob)
```

---

## 📚 Dataset

The Enron email dataset was made public by the Federal Energy Regulatory Commission (FERC) during its investigation. It is widely used in academic NLP and data science research.

Original source: [CMU Enron Email Dataset](https://www.cs.cmu.edu/~enron/)

---

## 👤 Author

**Barnabas Ogodo**
MSc Data Science — University of Salford

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/barnabas-ogodo)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Barnabas-sir/Barnabas-sir<img width="1662" height="833" alt="Screenshot 2026-04-28 150009" src="https://github.com/user-attachments/assets/ea8235cc-af52-46ae-ae3d-5831cfce67d1" />
<img width="1636" height="625" alt="Screenshot 2026-04-26 235641" src="https://github.com/user-attachments/assets/114dceba-bb27-4c80-9a5d-725a3a3b3393" />
<img width="1703" height="936" alt="Screenshot 2026-04-26 225813" src="https://github.com/user-attachments/assets/681b6efb-55fc-4605-9993-bbdd2f83a850" />
<img width="1625" height="703" alt="Screenshot 2026-04-24 221605" src="https://github.com/user-attachments/assets/1732c41e-a5e2-41e1-bc6f-35cdf844ac3b" />
)
