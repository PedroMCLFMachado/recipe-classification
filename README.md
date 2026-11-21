# Who is the cook? Classical preprocessing vs. BERT embeddings

Authorship attribution of recipes: given a recipe's name, tags, steps, description, ingredients and date, predict which of six chefs wrote it.

**Research question.** Does a carefully designed classical preprocessing pipeline outperform a pre-trained transformer used as a fixed feature extractor?

## Approach

Two feature pipelines are compared under the same downstream classifiers:

- **Classical:** text fields concatenated and encoded with TF-IDF (10,000 features, unigrams and bigrams), plus date features.
- **BERT embeddings:** all fields joined into a templated string and encoded with frozen `distilbert-base-uncased` (768-dimensional [CLS] representation from the last layer, max. 512 tokens).

Classifiers: Logistic Regression, linear SVC, MLP, Random Forest, and a hard-voting ensemble of the four. Hyperparameters were chosen by randomized search, and models were evaluated with 5-fold stratified cross-validation on ~3,000 recipes. Differences between pipelines were tested per model with one-sided paired t-tests or Wilcoxon signed-rank tests, depending on normality.

## Results

| Model | Classical | BERT | p-value |
|---|---|---|---|
| Logistic Regression | 0.892 | 0.857 | 0.0014 |
| SVC | 0.891 | 0.846 | 0.0005 |
| MLP | 0.892 | 0.851 | 0.0001 |
| Random Forest | 0.803 | 0.705 | 0.0002 |
| **Voting ensemble (hard)** | **0.896** | 0.854 | 0.0007 |

*Mean 5-fold cross-validation accuracy.*

- Classical features outperformed BERT embeddings for every classifier, with all differences significant at p < 0.05.
- TF-IDF captures chef-specific lexical habits (ingredients, measurements, terminology); frozen DistilBERT, without fine-tuning, likely blurs these domain-specific cues.
- 60% of misclassifications involve three chefs with overlapping ingredient and phrasing patterns.

Details are in the [project report](report.pdf).

## Running

The notebook `recipe_classification.ipynb` contains the full pipeline. It expects `train.csv` and `test-no-labels.csv` (semicolon-separated) in the working directory. The dataset was provided by the course and is not included in this repository.

Main dependencies: `pandas`, `numpy`, `scikit-learn`, `nltk`, `torch`, `transformers`, `scipy`, `matplotlib`, `seaborn`.

---

Course project for Natural Language (MSc in Computer Science and Engineering, Instituto Superior Técnico, 2025/26), by Laura Scherzberg, Pedro Machado and Vasco Brito.
