# PhishLens

**Intelligent Web Threat Detection Using Machine Learning**

PhishLens detects phishing URLs using the PhiUSIIL Phishing URL Dataset and compares 8 machine learning algorithms.

## Dataset
- **PhiUSIIL Phishing URL Dataset** (UCI Machine Learning Repository; Prasad & Chandra, 2024)
- 235,795 URLs, 54 features, binary label (phishing / legitimate)

## Algorithms
| Type | Algorithms |
|---|---|
| Supervised | Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM, Random Forest, XGBoost |
| Unsupervised | Isolation Forest |

## Project Structure
```
phishlens-ml/
├── data/          # dataset (not pushed to GitHub)
├── notebooks/     # Jupyter notebooks
├── results/       # graphs and result tables
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/imansharinda-ux/phishlens-ml.git
cd phishlens-ml
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Team
| Name | Reg. No |
|---|---|
| Walakanda W.K.C.D. | EG/2023/5939 |
| Sharinda W.G.I. | EG/2023/5889 |
| Nelundeniya N.Y.M. | EG/2023/5748 |
| Koswattage K.S.T. | EG/2023/5687 |

EC5203 Machine Learning – Mini Project
Department of Electrical and Information Engineering, Faculty of Engineering, University of Ruhuna
