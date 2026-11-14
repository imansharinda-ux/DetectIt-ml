# DetectIt

**Intelligent Web Threat Detection Using Machine Learning**

DetectIt detects phishing URLs using machine learning. We compare 8 algorithms on the PhiUSIIL Phishing URL Dataset, and test how well the models work on a second, independent dataset.

## Datasets
| Dataset | Use | Size |
|---|---|---|
| [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (Prasad & Chandra, 2024) | Training and internal testing | 235,795 URLs, 54 features |
| [Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset) (Hannousse & Yahiouche, 2021) | External testing | 11,430 URLs, 87 features |

## Algorithms
| Type | Algorithms |
|---|---|
| Supervised | Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM, Random Forest, XGBoost |
| Unsupervised | Isolation Forest |

## Project Structure
```
DetectIt-ml/
├── data/                      # datasets (not pushed to GitHub)
│   └── processed/             # cleaned train/test files created by 02_preprocessing
├── notebooks/
│   ├── 01_eda.ipynb           # exploratory data analysis
│   └── 02_preprocessing.ipynb # cleaning, feature engineering and train/test split
├── results/                   # graphs and result tables
├── src/
│   └── features.py            # feature engineering functions
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/imansharinda-ux/DetectIt-ml.git
cd DetectIt-ml
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
1. Download the [PhiUSIIL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) and place `PhiUSIIL_Phishing_URL_Dataset.csv` in the `data/` folder.
2. Run `notebooks/01_eda.ipynb` to explore the data.
3. Run `notebooks/02_preprocessing.ipynb` to create the training and test files in `data/processed/`.

## Team
| Name | Reg. No |
|---|---|
| Sharinda W.G.I. | EG/2023/5889 |
| Walakanda W.K.C.D. | EG/2023/5939 |
| Nelundeniya N.Y.M. | EG/2023/5748 |
| Koswattage K.S.T. | EG/2023/5687 |

EC5203 Machine Learning – Mini Project
Department of Electrical and Information Engineering, Faculty of Engineering, University of Ruhuna
