# DetectIt

**Intelligent Web Threat Detection Using Machine Learning**

DetectIt detects phishing URLs using machine learning. We compare 8 algorithms in two experiments:

- **Experiment 1:** all 69 features (URL and web page features) on the PhiUSIIL Phishing URL Dataset.
- **Experiment 2:** only the 40 features that can be created from a URL, trained on PhiUSIIL and a second dataset together, and tested on each dataset separately.

## Datasets
| Dataset | Use | Size |
|---|---|---|
| [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (Prasad & Chandra, 2024) | Experiments 1 and 2: training (80%) and testing (20%) | 235,795 URLs, 54 features |
| [Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset) (Hannousse & Yahiouche, 2021) | Experiment 2: training (80%) and testing (20%) | 11,430 URLs, 87 features |

## Algorithms
| Type | Algorithms |
|---|---|
| Supervised | Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM, Random Forest, XGBoost |
| Unsupervised | Isolation Forest |

## Project Structure
```
DetectIt-ml/
├── data/                           # datasets (not pushed to GitHub)
│   └── processed/                  # cleaned files created by the notebooks
├── notebooks/
│   ├── 01_eda.ipynb                # exploratory data analysis
│   ├── 02_preprocessing.ipynb      # cleaning, feature engineering and train/test split
│   ├── 03_external_dataset.ipynb   # preparing the second dataset and splitting it 80/20
│   ├── 04_xgboost.ipynb            # one notebook for each model
│   ├── 05_random_forest.ipynb
│   ├── 06_decision_tree.ipynb
│   ├── 07_logistic_regression.ipynb
│   ├── 08_knn.ipynb
│   ├── 09_svm.ipynb
│   ├── 10_naive_bayes.ipynb
│   ├── 11_isolation_forest.ipynb
│   ├── 12_model_comparison.ipynb   # comparing all models
│   └── model_template.ipynb        # starting point for every model notebook
├── results/                        # graphs, confusion matrices and metric tables
├── src/
│   ├── features.py                 # feature engineering functions
│   ├── data.py                     # loading the processed data for both experiments
│   └── evaluation.py               # shared metrics, cross-validation, sample weights, plots and result saving
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
2. Download the [Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset) and place `dataset_phishing.csv` in the `data/` folder.
3. Run `notebooks/01_eda.ipynb` to explore the data.
4. Run `notebooks/02_preprocessing.ipynb` to create the training and test files in `data/processed/`.
5. Run `notebooks/03_external_dataset.ipynb` to create the training and test parts of the second dataset in `data/processed/`.
6. Run the model notebooks `04` to `11`. Each one saves its results in `results/`.
7. Run `notebooks/12_model_comparison.ipynb` to compare all models.

## Team
| Name | Reg. No |
|---|---|
| Sharinda W.G.I. | EG/2023/5889 |
| Walakanda W.K.C.D. | EG/2023/5939 |
| Nelundeniya N.Y.M. | EG/2023/5748 |
| Koswattage K.S.T. | EG/2023/5687 |

EC5203 Machine Learning – Mini Project
Department of Electrical and Information Engineering, Faculty of Engineering, University of Ruhuna
