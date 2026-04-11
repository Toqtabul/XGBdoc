import InfoBox from '../components/InfoBox'
import CodeBlock from '../components/CodeBlock'

const faqs = [
  {
    question: "How do I choose the number of boosting rounds?",
    answer: "Use early stopping with a validation set. Set a large number of rounds and let XGBoost stop automatically when validation performance stops improving.",
    code: `model = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=[(dtest, 'eval')],
    early_stopping_rounds=10
)`
  },
  {
    question: "How do I handle missing values?",
    answer: "XGBoost handles missing values automatically. It learns the best direction to go when a value is missing during training. You don't need to impute missing values."
  },
  {
    question: "How do I prevent overfitting?",
    answer: "There are several strategies: reduce max_depth, increase min_child_weight, use subsample and colsample_bytree for regularization, increase lambda (L2 regularization), and use early stopping."
  },
  {
    question: "Does Xgb provide Feature Enginnering?",
    answer: "Yes, XGBoost has native support for categorical features when using the hist tree method. Set enable_categorical=True and ensure your categorical columns have the 'category' dtype in pandas.",
    code: `import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def feature_engineering(df, num, cat, dt_cols=None, target=None,
                         lag_col=None, lags=[1,2], interact=None):
    df = df.copy()

    # Пропуски
    for c in num: df[f"{c}_null"] = df[c].isnull().astype(int); df[c] = df[c].fillna(df[c].median())
    for c in cat: df[c] = df[c].fillna(df[c].mode()[0])

    # Числовые: log + scale + poly
    for c in num: df[f"{c}_log"] = np.log1p(df[c].clip(0))
    df[num] = StandardScaler().fit_transform(df[num])
    if len(num) >= 2:
        poly = PolynomialFeatures(2, include_bias=False).fit_transform(df[num[:2]])
        df[[f"poly_{i}" for i in range(poly.shape[1])]] = poly

    # Категориальные: OHE / target+freq encoding
    for c in cat:
        if df[c].nunique() <= 10:
            df = pd.get_dummies(df, columns=[c], prefix=c)
        else:
            df[f"{c}_freq"] = df[c].map(df[c].value_counts(normalize=True))
            if target: df[f"{c}_target"] = df[c].map(df.groupby(c)[target].mean())

    # Временные
    for c in (dt_cols or []):
        dt = pd.to_datetime(df[c])
        df[f"{c}_hour"] = dt.dt.hour; df[f"{c}_dow"] = dt.dt.dayofweek
        df[f"{c}_month"] = dt.dt.month; df[f"{c}_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Лаги + rolling
    if lag_col:
        for l in lags: df[f"{lag_col}_lag{l}"] = df[lag_col].shift(l)
        df[f"{lag_col}_roll3"] = df[lag_col].shift(1).rolling(3).mean()

    # Взаимодействия
    if interact:
        a, b = interact
        df[f"{a}x{b}"] = df[a] * df[b]; df[f"{a}d{b}"] = df[a] / (df[b] + 1)

    return df


# Пример
if __name__ == "__main__":
    df = pd.DataFrame({
        "age": [25, np.nan, 35, 40], "income": [1000, 5000, 300, 8000],
        "cat": ["A","B","A","C"], "city": ["x","y","x","z"],
        "ts": pd.date_range("2024-01-01", periods=4, freq="h"),
        "sales": [10, 20, 15, 25], "target": [0.1, 0.9, 0.4, 0.7]
    })
    out = feature_engineering(df, num=["age","income"], cat=["cat","city"],
                               dt_cols=["ts"], target="target",
                               lag_col="sales", interact=("age","income"))
    print(f"{df.shape[1]} → {out.shape[1]} признаков")`
  },
  {
    question: "What's the difference between native API and scikit-learn API?",
    answer: "The native API uses DMatrix and xgb.train(), offering more control and features. The scikit-learn API (XGBClassifier, XGBRegressor) is easier to use and integrates with sklearn pipelines and tools like cross_val_score and GridSearchCV."
  },
  {
    question: "How do I tune hyperparameters?",
    answer: "Use GridSearchCV or RandomizedSearchCV from scikit-learn, or dedicated tools like Optuna or Hyperopt for more efficient hyperparameter optimization.",
    code: `from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300]
}

model = xgb.XGBClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")`
  },
  {
    question: "How do I handle imbalanced datasets?",
    answer: "Use scale_pos_weight parameter to balance class weights. Set it to the ratio of negative to positive samples.",
    code: `# For binary classification with imbalanced classes
scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

params = {
    'scale_pos_weight': scale,
    'objective': 'binary:logistic'
}`
  }
]

export default function FAQ() {
  return (
    <div>
      <h1 className="mt-0">Frequently Asked Questions</h1>

      <p className="text-gray-600 text-lg mb-6">
        Common questions and answers about XGBoost usage, best practices, and troubleshooting.
      </p>

      <div className="space-y-8">
        {faqs.map((faq, idx) => (
          <div key={idx} className="border-b border-gray-200 pb-6">
            <h3 className="text-lg font-semibold text-[#1565c0] mb-3">
              {faq.question}
            </h3>
            <p className="text-gray-700 mb-3">{faq.answer}</p>
            {faq.code && (
              <CodeBlock language="python">{faq.code}</CodeBlock>
            )}
          </div>
        ))}
      </div>

      <InfoBox type="info" title="Need more help?">
        Check out the official XGBoost documentation or visit the GitHub repository for
        more examples and community support.
      </InfoBox>
    </div>
  )
}
