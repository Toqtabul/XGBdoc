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
    question: "Can XGBoost handle categorical features?",
    answer: "Yes, XGBoost has native support for categorical features when using the hist tree method. Set enable_categorical=True and ensure your categorical columns have the 'category' dtype in pandas.",
    code: `# Enable categorical feature support
model = xgb.XGBClassifier(
    tree_method='hist',
    enable_categorical=True
)

# Make sure to convert columns to category dtype
df['category_col'] = df['category_col'].astype('category')`
  },
  {
    question: "How do I save and load a model?",
    answer: "Use the save_model and load_model methods. JSON format is recommended for portability.",
    code: `# Save model
model.save_model('model.json')

# Load model
loaded_model = xgb.Booster()
loaded_model.load_model('model.json')`
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
