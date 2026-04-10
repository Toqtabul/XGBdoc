import InfoBox from '../components/InfoBox'
import CodeBlock from '../components/CodeBlock'

export default function GetStarted() {
  return (
    <div>
      <h1 className="mt-0">Get Started with XGBoost</h1>

      <p className="text-gray-600 text-lg mb-6">
        This is a quick start tutorial showing snippets for you to quickly try out XGBoost
        on the demo dataset on a binary classification task.
      </p>

      <h2 id="basic-usage">Basic Usage</h2>
      <p>
        Here is a simple example to get you started with XGBoost in Python:
      </p>
      <CodeBlock language="python">{`import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

# Load data
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',  
    early_stopping_rounds=50,
    tree_method='hist',
    device='cuda'
)

 model = XGBRegressor(
    n_estimators=1000,
     learning_rate=0.05,
     max_depth=6,
     subsample=0.8,
     colsample_bytree=0.8,
     random_state=42,
     eval_metric='rmse',
     early_stopping_rounds=50,
     tree_method='hist',
     device='cuda'
 )

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# Make predictions
val_preds = model.predict(X_val)
val_proba = model.predict_proba(X_val)[:, 1] 
print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(20))


test_preds = model.predict(test_df)
test_proba = model.predict_proba(test_df)[:, 1]



submission = pd.DataFrame({
    'id': test_df['id'], 
    'target': test_proba
})


submission.to_csv('submission.csv', index=False)`}</CodeBlock>

      <InfoBox type="note" title="Note">
        <code>DMatrix</code> is an internal data structure used by XGBoost, which is optimized
        for both memory efficiency and training speed.
      </InfoBox>

      <h2 id="sklearn-api">Scikit-Learn API</h2>
      <p>
        XGBoost provides a scikit-learn compatible API for easy integration with the
        scikit-learn ecosystem:
      </p>
      <CodeBlock language="python">{`from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create and train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=3
)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")`}</CodeBlock>

      <h2 id="early-stopping">Early Stopping</h2>
      <p>
        Early stopping is a useful technique to prevent overfitting:
      </p>
      <CodeBlock language="python">{`# Using early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtest, 'eval'), (dtrain, 'train')],
    early_stopping_rounds=10,
    verbose_eval=True
)`}</CodeBlock>

      <InfoBox type="info" title="Tip">
        Setting <code>early_stopping_rounds</code> helps automatically stop training when
        validation performance stops improving, saving computation time and preventing overfitting.
      </InfoBox>

      <h2 id="feature-importance">Feature Importance</h2>
      <p>
        XGBoost provides methods to understand feature importance:
      </p>
      <CodeBlock language="python">{`import xgboost as xgb
import matplotlib.pyplot as plt

# After training your model
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()`}</CodeBlock>

      <h2 id="save-load">Save and Load Models</h2>
      <p>
        Save your trained model for later use:
      </p>
      <CodeBlock language="python">{`# Save model
model.save_model('model.json')

# Load model
loaded_model = xgb.Booster()
loaded_model.load_model('model.json')`}</CodeBlock>

      <InfoBox type="warning" title="Warning">
        Always save models in JSON or UBJON format for production use. The pickle format
        is not guaranteed to be compatible across different XGBoost versions.
      </InfoBox>
    </div>
  )
}
