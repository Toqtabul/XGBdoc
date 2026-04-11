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

  
    for c in num: df[f"{c}_null"] = df[c].isnull().astype(int); df[c] = df[c].fillna(df[c].median())
    for c in cat: df[c] = df[c].fillna(df[c].mode()[0])

    for c in num: df[f"{c}_log"] = np.log1p(df[c].clip(0))
    df[num] = StandardScaler().fit_transform(df[num])
    if len(num) >= 2:
        poly = PolynomialFeatures(2, include_bias=False).fit_transform(df[num[:2]])
        df[[f"poly_{i}" for i in range(poly.shape[1])]] = poly

    for c in cat:
        if df[c].nunique() <= 10:
            df = pd.get_dummies(df, columns=[c], prefix=c)
        else:
            df[f"{c}_freq"] = df[c].map(df[c].value_counts(normalize=True))
            if target: df[f"{c}_target"] = df[c].map(df.groupby(c)[target].mean())

    for c in (dt_cols or []):
        dt = pd.to_datetime(df[c])
        df[f"{c}_hour"] = dt.dt.hour; df[f"{c}_dow"] = dt.dt.dayofweek
        df[f"{c}_month"] = dt.dt.month; df[f"{c}_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    
    if lag_col:
        for l in lags: df[f"{lag_col}_lag{l}"] = df[lag_col].shift(l)
        df[f"{lag_col}_roll3"] = df[lag_col].shift(1).rolling(3).mean()


    if interact:
        a, b = interact
        df[f"{a}x{b}"] = df[a] * df[b]; df[f"{a}d{b}"] = df[a] / (df[b] + 1)

    return df

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
    code: `"""
Transfer Learning with PyTorch - Basic Implementation

This script demonstrates the classic transfer learning approach:
1. Load a pretrained model (ImageNet weights)
2. Freeze the feature extractor
3. Train only the classifier on a new dataset

Strategy: Feature Extraction
Architecture: ResNet18
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10  # number of classes in your task
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Augmentation for training set
train_transform = transforms.Compose([
    transforms.Resize(256),              # resize shorter side
    transforms.RandomCrop(224),          # random crop 224x224
    transforms.RandomHorizontalFlip(),   # horizontal flip with p=0.5
    transforms.ToTensor(),
    transforms.Normalize(                # ImageNet stats normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Preprocessing for validation (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),          # center crop instead of random
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================================
# DATA LOADING
# ============================================================================

# Dataset should have structure:
# path/to/train/
#   class1/
#     img1.jpg
#     img2.jpg
#   class2/
#     ...

train_dataset = datasets.ImageFolder('path/to/train', transform=train_transform)
val_dataset = datasets.ImageFolder('path/to/val', transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ============================================================================
# MODEL
# ============================================================================

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers for feature extraction
for param in model.parameters():
    param.requires_grad = False

# Replace classifier for new task
num_features = model.fc.in_features  # 512 for ResNet18
model.fc = nn.Linear(num_features, NUM_CLASSES)

# Move model to GPU/CPU
model = model.to(DEVICE)


# ============================================================================
# TRAINING SETUP
# ============================================================================

criterion = nn.CrossEntropyLoss()

# Optimize only the new classifier parameters
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


# ============================================================================
# TRAINING AND VALIDATION FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """
    One training epoch
    
    Returns:
        epoch_loss: average loss over epoch
        epoch_acc: accuracy over epoch (%)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Model validation
    
    Returns:
        val_loss: average loss on validation set
        val_acc: accuracy on validation set (%)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(loader)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print(f'Device: {DEVICE}')
print(f'Train samples: {len(train_dataset)}')
print(f'Val samples: {len(val_dataset)}')
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
print('-' * 60)

best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Training
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    
    # Validation
    val_loss, val_acc = validate(
        model, val_loader, criterion, DEVICE
    )
    
    # Logging
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_model.pth')
        print(f'  ✓ Saved best model (Val Acc: {val_acc:.2f}%)')
    
    print('-' * 60)

print(f'\nTraining completed. Best Val Acc: {best_val_acc:.2f}%')


# ============================================================================
# SAVE FINAL MODEL
# ============================================================================

torch.save(model.state_dict(), 'transfer_model.pth')
print('Final model saved to transfer_model.pth')`
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
