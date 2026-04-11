import InfoBox from '../components/InfoBox'
import CodeBlock from '../components/CodeBlock'

export default function GPUSupport() {
  return (
    <div>
      <h1 className="mt-0">GPU Support</h1>

      <p className="text-gray-600 text-lg mb-6">
        XGBoost supports GPU acceleration for training, which can provide significant speedups
        for large datasets.
      </p>

      <h2 id="requirements">Requirements</h2>
      <p>To use GPU acceleration, you need:</p>
      <ul>
        <li>NVIDIA GPU with compute capability 3.5 or higher</li>
        <li>CUDA Toolkit 11.0 or later</li>
        <li>XGBoost compiled with GPU support (pip package includes GPU support by default)</li>
      </ul>

      <InfoBox type="note" title="Note">
        Starting from XGBoost 1.6.0, the pip package includes GPU support out of the box
        on supported platforms (Linux and Windows with CUDA).
      </InfoBox>

      <h2 id="installation">Installation</h2>
      <p>Install XGBoost with GPU support using pip:</p>
      <CodeBlock language="bash">pip install xgboost</CodeBlock>

      <p>Verify GPU is available:</p>
      <CodeBlock language="python">{`"""
LoRA Fine-tuning - Basic Implementation

Low-Rank Adaptation of Large Language Models
Paper: https://arxiv.org/abs/2106.09685

Key idea: Instead of fine-tuning all parameters, inject trainable
rank decomposition matrices into each layer while keeping pretrained
weights frozen.

Benefits:
- Drastically reduces trainable parameters (often by 10,000x)
- Lower memory footprint
- Faster training
- Easy to merge/switch between different task adaptations
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # base model
DATASET_NAME = "tatsu-lab/alpaca"        # instruction dataset

# LoRA hyperparameters
LORA_R = 8              # rank of decomposition matrices
LORA_ALPHA = 16         # scaling factor (typically 2 * r)
LORA_DROPOUT = 0.05     # dropout probability
TARGET_MODULES = [      # which layers to inject LoRA
    "q_proj",
    "k_proj", 
    "v_proj",
    "o_proj",
]

# Training hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
OUTPUT_DIR = "./lora_model"


# ============================================================================
# LOAD BASE MODEL AND TOKENIZER
# ============================================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True,  # 8-bit quantization for memory efficiency
)


# ============================================================================
# CONFIGURE LORA
# ============================================================================

lora_config = LoraConfig(
    r=LORA_R,                          # rank
    lora_alpha=LORA_ALPHA,             # scaling
    target_modules=TARGET_MODULES,     # which modules to adapt
    lora_dropout=LORA_DROPOUT,
    bias="none",                       # don't train bias terms
    task_type=TaskType.CAUSAL_LM,      # causal language modeling
)

# Inject LoRA adapters into model
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
print(f"All params: {all_params:,}")


# ============================================================================
# PREPARE DATASET
# ============================================================================

def format_instruction(sample):
    """Format dataset sample into instruction-following format"""
    instruction = sample["instruction"]
    input_text = sample["input"]
    output = sample["output"]
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return prompt


def preprocess_function(examples):
    """Tokenize the dataset"""
    texts = [format_instruction(ex) for ex in examples]
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


# Load and preprocess dataset
dataset = load_dataset(DATASET_NAME)
train_dataset = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)


# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,                          # mixed precision training
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    remove_unused_columns=False,
)


# ============================================================================
# TRAINER
# ============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)


# ============================================================================
# TRAIN
# ============================================================================

print("Starting LoRA fine-tuning...")
trainer.train()

# Save LoRA adapters only (very small file ~10-50MB)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapters saved to {OUTPUT_DIR}")


# ============================================================================
# INFERENCE WITH LORA
# ============================================================================

def generate_response(instruction, input_text=""):
    """Generate response using fine-tuned model"""
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[-1].strip()


# Test inference
test_instruction = "Explain what is machine learning in simple terms"
response = generate_response(test_instruction)
print(f"\nInstruction: {test_instruction}")
print(f"Response: {response}")


# ============================================================================
# LOAD LORA MODEL LATER
# ============================================================================

"""
To load the LoRA model in a new session:

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "./lora_model")
tokenizer = AutoTokenizer.from_pretrained("./lora_model")

# Now use model for inference
"""


# ============================================================================
# MERGE LORA WEIGHTS (OPTIONAL)
# ============================================================================

"""
To merge LoRA weights back into base model for deployment:

model = model.merge_and_unload()
model.save_pretrained("./merged_model")

This creates a full model checkpoint (same size as original).
Useful for deployment but loses the modularity benefit of LoRA.
"""`}</CodeBlock>

      <h2 id="usage">Usage</h2>
      <p>
        To use GPU for training, set the <code>tree_method</code> and <code>device</code> parameters:
      </p>
      <CodeBlock language="python">{`params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic'
}

model = xgb.train(params, dtrain, num_boost_round=100)`}</CodeBlock>

      <InfoBox type="info" title="Scikit-Learn API">
        For the scikit-learn wrapper, use the same parameters:
      </InfoBox>

      <CodeBlock language="python">{`from xgboost import XGBClassifier

model = XGBClassifier(
    tree_method='hist',
    device='cuda',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

model.fit(X_train, y_train)`}</CodeBlock>

      <h2 id="multi-gpu">Multi-GPU Training</h2>
      <p>
        XGBoost supports distributed training across multiple GPUs using Dask:
      </p>
      <CodeBlock language="python">{`import dask.dataframe as dd
from xgboost import dask as dxgb
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

# Create a cluster with all available GPUs
cluster = LocalCUDACluster()
client = Client(cluster)

# Load data as Dask DataFrame
ddf = dd.read_parquet('data.parquet')
X = ddf[feature_columns]
y = ddf['target']

# Create DaskDMatrix
dtrain = dxgb.DaskDMatrix(client, X, y)

# Train with multiple GPUs
params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'objective': 'binary:logistic'
}

output = dxgb.train(
    client, params, dtrain,
    num_boost_round=100
)

booster = output['booster']`}</CodeBlock>

      <h2 id="performance-tips">Performance Tips</h2>
      <ul>
        <li><strong>Batch size:</strong> Use larger batch sizes for better GPU utilization</li>
        <li><strong>max_bin:</strong> Set to a reasonable value (256-512) for hist tree method</li>
        <li><strong>Data type:</strong> Consider using float32 instead of float64 to reduce memory usage</li>
        <li><strong>Memory monitoring:</strong> Use GPU memory monitoring tools to avoid out-of-memory errors</li>
      </ul>

      <CodeBlock language="python">{`# Optimize memory usage
params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'max_bin': 256,  # Reduce memory usage
    'max_depth': 8,
    'subsample': 0.8,  # Use less data per tree
}`}</CodeBlock>

      <InfoBox type="warning" title="Memory Warning">
        GPU memory is limited. For very large datasets, consider using:
        <ul className="mt-2">
          <li>Data subsampling with <code>subsample</code> parameter</li>
          <li>External memory mode</li>
          <li>Distributed training across multiple GPUs</li>
        </ul>
      </InfoBox>

      <h2 id="troubleshooting">Troubleshooting</h2>
      <table>
        <thead>
          <tr>
            <th>Issue</th>
            <th>Solution</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA not found</td>
            <td>Ensure CUDA toolkit is installed and in PATH</td>
          </tr>
          <tr>
            <td>Out of memory</td>
            <td>Reduce max_bin, use subsample, or use external memory</td>
          </tr>
          <tr>
            <td>Slow training</td>
            <td>Check if GPU is being used (nvidia-smi), increase batch size</td>
          </tr>
          <tr>
            <td>Wrong GPU</td>
            <td>Set CUDA_VISIBLE_DEVICES environment variable</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
