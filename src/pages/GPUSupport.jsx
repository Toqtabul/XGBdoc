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
      <CodeBlock language="python">{`import xgboost as xgb

# Check if CUDA is available
print(f"XGBoost version: {xgb.__version__}")

# Try to use GPU
params = {'tree_method': 'hist', 'device': 'cuda'}
try:
    dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
    model = xgb.train(params, dtrain, num_boost_round=1)
    print("GPU is working!")
except Exception as e:
    print(f"GPU not available: {e}")`}</CodeBlock>

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
