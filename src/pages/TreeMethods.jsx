import InfoBox from '../components/InfoBox'
import CodeBlock from '../components/CodeBlock'

export default function TreeMethods() {
  return (
    <div>
      <h1 className="mt-0">Tree Methods</h1>

      <p className="text-gray-600 text-lg mb-6">
        XGBoost supports several tree construction algorithms. Each has its own characteristics
        and use cases. Choose the right one based on your dataset size and available resources.
      </p>

      <h2 id="auto">auto</h2>
      <p>
        Use heuristics to choose the fastest method. For small datasets, <code>exact</code> will
        be used. For larger datasets, <code>approx</code> or <code>hist</code> will be selected.
        This is the default option.
      </p>
      <CodeBlock language="python">{`params = {'tree_method': 'auto'}`}</CodeBlock>

      <h2 id="exact">exact</h2>
      <p>
        Exact greedy algorithm. Enumerates all split points and finds the best split.
        This is the most accurate but slowest method. Best for small to medium datasets.
      </p>
      <CodeBlock language="python">{`params = {'tree_method': 'exact'}`}</CodeBlock>

      <InfoBox type="warning" title="Warning">
        The <code>exact</code> method can be very slow and memory-intensive for large datasets.
        Consider using <code>hist</code> for datasets with more than 10,000 samples.
      </InfoBox>

      <h2 id="approx">approx</h2>
      <p>
        Approximate greedy algorithm using quantile sketch and gradient histogram.
        Better for large datasets where exact method is too slow.
      </p>
      <CodeBlock language="python">{`params = {'tree_method': 'approx'}`}</CodeBlock>

      <h2 id="hist">hist</h2>
      <p>
        Fast histogram optimized approximate greedy algorithm. Recommended for large datasets.
        Uses histogram-based algorithm for split finding.
      </p>
      <CodeBlock language="python">{`params = {'tree_method': 'hist'}`}</CodeBlock>

      <h3>Features of hist:</h3>
      <ul>
        <li>Faster training speed</li>
        <li>Lower memory usage</li>
        <li>Support for categorical features</li>
        <li>Better scalability for large datasets</li>
      </ul>

      <InfoBox type="note" title="Note">
        The <code>hist</code> method bins continuous features into discrete bins, which
        speeds up training significantly while maintaining good accuracy.
      </InfoBox>

      <h2 id="gpu_hist">gpu_hist</h2>
      <p>
        GPU implementation of hist algorithm. Requires CUDA-enabled GPU.
        Provides significant speedup for large datasets.
      </p>
      <CodeBlock language="python">{`params = {
    'tree_method': 'hist',
    'device': 'cuda'
}`}</CodeBlock>

      <h3>Requirements:</h3>
      <ul>
        <li>CUDA-capable GPU (compute capability 3.5+)</li>
        <li>CUDA Toolkit installed</li>
        <li>XGBoost built with GPU support</li>
      </ul>

      <InfoBox type="info" title="Tip">
        For GPU training, also consider adjusting <code>max_bin</code> parameter. Larger
        values can improve accuracy but increase memory usage and training time.
      </InfoBox>

      <h2 id="comparison">Comparison Table</h2>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Speed</th>
            <th>Memory</th>
            <th>Best For</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>exact</code></td>
            <td>Slow</td>
            <td>High</td>
            <td>Small datasets (&lt;10K samples)</td>
          </tr>
          <tr>
            <td><code>approx</code></td>
            <td>Medium</td>
            <td>Medium</td>
            <td>Medium datasets</td>
          </tr>
          <tr>
            <td><code>hist</code></td>
            <td>Fast</td>
            <td>Low</td>
            <td>Large datasets, CPU training</td>
          </tr>
          <tr>
            <td><code>gpu_hist</code></td>
            <td>Very Fast</td>
            <td>GPU Memory</td>
            <td>Large datasets, GPU available</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
