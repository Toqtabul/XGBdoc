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
      <CodeBlock language="python">{`import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (down)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (up)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        
        return self.final_conv(x)

# Example usage
if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`}</CodeBlock>

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
