import InfoBox from '../components/InfoBox'
import CodeBlock from '../components/CodeBlock'

export default function Installation() {
  return (
    <div>
      <h1 className="mt-0">Installation Guide</h1>

      <p className="text-gray-600 text-lg mb-6">
        This page gives instructions on how to build and install XGBoost from various sources.
      </p>

      <h2 id="python">Python</h2>
      <p>
        Pre-built binary wheels are available for Linux, Windows, and macOS. You can install
        XGBoost using pip:
      </p>
      <CodeBlock language="bash">pip install xgboost</CodeBlock>

      <InfoBox type="note" title="Note">
        Starting from version 1.6.0, XGBoost provides GPU support out of the box. No separate
        installation is needed for GPU acceleration.
      </InfoBox>

      <h2 id="conda">Conda</h2>
      <p>
        You may use conda to install XGBoost from the conda-forge channel:
      </p>
      <CodeBlock language="bash">conda install -c conda-forge xgboost</CodeBlock>

      <h2 id="r">R</h2>
      <p>
        You can install XGBoost from CRAN:
      </p>
      <CodeBlock language="r">install.packages("xgboost")</CodeBlock>

      <h2 id="build-from-source">Build from Source</h2>
      <p>
        XGBoost can be built from source. This is useful if you need custom configurations
        or want to use the latest development version.
      </p>

      <h3>Prerequisites</h3>
      <ul>
        <li>CMake 3.18 or higher</li>
        <li>C++ compiler with C++17 support (GCC 8+, Clang 10+, MSVC 2019+)</li>
        <li>Git</li>
      </ul>

      <h3>Steps</h3>
      <p><strong>1. Clone the repository:</strong></p>
      <CodeBlock language="bash">git clone --recursive https://github.com/dmlc/xgboost</CodeBlock>

      <p><strong>2. Create build directory:</strong></p>
      <CodeBlock language="bash">cd xgboost && mkdir build && cd build</CodeBlock>

      <p><strong>3. Configure with CMake:</strong></p>
      <CodeBlock language="bash">cmake ..</CodeBlock>

      <p><strong>4. Build:</strong></p>
      <CodeBlock language="bash">make -j$(nproc)</CodeBlock>

      <InfoBox type="info" title="GPU Support">
        To build with GPU support, add <code>-DUSE_CUDA=ON</code> to the CMake command.
        You'll need CUDA Toolkit 11.0 or later installed.
      </InfoBox>

      <h2 id="verify">Verify Installation</h2>
      <p>
        After installation, you can verify that XGBoost is properly installed:
      </p>
      <CodeBlock language="python">{`import xgboost as xgb
print(xgb.__version__)`}</CodeBlock>
    </div>
  )
}
