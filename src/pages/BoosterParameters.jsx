import InfoBox from '../components/InfoBox'
import ParamTable from '../components/ParamTable'

const treeParams = [
  {
    name: 'eta',
    alias: 'learning_rate',
    type: 'float',
    default: '0.3',
    description: 'Step size shrinkage used in update to prevents overfitting.',
    range: '[0, 1]'
  },
  {
    name: 'gamma',
    alias: 'min_split_loss',
    type: 'float',
    default: '0',
    description: 'Minimum loss reduction required to make a further partition on a leaf node.',
    range: '[0, inf]'
  },
  {
    name: 'max_depth',
    type: 'int',
    default: '6',
    description: 'Maximum depth of a tree. Increasing this value will make the model more complex.',
    range: '[0, inf]'
  },
  {
    name: 'min_child_weight',
    type: 'float',
    default: '1',
    description: 'Minimum sum of instance weight (hessian) needed in a child.',
    range: '[0, inf]'
  },
  {
    name: 'max_delta_step',
    type: 'float',
    default: '0',
    description: 'Maximum delta step we allow each leaf output to be. 0 means no constraint.',
    range: '[0, inf]'
  },
  {
    name: 'subsample',
    type: 'float',
    default: '1',
    description: 'Subsample ratio of the training instances.',
    range: '(0, 1]'
  },
  {
    name: 'colsample_bytree',
    type: 'float',
    default: '1',
    description: 'Subsample ratio of columns when constructing each tree.',
    range: '(0, 1]'
  },
  {
    name: 'colsample_bylevel',
    type: 'float',
    default: '1',
    description: 'Subsample ratio of columns for each level.',
    range: '(0, 1]'
  },
  {
    name: 'colsample_bynode',
    type: 'float',
    default: '1',
    description: 'Subsample ratio of columns for each node (split).',
    range: '(0, 1]'
  },
  {
    name: 'lambda',
    alias: 'reg_lambda',
    type: 'float',
    default: '1',
    description: 'L2 regularization term on weights.',
    range: '[0, inf]'
  },
  {
    name: 'alpha',
    alias: 'reg_alpha',
    type: 'float',
    default: '0',
    description: 'L1 regularization term on weights.',
    range: '[0, inf]'
  },
  {
    name: 'scale_pos_weight',
    type: 'float',
    default: '1',
    description: 'Control the balance of positive and negative weights, useful for unbalanced classes.'
  },
  {
    name: 'grow_policy',
    type: 'string',
    default: 'depthwise',
    description: 'Controls a way new nodes are added to the tree.',
    options: ['depthwise', 'lossguide']
  },
  {
    name: 'max_leaves',
    type: 'int',
    default: '0',
    description: 'Maximum number of nodes to be added. Only relevant when grow_policy=lossguide.'
  }
]

const linearParams = [
  {
    name: 'lambda',
    alias: 'reg_lambda',
    type: 'float',
    default: '0',
    description: 'L2 regularization term on weights.'
  },
  {
    name: 'alpha',
    alias: 'reg_alpha',
    type: 'float',
    default: '0',
    description: 'L1 regularization term on weights.'
  },
  {
    name: 'updater',
    type: 'string',
    default: 'shotgun',
    description: 'Choice of algorithm to fit linear model.',
    options: ['shotgun', 'coord_descent']
  },
  {
    name: 'feature_selector',
    type: 'string',
    default: 'cyclic',
    description: 'Feature selection and ordering method.',
    options: ['cyclic', 'shuffle', 'random', 'greedy', 'thrifty']
  }
]

const dartParams = [
  {
    name: 'sample_type',
    type: 'string',
    default: 'uniform',
    description: 'Type of sampling algorithm.',
    options: ['uniform', 'weighted']
  },
  {
    name: 'normalize_type',
    type: 'string',
    default: 'tree',
    description: 'Type of normalization algorithm.',
    options: ['tree', 'forest']
  },
  {
    name: 'rate_drop',
    type: 'float',
    default: '0.0',
    description: 'Dropout rate (a fraction of previous trees to drop during the dropout).',
    range: '[0.0, 1.0]'
  },
  {
    name: 'one_drop',
    type: 'boolean',
    default: 'false',
    description: 'When this flag is enabled, at least one tree is always dropped during the dropout.'
  },
  {
    name: 'skip_drop',
    type: 'float',
    default: '0.0',
    description: 'Probability of skipping the dropout procedure during a boosting iteration.',
    range: '[0.0, 1.0]'
  }
]

export default function BoosterParameters() {
  return (
    <div>
      <h1 className="mt-0">Booster Parameters</h1>

      <p className="text-gray-600 text-lg mb-6">
        Detailed documentation for all booster parameters in XGBoost. The parameters depend
        on which booster you have chosen.
      </p>

      <h2 id="tree-booster">Parameters for Tree Booster (gbtree, dart)</h2>
      <p>
        These parameters are used when <code>booster</code> is set to <code>gbtree</code> or <code>dart</code>.
        Tree boosters use tree-based models for gradient boosting.
      </p>
      <ParamTable items={treeParams} />

      <InfoBox type="note" title="Note">
        The <code>colsample_by*</code> parameters work cumulatively. For example, with
        <code>colsample_bytree=0.5</code> and <code>colsample_bylevel=0.5</code>, each level
        will use 25% of features.
      </InfoBox>

      <h2 id="linear-booster">Parameters for Linear Booster (gblinear)</h2>
      <p>
        These parameters are used when <code>booster</code> is set to <code>gblinear</code>.
        Linear boosters use linear functions for gradient boosting.
      </p>
      <ParamTable items={linearParams} />

      <h2 id="dart-booster">Additional Parameters for DART Booster</h2>
      <p>
        DART (Dropouts meet Multiple Additive Regression Trees) booster has additional
        parameters for controlling the dropout behavior.
      </p>
      <ParamTable items={dartParams} />

      <InfoBox type="info" title="Tip">
        DART is particularly useful when you want to reduce overfitting while maintaining
        model complexity. It randomly drops trees during training, similar to dropout in
        neural networks.
      </InfoBox>
    </div>
  )
}
