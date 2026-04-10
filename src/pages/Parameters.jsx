import InfoBox from '../components/InfoBox'
import ParamTable from '../components/ParamTable'

const generalParams = [
  {
    name: 'booster',
    type: 'string',
    default: 'gbtree',
    description: 'Which booster to use. Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.',
    options: ['gbtree', 'gblinear', 'dart']
  },
  {
    name: 'verbosity',
    type: 'int',
    default: '1',
    description: 'Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).',
    range: '[0, 3]'
  },
  {
    name: 'validate_parameters',
    type: 'boolean',
    default: 'false',
    description: 'When set to True, XGBoost will perform validation of input parameters to check whether a parameter is used or not.'
  },
  {
    name: 'nthread',
    type: 'int',
    default: 'max threads',
    description: 'Number of parallel threads used to run XGBoost. When choosing it, please keep thread contention and hyperthreading in mind.'
  },
  {
    name: 'disable_default_eval_metric',
    type: 'boolean',
    default: 'false',
    description: 'Flag to disable default metric. Set to 1 or true to disable.'
  }
]

const boosterParams = [
  {
    name: 'eta',
    alias: 'learning_rate',
    type: 'float',
    default: '0.3',
    description: 'Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.',
    range: '[0, 1]'
  },
  {
    name: 'gamma',
    alias: 'min_split_loss',
    type: 'float',
    default: '0',
    description: 'Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.',
    range: '[0, inf]'
  },
  {
    name: 'max_depth',
    type: 'int',
    default: '6',
    description: 'Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.',
    range: '[0, inf]'
  },
  {
    name: 'min_child_weight',
    type: 'float',
    default: '1',
    description: 'Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning.',
    range: '[0, inf]'
  },
  {
    name: 'subsample',
    type: 'float',
    default: '1',
    description: 'Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.',
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
    name: 'lambda',
    alias: 'reg_lambda',
    type: 'float',
    default: '1',
    description: 'L2 regularization term on weights. Increasing this value will make model more conservative.',
    range: '[0, inf]'
  },
  {
    name: 'alpha',
    alias: 'reg_alpha',
    type: 'float',
    default: '0',
    description: 'L1 regularization term on weights. Increasing this value will make model more conservative.',
    range: '[0, inf]'
  },
  {
    name: 'tree_method',
    type: 'string',
    default: 'auto',
    description: 'The tree construction algorithm used in XGBoost.',
    options: ['auto', 'exact', 'approx', 'hist', 'gpu_hist']
  }
]

const taskParams = [
  {
    name: 'objective',
    type: 'string',
    default: 'reg:squarederror',
    description: 'Specify the learning task and the corresponding learning objective.',
    options: ['reg:squarederror', 'reg:logistic', 'binary:logistic', 'binary:hinge', 'multi:softmax', 'multi:softprob', 'rank:pairwise', 'rank:ndcg']
  },
  {
    name: 'base_score',
    type: 'float',
    default: '0.5',
    description: 'The initial prediction score of all instances, global bias.'
  },
  {
    name: 'eval_metric',
    type: 'string',
    default: 'depends on objective',
    description: 'Evaluation metrics for validation data. Default metric is assigned according to objective.',
    options: ['rmse', 'mae', 'logloss', 'error', 'merror', 'mlogloss', 'auc', 'aucpr', 'ndcg', 'map']
  },
  {
    name: 'seed',
    type: 'int',
    default: '0',
    description: 'Random number seed.'
  }
]

export default function Parameters() {
  return (
    <div>
      <h1 className="mt-0">XGBoost Parameters</h1>

      <p className="text-gray-600 text-lg mb-6">
        Before running XGBoost, we must set three types of parameters: general parameters,
        booster parameters and task parameters.
      </p>

      <InfoBox type="note" title="Note">
        Parameters with a "default" value of "depends on objective" will have their default
        set based on the objective function you specify. Check the specific objective
        documentation for details.
      </InfoBox>

      <ul className="mb-8">
        <li><strong>General parameters</strong> relate to which booster we are using to do boosting, commonly tree or linear model</li>
        <li><strong>Booster parameters</strong> depend on which booster you have chosen</li>
        <li><strong>Task parameters</strong> decide on the learning task and the corresponding learning objective</li>
      </ul>

      <h2 id="general-parameters">General Parameters</h2>
      <p className="text-gray-600 mb-4">
        Parameters that relate to which booster we are using to do boosting, commonly tree or linear model.
      </p>
      <ParamTable items={generalParams} />

      <h2 id="booster-parameters">Booster Parameters</h2>
      <p className="text-gray-600 mb-4">
        Parameters for tree booster (gbtree and dart). These control the tree construction algorithm and regularization.
      </p>
      <ParamTable items={boosterParams} />

      <InfoBox type="warning" title="Warning">
        Setting inappropriate parameter values can lead to training failures or poor model
        performance. Always validate your parameters on a held-out validation set.
      </InfoBox>

      <h2 id="task-parameters">Task Parameters</h2>
      <p className="text-gray-600 mb-4">
        Parameters that decide on the learning task and the corresponding learning objective.
      </p>
      <ParamTable items={taskParams} />

      <InfoBox type="info" title="Tip">
        For classification tasks with imbalanced data, consider using <code>scale_pos_weight</code> to
        balance the positive and negative weights.
      </InfoBox>
    </div>
  )
}
