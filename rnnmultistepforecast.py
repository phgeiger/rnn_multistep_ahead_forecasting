"""
Preliminary code for blog post "Implementing time series multi-step ahead forecasts using recurrent neural networks in 
TensorFlow". See also the README.md.

Code/idea sources, among others:
- https://medium.com/google-cloud/how-to-do-time-series-prediction-using-rnns-and-tensorflow-and-cloud-ml-engine-2ad2eeb189e8
- https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
- https://www.quora.com/In-TensorFlow-given-an-RNN-is-it-possible-to-use-the-output-of-the-previous-time-step-as-input-for-the-current-time-step-during-the-training-phase-How
- https://stackoverflow.com/questions/41123367/single-step-simulation-in-tensorflow-rnn

Model/intuition (note that currently no covariates are implemented):
X_t (state) = f(X_{t-1}, W_t (covariate), noise)
Y_t (observation) = g(X_t, noise)
"""


import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import numpy as np
import matplotlib.pyplot as plt


# SETUP AND PARAMS


# For reproducability:
tf.set_random_seed(43)
np.random.seed(43)

filename_data = 'daywise_max_planck_cafeteria_data.csv'
num_training_steps = 20000
num_hidden = 20
batch_size = 30


# SPECIFY MODEL FUNCTIONS


def recursive_multistep_model_fn(features, labels, mode, params=None):
    """
    Model function, based on "recursive" approach (for explanation see blog post). Idea: simultaneously cover the 
    classical 1-step ahead case, as well as the multistep ahead case, and later share weights between them via checkpoints.
    
    Notes:
    - Requires that at least one observation has to be made already.
    """

    params = params if len(params) is not 0 else {'num_hidden': 20}
    #further_params = features['further_params'] if 'further_params' in features.keys() else {'predict_steps_ahead': 1}

    #predict_steps_ahead = features['predict_steps_ahead'][0] if 'predict_steps_ahead' in features.keys() else 1
    predict_steps_ahead = params['predict_steps_ahead'] if 'predict_steps_ahead' in params.keys() else 1
    predict_multi_steps_ahead = predict_steps_ahead > 1


    if predict_multi_steps_ahead and mode == tf.estimator.ModeKeys.TRAIN:
        raise ValueError("Can only run prediction mode when being asked for multistep ahead predictions.")

    # if predict_multi_steps_ahead:
    #     tf.train.init_from_checkpoint(params['model_dir']

    observations = features['observations']
    batch_size = tf.shape(observations)[0]
    sequence_length = int(observations.shape[1])
    source_size = int(observations.shape[2])

    sources = observations
    targets = labels

    num_hidden = params['num_hidden'] if 'num_hidden' in params.keys() else 20
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=False)  # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, sources, dtype=tf.float32)

    outputs = tf.reshape(outputs, [batch_size * sequence_length, num_hidden])

    target_size = source_size

    weight = tf.Variable(tf.random_normal([num_hidden, target_size]))
    bias = tf.Variable(tf.random_normal([target_size]))

    predictions = tf.matmul(outputs, weight, name="predictions") + bias
    predictions = tf.reshape(predictions, [batch_size, sequence_length, target_size], name="predictions")

    if not predict_multi_steps_ahead:  # Classical singlestep ahead "mode":

        predictions_dict = {"predictions": predictions}

    else:  # Multistep ahead "mode":

        states = tf.expand_dims(state, 1)
        # to make predictions and state treatable in the same way

        last_prediction = predictions[:, -1, :]
        last_state = states[:, -1, :]

        i = 2
        while i <= predict_steps_ahead:

            next_output, next_state = rnn_cell(last_prediction, last_state)
            next_prediction = tf.matmul(next_output, weight) + bias
            # i.e., prediction for timepoint = timepoint of last observation + i
            # based on <https://stackoverflow.com/questions/41123367/single-step-simulation-in-tensorflow-rnn>

            predictions = tf.concat([predictions, tf.expand_dims(next_prediction, 1)], 1)
            states = tf.concat([states, tf.expand_dims(next_state, 1)], 1)

            last_prediction, last_state = next_prediction, next_state

            i += 1

        # Later: implement the above while loop using a tf.while_loop instaed ...
        #while_i = tf.constant(0)
        #while_cond = lambda ...
        #while_body = lambda ...

        predictions = predictions[:, sequence_length-1:, :]  # assuming that the task is to *only* deliver the ahead-predictions
        predictions_dict = {"predictions": predictions}

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        print(targets)
        print(predictions)
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(targets, predictions)
        }
        loss = tf.losses.mean_squared_error(targets, predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)


def joint_multistep_model_fn(features, labels, mode, params):
    """
    Baseline model function, based on "joint" approach (for explanation see blog post).
    """

    observations = features['observations']
    batch_size = tf.shape(observations)[0]
    sequence_length = int(observations.shape[1])
    source_size = int(observations.shape[2])
    targets_length = params['targets_length']

    sources = observations
    targets = labels

    num_hidden = params['num_hidden'] if 'num_hidden' in params.keys() else 20
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=False)
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, sources, dtype=tf.float32)

    output = outputs[:, -1 , :]
    #output = tf.reshape(output, [batch_size * sequence_length, num_hidden])

    weight = tf.Variable(tf.random_normal([num_hidden, targets_length]))
    bias = tf.Variable(tf.random_normal([targets_length]))

    predictions = tf.matmul(output, weight, name="predictions") + bias
    predictions = tf.reshape(predictions, [batch_size, targets_length, 1], name="predictions")

    predictions_dict = {"predictions": predictions}

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        print(targets)
        print(predictions)
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(targets, predictions)
        }
        loss = tf.losses.mean_squared_error(targets, predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)


# LOAD AND PREPARE DATA


def get_data():
    return np.loadtxt(filename_data)


panel_data = get_data()
#print(panel_data)
data = np.expand_dims(panel_data, axis=2)

panel_length = data.shape[1]
length_train = 100

source_length_test = 12
target_length_test = panel_length - source_length_test

# Train:

data_train = data[:length_train]

sources_train_recursive = np.array([panel[0:-1] for panel in data_train], dtype=np.float32)
targets_train_recursive = np.array([panel[1:] for panel in data_train], dtype=np.float32)

sources_train_joint = np.array([panel[0:source_length_test] for panel in data_train], dtype=np.float32)
targets_train_joint = np.array([panel[source_length_test:] for panel in data_train], dtype=np.float32)

# Test - same for all:

data_test = data[length_train:]
sources_test = np.array([panel[0:source_length_test] for panel in data_test], dtype=np.float32)
targets_test = np.array([panel[source_length_test:] for panel in data_test], dtype=np.float32)


# TRAIN/DERIVE RECURSIVE MULTISTEP AHEAD FORECASTER


# Idea:
# 1. training singlestep case
# 2. load trained weights (variables) from model_dir into model_fn that is now set to "predict_multi_steps_ahead"-mode

model_dir = 'tf_model'

# Create the Estimator:
singlestep_forecaster = tf.estimator.Estimator(model_fn=recursive_multistep_model_fn, model_dir=model_dir, params={'num_hidden': num_hidden})

# Set up logging for predictions:
tensors_to_log = {"predictions": "predictions"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the singlestep model:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"observations": sources_train_recursive},
    y=targets_train_recursive,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True)
singlestep_forecaster.train(
    input_fn=train_input_fn,
    steps=num_training_steps,
    hooks=[logging_hook])

# Derive recursive multistep model:
recursive_forecaster = tf.estimator.Estimator(model_fn=recursive_multistep_model_fn, model_dir=model_dir, params={'predict_steps_ahead': target_length_test, 'num_hidden': num_hidden})


# TRAIN BASELINE JOINT MULTISTEP AHEAD FORECASTER


# Create the Estimator:
joint_forecaster = tf.estimator.Estimator(model_fn=joint_multistep_model_fn, params={'num_hidden': num_hidden, 'targets_length': target_length_test})

# Set up logging for predictions:
tensors_to_log = {"predictions": "predictions"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the model:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"observations": sources_train_joint},
    y=targets_train_joint,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True)
joint_forecaster.train(
    input_fn=train_input_fn,
    steps=num_training_steps,
    hooks=[logging_hook])


# EVALUATION

# Metrics:

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"observations": sources_test},
    y=targets_test,
    num_epochs=1,
    shuffle=False)

eval_results_recursive = recursive_forecaster.evaluate(input_fn=eval_input_fn)
eval_results_joint = joint_forecaster.evaluate(input_fn=eval_input_fn)

print('Evaluation results of "recursive":')
print(eval_results_recursive)
print('Evaluation results of "joint":')
print(eval_results_joint)


# Plot:

plt.clf()

test_plot_indices = slice(0, 10)

targets_plot = np.concatenate([data_test[i] for i in range(test_plot_indices.start, test_plot_indices.stop)])
length = len(targets_plot)

plt.plot(range(length), targets_plot)

plot_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"observations": sources_test[test_plot_indices]},
    num_epochs=1,
    shuffle=False)

for forecaster in [recursive_forecaster, joint_forecaster]:
    predictions_raw = list(forecaster.predict(plot_input_fn))
    padding = np.zeros(source_length_test)
    padding.fill(None)
    predictions_plot = np.concatenate([np.concatenate([padding, np.squeeze(item["predictions"], axis=1)]) for item in predictions_raw])
    plt.plot(range(length), predictions_plot)

plt.show()






