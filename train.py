from __future__ import division
from __future__ import print_function

import scipy.sparse as sp
import time
import tensorflow as tf

from utils import *
from models import GCNMI, GCNMI1L, GCNMIA, GCNMIA1L

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('train_ratio', 20, 'Number of labeled nodes per class')
flags.DEFINE_integer('repeat', 0, 'repeat ID of data split')

print(FLAGS.dataset)
print(FLAGS.flag_values_dict())

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
    load_data_split(FLAGS.dataset, train_ratio=FLAGS.train_ratio,
                    repeat=FLAGS.repeat)
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)

if FLAGS.model == 'gcn_mi':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCNMI
elif FLAGS.model == 'gcn_mi1':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCNMI1L
elif FLAGS.model == 'gcn_mia':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCNMIA
elif FLAGS.model == 'gcn_mia1':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCNMIA1L
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'training': tf.placeholder_with_default(False, shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

FLAGS.early_stopping = FLAGS.epochs
print(FLAGS.epochs, FLAGS.early_stopping)

best_epoch = -1
best_val_acc = -1
best_te_acc = -1

for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['training']: True})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test,
                                                  test_mask, placeholders)
    print("Test results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=",
          "{:.5f}".format(test_duration))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

    if acc > best_val_acc:
        best_epoch = epoch
        best_val_acc = acc
        best_te_acc = test_acc

print('best epoch:', best_epoch)
print('best test:', best_te_acc)
print("Optimization Finished!")