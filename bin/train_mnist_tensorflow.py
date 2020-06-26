#!/home/user/miniconda/envs/tensorflow-2.2-cuda-10.1/bin/python
import tensorflow as tf
import numpy as np
import argparse
import time
import random
import os
import click

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

    return model


@click.command()
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=10)
@click.option('--no-cuda', type=bool, default=False)
def start_training(epochs, no_cuda, seed):
  # Load MNIST
  mnist = tf.keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  # Adding a dimension to the array -> new shape == (28, 28, 1), since the first layer in our model is a convolutional
  # layer and it requires a 4D input (batch_size, height, width, channels).
  # batch_size dimension will be added later on.
  train_images = train_images[..., None]
  test_images = test_images[..., None]

  # Normalizing the images to [0, 1] range.
  train_images = train_images / np.float32(255)
  test_images = test_images / np.float32(255)

  # Use MirroredStrategy for multi GPU support
  # If the list of devices is not specified in the`tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
  strategy = tf.distribute.MirroredStrategy()

  BUFFER_SIZE = len(train_images)
  BATCH_SIZE_PER_REPLICA = 64
  GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

  # Batch and distribute data
  train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE, seed=seed, reshuffle_each_iteration=False).batch(GLOBAL_BATCH_SIZE) 
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(BUFFER_SIZE, seed=seed, reshuffle_each_iteration=False).batch(GLOBAL_BATCH_SIZE) 
  train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

  # Fix seeds
  random_seed(seed)

  # Define Loss and accuracyc metrics
  with strategy.scope():
      # Set reduction to `none` so reduction can be done afterwards and divide by global batch size.
      loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True,
          reduction=tf.keras.losses.Reduction.NONE)
      def compute_loss(labels, predictions):
          per_example_loss = loss_object(labels, predictions)

          return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

      test_loss = tf.keras.metrics.Mean(name='test_loss')

      train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
      test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


  # Define model, optimizer, training- and test step
  with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()

    def train_step(inputs):
      images, labels = inputs

      with tf.GradientTape() as tape:
          predictions = model(images, training=True)
          loss = compute_loss(labels, predictions)

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      train_accuracy.update_state(labels, predictions)

      return loss 

    def test_step(inputs):
      images, labels = inputs

      predictions = model(images, training=False)
      t_loss = loss_object(labels, predictions)
      test_loss.update_state(t_loss)
      test_accuracy.update_state(labels, predictions)


  with strategy.scope():
    # `run` replicates the provided computation and runs it with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
      per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
  
    @tf.function
    def distributed_test_step(dataset_inputs):
      return strategy.run(test_step, args=(dataset_inputs,))

    gpu_runtime = time.time()
    for epoch in range(epochs):
      # TRAIN LOOP
      total_loss = 0.0
      num_batches = 0
      for dist_dataset in train_dist_dataset:
        total_loss += distributed_train_step(dist_dataset)
        num_batches += 1
      train_loss = total_loss / num_batches

      # TEST LOOP
      for dist_dataset in test_dist_dataset:
        distributed_test_step(dist_dataset)

      print(f'Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {train_accuracy.result()},'
            f'Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()}')

      # Reset states
      test_loss.reset_states()
      train_accuracy.reset_states()
      test_accuracy.reset_states()

    print(f'GPU Run Time: {str(time.time() - gpu_runtime)} seconds')


def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed) # Python general
    np.random.seed(seed)
    random.seed(seed) # Python random
    tf.random.set_seed(seed)
    tf.config.threading.set_intra_op_parallelism_threads = 1 # CPU only -> https://github.com/NVIDIA/tensorflow-determinism
    tf.config.threading.set_inter_op_parallelism_threads = 1 # CPU only
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


if __name__ == '__main__':
    print(f'Num GPUs Available: {len(tf.config.experimental.list_physical_devices("GPU"))}')

    start_training()
