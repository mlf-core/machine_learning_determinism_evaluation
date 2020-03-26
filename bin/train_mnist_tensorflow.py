#!/home/user/miniconda/envs/tensorflow-2.1-cuda-10.1/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
