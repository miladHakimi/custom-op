
make
bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip2 install artifacts/*.whl

python -c "
import tensorflow as tf;
import tensorflow_zero_out as z;
import numpy as np;
a = [[1, 2, 3, 4, 5] , [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]];
b = [1, 2];
a = tf.convert_to_tensor(a, dtype=tf.float32);
b = tf.convert_to_tensor(b, dtype=tf.float32);
print(z.custom_conv(input=tf.reshape(a, [1, 4, 5, 1]), filter=tf.reshape(b, [1, 2, 1, 1]), strides=[2, 2], padding='SAME')[0].eval(session=tf.Session()))"

