import numpy as np
import tensorflow as tf

import valohai

# s3 database valohai, copy and paste url of the dataset
'''default_inputs = {
    'dataset':'dataset://vgg_fuoco/new-dataset-mnist' 
}'''

# vgg fuoco dataset
default_inputs = { 
    'train/images':'dataset://vgg_fuoco/new-dataset-version', # train
    'val/images':'dataset://vgg_fuoco/new-dataset-version2', # val
    'train/labels':'dataset://vgg_fuoco/new-dataset-train_labels',
    'val/labels':'dataset://vgg_fuoco/new-dataset-val_labels'
    } 

# custom docker gpu
valohai.prepare(step='mnist', image='fikrat/tensorflow:latest-gpu', default_inputs=default_inputs)

input_path = valohai.inputs('train/images').dir_path

#train - > /valohai/inputs/dataset
#val - > /valohai/inputs/dataset2

input_path2 = valohai.inputs('val/images').dir_path

#input_train_labels = valohai.inputs


'''with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']'''
 
#x_train, x_test = x_train / 255.0, x_test / 255.0
 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
 
#model.fit(x_train, y_train, epochs=5)
 
#model.evaluate(x_test,  y_test, verbose=2)
 
output_path = valohai.outputs().path('model.h5')
model.save(output_path)
