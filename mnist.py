import numpy as np
import tensorflow as tf
import valohai

# s3 database valohai, copy and paste url of the dataset
'''default_inputs = {
    'dataset':'dataset://vgg_fuoco/new-dataset-mnist' 
}'''

# vgg fuoco dataset
default_inputs = { 
    'dataset':'//vgg_fuoco/new-dataset-version',
    'dataset2':'dataset://vgg_fuoco/new-dataset-version2'
    }

valohai.prepare(step='mnist', image="ultralytics/yolov5", default_inputs=default_inputs)

input_path = valohai.inputs('dataset').path()
with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
 
x_train, x_test = x_train / 255.0, x_test / 255.0
 
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
 
model.fit(x_train, y_train, epochs=5)
 
model.evaluate(x_test,  y_test, verbose=2)
 
output_path = valohai.outputs().path('model.h5')
model.save(output_path)
