import tensorrt as trt
import numpy as np

def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_file_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        #output_shape = engine.get_binding_shape(1)
    return engine


'''def process_output(output):
    #engine_file_path = '/tensorfl_vision/Tensorflow_Yolov5/yolov5/runs/train/exp18/weights/dynamic_minbatch/yolov5s.engine'
    #output = load_engine(engine_file_path)
    print(f'Output shape: {output}')
    output = np.reshape((output.shape[1], output.shape[2]))

    x,y,w,h,obj_score,class_score = np.split(output, [1,2,3,4,5,6], axis=-1)
    print(x,y,w,h,obj_score,class_score)
    return x, y, w, h, obj_score, class_score'''



engine_file_path = '/tensorfl_vision/Inference_Tensorrt_YOLOV5/my_custom_model/yolov5s.engine'

engine = load_engine(engine_file_path)

#etCoordinates = process_output(engine_shape)



# get input
input_tensor_name = engine.get_binding_name(0)
output_tensor_name = engine.get_binding_name(1)

# get input output tensor shapes

input_shape = engine.get_binding_shape(0)
output_shape = engine.get_binding_shape(1)

print(f'Input tensor name : {input_tensor_name}, shape: {input_shape}')
print(f'Output tensor name : {output_tensor_name}, shape: {output_shape}')