import tensorrt as trt

'''
This notebook prints input and output shape of tensorrt engine
'''

class VisualizeEngine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Runtime(trt.Logger())
        
        with open(self.engine_path, 'rb') as f:
            self.engine = self.logger.deserialize_cuda_engine(f.read())
        
    def getShapeEngine(self):
        input_shape =  self.engine.get_binding_shape(0)
        print("****Input shape: ****", input_shape)
        
        
        input_size = trt.volume(input_shape) * self.engine.max_batch_size
        print('****Input Size :*** ',input_size)
        
        output_shape = self.engine.get_binding_shape(1)
        print("****Output Shape : ****", output_shape)
        
        output_size = trt.volume(output_shape) * self.engine.max_batch_size
        print('****Output Size : ****',output_size)
   
    # init classs
engine_model = VisualizeEngine('/tensorfl_vision/Tensorflow_Yolov5/yolov5/runs/train/exp18/weights/yolov5s.engine')
engine_model.getShapeEngine()