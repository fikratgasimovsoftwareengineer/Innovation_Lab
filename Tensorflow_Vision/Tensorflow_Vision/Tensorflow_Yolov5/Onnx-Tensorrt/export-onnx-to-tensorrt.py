
import tensorrt as trt

'''
    This python model exports model from onnx to tensorrt engine with required params:
        1. Half Precision and Precision = 16 Float
        2. Max WorkSpace 1<<30
        3. Create Dynamic shape, thus, model can provide inference on mini-batch of
            [1. 10, 20]
'''


class ConvertToEngine:

    def __init__(self, path_to_onnx, path_to_engine, max_workspace_size=1<<30, half_precision=False):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.path_to_onnx = path_to_onnx
        self.path_to_engine = path_to_engine
        self.max_workspace_size = max_workspace_size
        self.half_precision = half_precision
        #self.workspace = workspace

    def convert(self):
        builder = trt.Builder(self.TRT_LOGGER)
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, self.TRT_LOGGER)

        with open(self.path_to_onnx, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in parser.errors:
                    print(error)
                return None
        # set explicit batch max to 20
        profile = builder.create_optimization_profile()
        profile.set_shape('input_name', min=(1, 3, 640, 640), opt = (10, 3, 640, 640), max = (20, 3, 640, 640))
        config.add_optimization_profile(profile)
        print('Successfully TensorRT engine configured to [1, 10, 20] batch size')
        print('\n')

        if builder.platform_has_fast_fp16 and self.half_precision:
            config.set_flag(trt.BuilderFlag.FP16)
            
        engine = builder.build_engine(network, config)
        
        with open(self.path_to_engine, "wb") as f:
            f.write(engine.serialize())
        
        print('Successfully converted ONNX to TensorRT engine')
        print(f'Serialized in engine path: {self.path_to_engine}')


convert = ConvertToEngine('/tensorfl_vision/Tensorflow_Yolov5/yolov5/runs/train/exp18/weights/best.onnx','/tensorfl_vision/Tensorflow_Yolov5/yolov5/runs/train/exp18/weights/dynamic_minbatch/yolov5s.engine')
convert.convert()