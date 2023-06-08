import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
class TRTInference:
        
    # specify engine file path and input and output shape
    def __init__(self, engine_file_path, input_shape, output_shape, class_labels_file):
        self.logger = trt.Logger(trt.Logger.WARNING)

        self.engine_file_path = engine_file_path
            
        ## load engine here
        self.engine = self.load_engine(self.engine_file_path)

        # craete context
        self.context = self.engine.create_execution_context()

        # input shape
        self.input_shape = input_shape
            
        self.class_labels_file =  class_labels_file
        self.count = 0
            
        # output shape
        self.output_shape = output_shape
          
        with open(class_labels_file, 'r') as class_read:
            self.class_labels = [line.strip() for line in class_read.readlines()]

    def load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine_desentriliazed = runtime.deserialize_cuda_engine(f.read())

            return engine_desentriliazed
            
    def preprocess_image(self, image_path):
        
        # img list
        img_list = []
        # img path
        img_path = []
        count = 0
       
        for img_original in os.listdir(image_path):
            if count >= 10:
                continue
            if img_original.endswith('.jpg') or img_original.endswith('.png') or img_original.endswith('jpeg'):
                img_full_path = os.path.join(image_path, img_original)
                
                img = Image.open(img_full_path)
                
                # get image size
                #width, height = img.size
                #print(width, height)

                #img size = 640, 640]
                img = img.resize((self.input_shape[2], self.input_shape[3]), Image.NEAREST)

                img_np = np.array(img).astype(np.float32) / 255.0
            
                img_np = img_np.transpose((2,0,1))

                img_np = np.expand_dims(img_np, axis=0)
                count +=1
                
                img_list.append(img_np)
                img_path.append(img_full_path)
        

        return img_list , img_path
    
    # normalize model output

    def normalize_model(self, box, img_width, img_height):
        box[0] = box[0] / img_width
        box[1] = box[1] / img_height
        box[2] = box[2] / img_width
        box[3] = box[3] / img_height
        #print(box[0],box[1],box[2],box[3])
       
        return box
        

   
      
    def inference_detection(self,image_path):

        
        input_list, full_img_paths = self.preprocess_image(image_path)
        results = []

        for inputs , full_img_path in zip(input_list, full_img_paths):

            inputs = np.ascontiguousarray(inputs)

            outputs = np.empty(self.output_shape, dtype=np.float32)
            #print(outputs)

            d_inputs = cuda.mem_alloc(1 * inputs.nbytes)

            d_outpus = cuda.mem_alloc(1 * outputs.nbytes)

            bindings = [d_inputs ,d_outpus]

            cuda.memcpy_htod(d_inputs, inputs)

            self.context.execute_v2(bindings)

            # copy output back to host
            cuda.memcpy_dtoh(outputs, d_outpus)

           
            result = self.postprocess_img(outputs)        

            d_inputs.free()

            d_outpus.free()
            
            results.append(result)
          
        
            self.display_recognized_image(full_img_path, result)    
        return results
        
    # save images with detected results
    def display_recognized_image(self, image_path, result):
        
       
        #image = Image.open(image_path)
        image = cv2.imread(image_path)
        
            
        #for class_name in class_label:
        boxes, conf_scores, class_scores = result

        for box, conf_score, class_score in zip(boxes, conf_scores, class_scores):
            x_min, y_min, x_max, y_max = box
            label = self.class_labels[int(class_score)]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f'{label}: {conf_score:.2f}', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        path_to_detected_img = "/tensorfl_vision/Tensorflow_Yolov5/inference_yolov5_custom/detected_images" # test images
        
        if not os.path.exists(path_to_detected_img):
            
            os.makedirs(path_to_detected_img)
            
      
        save_img = os.path.join(path_to_detected_img, 'detected' + str(self.count) + '.png')
 
        cv2.imwrite(save_img, image)

        self.count += 1

        
        return image, save_img
#engien path 
engine_file_path ='/tensorfl_vision/Tensorflow_Yolov5/yolov5/runs/train/exp18/weights/dynamic_minbatch/yolov5s.engine'

# Load the TensorRT engine
input_shape = (10,3, 640, 640)
output_shape = (10, 25500, 7)

#image_path = '/deeplearning/resnet/rose.jpeg'

image_path = '/tensorfl_vision/Tensorflow_Yolov5/yolo/val/images'

label_path = '/tensorfl_vision/Tensorflow_Yolov5/yolo/val/labels'

path_to_class = "/tensorfl_vision/Tensorflow_Yolov5/inference_yolov5_custom/classes_yolo.txt"

inference = TRTInference(engine_file_path, input_shape, output_shape, path_to_class)

class_name = inference.inference_detection(image_path)
