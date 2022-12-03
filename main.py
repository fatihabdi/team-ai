import numpy as np
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
import time
from postprocessing import result_information
import cv2
from init_model import model_initialization
from preprocess_model import preprocessing_image
from inference import model_inference
from postprocessing import result_info_fashion


def main():
    path_model = "./model.pt"
    model_init = model_initialization(model_path=path_model)

    image_path = input()
    image_cv = cv2.imread(image_path)

    image_tensor = preprocessing_image(image_cv=image_cv)
    output_tensor = model_inference(image=image_tensor, model_infer=model_init)
    #post processing 
    dict_result = result_info_fashion(result_model)
    print("highest Score label :", highest_class)

if __name__ == "_main_":
    print("Starting program")
    main() 
