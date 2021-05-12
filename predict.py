import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse 
from utils import predict, hightest_k, process_image

parser = argparse.ArgumentParser(description="Query the model")
    
parser.add_argument("path_to_img", action="store",
                    help="path to input image")
parser.add_argument("saved_model_path", action="store",
                    help="path the saved model")

parser.add_argument("--top_k", action="store",
                        default= 5,
                    help="top k classes with probability")

parser.add_argument("--category_names", action="store",
                    help="path to class_names file")
   
    
args = parser.parse_args()
print(args)
    
saved_model_path = args.saved_model_path
path_to_img = args.path_to_img
top_k = int(args.top_k)
category_names = args.category_names
probs, classes = predict(path_to_img, saved_model_path, top_k)

if (args.category_names == None):
          print("{} highest proabilities: {} ".format(top_k,probs))
          print("Corresponding labels: {}".format(classes))
else:
          with open(category_names, 'r') as f:
            class_names = json.load(f)
          labels = [class_names[str(x)] for x in classes]
          print("{} highest proabilities: {} ".format(top_k,probs))
          print("Corresponding labels: {}".format(labels))
          
      
        



    

    
    
    
    
   








