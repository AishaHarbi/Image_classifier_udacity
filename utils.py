import numpy as np 
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model


image_size=224


def predict(image_path,model_path,top_k):
    img = Image.open(image_path)
    img = process_image(np.asarray(img))
    img_expanded = np.expand_dims(img,axis=0)
    model = load_model(model_path,compile = False,custom_objects={'KerasLayer': hub.KerasLayer})
    pred = model.predict(img_expanded)[0]
    top_k = hightest_k(pred,top_k)
    return list(zip(*top_k))

def hightest_k(pred,k_num):
  k_list = sorted( [(x,i) for (i,x) in enumerate(pred)], reverse=True) [:k_num]
  return k_list

def process_image(img):
  img = tf.cast(img, tf.float32)
  img = tf.image.resize(img, (image_size, image_size))
  img /= 255
  return np.asarray(img)