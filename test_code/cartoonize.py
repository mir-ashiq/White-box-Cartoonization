import os
import cv2
import numpy as np
import tensorflow as tf 
import network
import guided_filter
from tqdm import tqdm



def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
    

def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.keras.layers.Input(shape=(None, None, 3))
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    model = tf.keras.Model(inputs=input_photo, outputs=final_out)
    model.load_weights(model_path)
    model.compile(optimizer='adam', loss='mse')

    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = image/127.5 - 1
            output = model.predict(image[np.newaxis, ...])
            output = tf.image.convert_image_dtype(output, dtype=tf.uint8)
            cv2.imwrite(save_path, output[0])
        except:
            print('cartoonize {} failed'.format(load_path))


    

if __name__ == '__main__':
    model_path = 'saved_models'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)
