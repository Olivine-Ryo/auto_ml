from icrawler.builtin import GoogleImageCrawler, BaiduImageCrawler, BingImageCrawler
import logging

import shutil
import os

import random
random.seed(1984)

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense
from keras import optimizers
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
from PIL import ImageFile

import sys

def crawel_auto(search_word, get_num):
    dir_name = "crawel"
    print("Googleのクローリングを開始しました。")
    # Google
    googleCrawler = GoogleImageCrawler(storage={"root_dir": f'{dir_name}/{search_word}'}, log_level=logging.CRITICAL)
    googleCrawler.crawl(keyword=search_word, max_num=get_num)

    #print("Baiduのクローリングを開始しました。")
    #Baidu
    #baiduCrawler = BaiduImageCrawler(storage={"root_dir": f'{dir_name}/{search_word}'}, log_level=logging.CRITICAL)
    #baiduCrawler.crawl(keyword=search_word, max_num=get_num, file_idx_offset=get_num)

    print("Bingのクローリングを開始しました。")
    #Bing
    bingCrawler = BingImageCrawler(storage={"root_dir": f'{dir_name}/{search_word}'}, log_level=logging.CRITICAL)
    bingCrawler.crawl(keyword=search_word, max_num=get_num, file_idx_offset=get_num * 2)

def data_split(topics, ratio):
    shutil.rmtree('train')
    shutil.rmtree('test')
    os.makedirs("train",exist_ok=True)
    os.makedirs("test",exist_ok=True)
    for s in topics:
        os.makedirs("train/" + s,exist_ok=True)
        os.makedirs("test/" + s,exist_ok=True)
    for c in topics:
        pics = os.listdir("crawel/" + c)
        pics = [s for s in pics if "jpg" in s]
        random.shuffle(pics)    
        n_pics = len(pics)
        n_train = int(n_pics*ratio)
        for pic in pics[:n_train]:
            shutil.copy("crawel/" + c + "/" + pic, "train/" + c)
        for pic in pics[n_train:]:
            shutil.copy("crawel/" + c + "/" + pic, "test/" + c)

def main():
    topics = sys.argv[1:]
    for word in topics:
        crawel_auto(word,10000)
    data_split(topics,0.8)

    nb_classes = len(topics)
    train_data_dir = './train'
    validation_data_dir = './test'
    nb_train_samples = 100
    nb_validation_samples = 100 
    img_width, img_height = 224, 224

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
      train_data_dir,
      target_size=(img_width, img_height),
      color_mode='rgb',
      classes=topics,
      class_mode='categorical',
      batch_size=16)

    validation_generator = validation_datagen.flow_from_directory(
      validation_data_dir,
      target_size=(img_width, img_height),
      color_mode='rgb',
      classes=topics,
      class_mode='categorical',
      batch_size=16)



    ImageFile.LOAD_TRUNCATED_IMAGES = True

    nb_classes = len(topics)
    train_data_dir = './train'
    validation_data_dir = './test'
    nb_train_samples = 100
    nb_validation_samples = 100 
    img_width, img_height = 500, 500

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
      train_data_dir,
      target_size=(img_width, img_height),
      color_mode='rgb',
      classes=topics,
      class_mode='categorical',
      batch_size=16)

    validation_generator = validation_datagen.flow_from_directory(
      validation_data_dir,
      target_size=(img_width, img_height),
      color_mode='rgb',
      classes=topics,
      class_mode='categorical',
      batch_size=16)

    input_tensor = Input(shape=(img_width, img_height, 3))
    ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))
    top_model.add(Dense(nb_classes, activation='softmax'))

    model = Model(input=ResNet50.input, output=top_model(ResNet50.output))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=5,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

    
    print('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join(f_model,'model.json'), 'w').write(json_string)
    
    print('save weights')
    model.save_weights('model.hdf5')
    
if __name__ == "__main__":
    main()
