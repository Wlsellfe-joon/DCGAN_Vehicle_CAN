import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import cv2
import pandas as pd
import zipfile

from IPython import display
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 100
img_width = 16
dir = 'C:/~image file path/'

train = tf.keras.preprocessing.image_dataset_from_directory(dir+'/Train',
                                             shuffle=False,
                                             label_mode='categorical',
                                             #validation_split=0.3,
                                             #subset="training",
                                             color_mode = "rgb", # RGB image
                                             #interpolation='nearest',
                                             #seed = 123,
                                             batch_size=21500,
                                             image_size=(img_height,img_width))

for img, lab in train.take(1):
  print("Image shape: ", img.numpy().shape)
  print("Label: ", lab.numpy().shape)
print(train.class_names)

IMG = img.numpy()
Label = lab.numpy()
IMG_Impersonation = IMG[4000:7000,:,:,:] # Using Only Impersonation attack data
print(IMG_Impersonation.shape)


#Data preprocessing
train_images = IMG_Impersonation.reshape(IMG_Impersonation.shape[0], img_height, img_width, 3).astype('float32')
train_images = (train_images - 127.5) / 127.5 # 이미지를 [-1, 1]로 정규화합니다.
BUFFER_SIZE = 3000
BATCH_SIZE = 256

#Data Batch Set
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#Generator define
def make_generator_model():
    #project and reshape
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*4*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((25, 4, 256)))
    assert model.output_shape == (None, 25, 4, 256) # 배치사이즈: None

    #conv1, filter:128; kernel_size=5x5; stride : 1,1; padding YES(zeros); use_biase : False;
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #conv2
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 4, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #conv3
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 8, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #conv4
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 100, 16, 3)

    return model

# Before training, Generate imgs using Generator
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)


# Discriminator
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[100, 16, 3]))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Before Training, Discriminate img, img가 진짜인지 가짜인지, 양수/음수를 이용해서 판단
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# Loss func, Optimizer Define
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 생성자 손실함수
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# 감별자 손실함수
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# 생성자, 감별자 Optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#Check Point 저장
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# 훈련루프
EPOCHS = 300
noise_dim = 100
num_examples_to_generate = 4500

#Seed 재활용
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # GIF를 위한 이미지 생성
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # 15 에포크가 지날 때마다 모델을 저장
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # 마지막 에포크가 끝난 후 생성
  #display.clear_output(wait=True)
  #generate_and_save_images(generator,epochs,seed)

def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  #fig = plt.figure(figsize=(50,50))
  for i in range(predictions.shape[0]):
      data = np.uint8(predictions[i, :, :, :] * 127.5 + 127.5)
      plt.imshow(data)
      plt.axis('off')

  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

#%%time
train(train_dataset, EPOCHS)

# 마지막 Check point 저장
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def generating_MODEL(model, epoch, test_input):
  global predictions
  predictions = model(test_input, training=False)

  for i in range(predictions.shape[0]):
    data = predictions[i, :, :, :] * 127.5 + 127.5
    data = np.uint8(data)
    imageio.imwrite('C:/~Writing path~/image_name'+str(i)+'.png', data)
    print(i, "번째 완료!")

path = 'C:/~path to save a model/<model name>.h5'
generator.save(path)
generating_MODEL(generator,EPOCHS,seed)


