

```python
#!pip install keras
#!pip install numpy
#!pip install imageio
#!pip install matplotlib
#!pip install opencv-python
#!pip install pydot
#!pip install graphviz
#!pip install azure-cognitiveservices-search-imagesearch
# Note: restart kernel after installing the above.
#!pip freeze

import threading
import sys
import cv2
import sys
import os
import csv
import itertools
import numpy as np
from PIL import Image
import imageio
import itertools as it
import tensorflow as tf
import keras
import pprint as pp
print("Keras version %s" % keras.__version__)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K

print("Tensorflow version %s" % tf.__version__)

import pprint
pp = pprint.PrettyPrinter(depth=6)
```

    Using TensorFlow backend.
    

    Keras version 2.0.8
    Tensorflow version 1.3.0
    


```python
# Add the standard include path for FoosMetrics
sys.path.insert(0, './../../Code')

import importlib
import process
importlib.reload(process)
```




    <module 'process' from './../../Code\\process.py'>




```python
# Load the data and frames, and add the normalizer
training = process.Processor("segmentation_loader.json")
validation = process.Processor("segmentation_loader.json")
#validation = process.Processor("data_loader_validation.json")
#output_names = ["g1","d1","o1","f1","f2","o2","d2","g2","-2"]
```

    Creating transform: random_video_loader
    Adding folder '.\..\..\Data\Raw\RawMatchesContinuous\'
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_1.mp4': 3430 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_2.mp4': 8916 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_3.mp4': 15947 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_4.mp4': 15260 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_5.mp4': 6345 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\images_0.mp4': 3040 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\images_0.mp4': Repeating 10.000000 times.
    '.\..\..\Data\Raw\RawMatchesContinuous\Nationals2_1.mp4': 511 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Nationals2_2.mp4': 1471 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Nationals2_3.mp4': 931 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Texas9_1.mp4': 10354 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Texas9_2.mp4': 5318 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Texas9_3.mp4': 6016 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Texas9_4.mp4': 7837 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko10_1.mp4': 3196 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko10_2.mp4': 4374 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko10_3.mp4': 3954 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko12_1.mp4': 815 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko12_2.mp4': 840 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko6_1.mp4': 3263 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko6_2.mp4': 2947 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko6_3.mp4': 3262 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko6_4.mp4': 2631 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko8_1.mp4': 5276 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko8_2.mp4': 4760 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Worlds2_1.mp4': 931 frames found.
    Loaded 148985 frames in loader.
    Distribution:
    {'images_.mp': 30400, 'Bart_.mp': 49898, 'Texas_.mp': 29525, 'Nationals_.mp': 2913, 'Worlds_.mp': 931, 'Tko_.mp': 35318}
    Creating transform: require
    Creating transform: zeros_like
    Creating transform: draw_polygon
    Creating transform: draw_lines
    Creating transform: draw_lines
    Creating transform: resize
    Creating transform: resize
    Creating transform: add_random_number
    Creating transform: randomize_frame
    Creating transform: randomize_frame
    Creating transform: normalize_channels
    Creating transform: normalize_channels
    Creating transform: random_video_loader
    Adding folder '.\..\..\Data\Raw\RawMatchesContinuous\'
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_1.mp4': 3430 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_2.mp4': 8916 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_3.mp4': 15947 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_4.mp4': 15260 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Bart9_5.mp4': 6345 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\images_0.mp4': 3040 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\images_0.mp4': Repeating 10.000000 times.
    '.\..\..\Data\Raw\RawMatchesContinuous\Nationals2_1.mp4': 511 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Nationals2_2.mp4': 1471 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Nationals2_3.mp4': 931 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Texas9_1.mp4': 10354 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Texas9_2.mp4': 5318 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Texas9_3.mp4': 6016 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Texas9_4.mp4': 7837 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko10_1.mp4': 3196 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko10_2.mp4': 4374 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko10_3.mp4': 3954 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko12_1.mp4': 815 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko12_2.mp4': 840 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko6_1.mp4': 3263 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko6_2.mp4': 2947 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko6_3.mp4': 3262 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko6_4.mp4': 2631 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko8_1.mp4': 5276 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Tko8_2.mp4': 4760 frames found.
    '.\..\..\Data\Raw\RawMatchesContinuous\Worlds2_1.mp4': 931 frames found.
    Loaded 148985 frames in loader.
    Distribution:
    {'images_.mp': 30400, 'Bart_.mp': 49898, 'Texas_.mp': 29525, 'Nationals_.mp': 2913, 'Worlds_.mp': 931, 'Tko_.mp': 35318}
    Creating transform: require
    Creating transform: zeros_like
    Creating transform: draw_polygon
    Creating transform: draw_lines
    Creating transform: draw_lines
    Creating transform: resize
    Creating transform: resize
    Creating transform: add_random_number
    Creating transform: randomize_frame
    Creating transform: randomize_frame
    Creating transform: normalize_channels
    Creating transform: normalize_channels
    


```python
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline

def threshold(data, minimum=0.0, maximum=1.0):
    mask = data > maximum
    data[mask] = maximum

    mask = data < minimum
    data[mask] = minimum
    return data

# View the first few frames
for k in range(10):
    data = training.get_next_frame()
    print(data["frame"].shape)
    frames = np.concatenate((data["frame"],data["segmentation"]),1)
    print(frames.shape)
    fig, ax = plt.subplots(figsize=(24, 5))
    plt.imshow(threshold(frames))
    plt.show()
```

    (256, 512, 3)
    (256, 1024, 3)
    


![png](output_3_1.png)


    (256, 512, 3)
    (256, 1024, 3)
    


![png](output_3_3.png)


    (256, 512, 3)
    (256, 1024, 3)
    


![png](output_3_5.png)


    (256, 512, 3)
    (256, 1024, 3)
    


![png](output_3_7.png)


    (256, 512, 3)
    (256, 1024, 3)
    


![png](output_3_9.png)


    (256, 512, 3)
    (256, 1024, 3)
    


![png](output_3_11.png)


    (256, 512, 3)
    (256, 1024, 3)
    


![png](output_3_13.png)


    (256, 512, 3)
    (256, 1024, 3)
    

    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    

    (256, 512, 3)
    (256, 1024, 3)
    


![png](output_3_17.png)



```python
# https://stanford.edu/~shervine/blog/keras-generator-multiprocessing.html
class threadsafe_iter(object):
  """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
  def __init__(self, it):
      self.it = it
      self.lock = threading.Lock()

  def __iter__(self):
      return self

  def __next__(self):
      with self.lock:
          return self.it.__next__()

# https://stanford.edu/~shervine/blog/keras-generator-multiprocessing.html
def threadsafe_generator(f):
  """
    A decorator that takes a generator function and makes it thread-safe.
    """
  def g(*a, **kw):
      return threadsafe_iter(f(*a, **kw))
  return g


# Define our training and validation iterators
@threadsafe_generator
def TrainGen(training):
    while True:
        data = training.get_next_frame()
        if data is not None:
            frame = data["frame"]
            segmentation = data["segmentation"]
            
            while frame is not None:
                yield (frame, segmentation)
                data = training.get_next_frame()
                if data is not None:
                    frame = data["frame"]
                    segmentation = data["segmentation"]
                else:
                    frame = None
                    segmentation = None
                
             
# Generators for training the position
@threadsafe_generator
def TrainBatchGen(batch_size, training):
    gen = TrainGen(training)
    while True:
        # Build the next batch
        batch_frames = np.zeros(shape=(batch_size, 256, 512, 3), dtype=np.float32)
        batch_segmentations = np.zeros(shape=(batch_size, 256, 512, 3), dtype=np.float32)
        for i in range(batch_size):
            (frame, segmentation) = next(gen)
            batch_frames[i,:,:,:] = frame
            batch_segmentations[i,:,:,:] = segmentation
            batch_segmentations[i,:,:,0] = batch_segmentations[i,:,:,0] * 5
            batch_segmentations[i,:,:,1] = batch_segmentations[i,:,:,1] * 5
            
        
        #pp.pprint("Yielding batch")
        #pp.pprint(batch_outputs)
        yield (batch_frames, batch_segmentations)
        #pp.pprint("Yielded batch")

    
    
        
    

```


```python
from keras.losses import mean_squared_error



# Utilities for plotting the result and training
def plot_validate(generator, model, count, name):   
    errors_true = []
    errors_prediction = []
    
    
    while len(errors_true) < count:
        
        (frames, outputs_true) = next(generator)
        outputs_predicted = model.predict(frames, batch_size=frames[0].shape[0], verbose=0)
        
        for i in range(outputs_predicted.shape[0]):
            if len(errors_true) < 5:
                # View the first set
                inputs = np.concatenate((frames[i], outputs_true[i], outputs_predicted[i]),1)
                
                fig, ax = plt.subplots(figsize=(45, 8))
                h = plt.imshow(threshold(inputs))
                plt.title( 'Frame, True Segmentation, Predicted Segmentation' )
                plt.show()
            
            # Calculate the actual difference and predicted image difference
            mse_predicted = ((outputs_true[i] - outputs_predicted[i]) ** 2).mean()
            #mse_true = K.eval(K.mean(mean_squared_error(outputs_true[i], frames[1][i,6:,16:])))
            #mse_predicted = K.eval(K.mean(mean_squared_error(outputs_true[i], outputs_predicted[i])))
            errors_prediction.append(mse_predicted)
            errors_true.append(mse_predicted)
    
    
    true, predicted = zip(*sorted(zip(errors_true, errors_prediction)))
    l1, = plt.plot(range(len(errors_true)), true, label="Error baseline by assuming no changes")
    l2, = plt.plot(range(len(errors_true)), predicted, label="Predicted error to next frame")
    plt.legend(handles=[l1, l2])
    plt.show()

#plot_validate(TrainBatchGen(batch_size, validation), frame_prediction_model, 5, "Epoch validation results %i" % epoch)


#def mean_squared_error_weighted(y_true, y_pred):
#    return K.mean(K.square(y_pred[:,:,:,1] - y_true[:,:,:,1]), axis=-1)

```


```python
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
#from IPython.display import SVG, display
#from keras.utils.vis_utils import model_to_dot

import gc
K.clear_session()
gc.collect()


image_height       = 256
image_width        = 512
image_channels     = 3

# Model options
batch_size = 10
lstm_output_size = 300
cnn_kernel_count = 40

# Build an hourglass model. Notes on the model structure:
# Two camera frames as input.
# frame0 -> Conv and pooling until embedding (shared convoluations to both frame0 and frame1)
# frame1 -> Conv and pooling until embedding
# Conv both paths together
# Residual upsampling of frame1 until original size
# Pixel-wise error to frame2 as loss

pp.pprint("Input shape without batches:")
pp.pprint((image_height, image_width, image_channels))

# Used to give fixed names to the layers for transferring the model
conv_num = 0 
pool_num = 0
dense_num = 0


# Create the embedding module, the same model runs both frames
frame_in = Input(shape=(image_height, image_width, image_channels,))
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(frame_in)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
res0 = x

x = MaxPooling2D((1, 2))(x)

x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
res1 = x

x = MaxPooling2D((2, 2))(x)



x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
res2 = x
x = MaxPooling2D((2, 2))(x)



x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
res3 = x
x = MaxPooling2D((2, 2))(x)



x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
res4 = x
x = MaxPooling2D((2, 2))(x)



x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
res5 = x
x = MaxPooling2D((2, 2))(x)


# Start the upsampling and residual joining

x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)

# Create the upsampling stages, feeding residual from frame1
x = UpSampling2D(size=(2, 2), data_format=None)(x)
#res5 = Cropping2D(cropping=(((0, 0), (1, 0))), data_format=None)(res4_b)
x = keras.layers.concatenate([x, res5])
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)

x = UpSampling2D(size=(2, 2), data_format=None)(x)
#res5 = Cropping2D(cropping=(((0, 0), (1, 0))), data_format=None)(res4_b)
x = keras.layers.concatenate([x, res4])
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)

x = UpSampling2D(size=(2, 2), data_format=None)(x)
#res5 = Cropping2D(cropping=(((0, 0), (1, 0))), data_format=None)(res4_b)
x = keras.layers.concatenate([x, res3])
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)

x = UpSampling2D(size=(2, 2), data_format=None)(x)
#res5 = Cropping2D(cropping=(((0, 0), (1, 0))), data_format=None)(res4_b)
x = keras.layers.concatenate([x, res2])
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)

x = UpSampling2D(size=(2, 2), data_format=None)(x)
#res5 = Cropping2D(cropping=(((0, 0), (1, 0))), data_format=None)(res4_b)
x = keras.layers.concatenate([x, res1])
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)

x = UpSampling2D(size=(1, 2), data_format=None)(x)
#res5 = Cropping2D(cropping=(((0, 0), (1, 0))), data_format=None)(res4_b)
x = keras.layers.concatenate([x, res0])
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)
x = Conv2D(cnn_kernel_count, (3, 3),
           padding = "same",
           activation = "relu",)(x)

# RGB output as three channels
x = Conv2D(3, (1, 1),
           padding = "same",
           activation = "relu",)(x)

frame_prediction_model = Model([frame_in], [x])

# For a multi-class classification problem
frame_prediction_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00002),
#model.compile(optimizer=keras.optimizers.adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


frame_prediction_model.summary()

#plot_model(frame_prediction_model, to_file='model.png')
#from IPython.display import SVG, display
#from keras.utils.vis_utils import model_to_dot
#display(SVG(model_to_dot(frame_prediction_model).create(prog='dot', format='svg')))

# Train the model to predict the future position. This is the control signal to the robot AI
WEIGHTS_FNAME = '.\\Goalie3Frames\\weights_%i.hdf'
MODELS_FNAME = '.\\Goalie3Frames\\models_%i.h5'
```

    'Input shape without batches:'
    (256, 512, 3)
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_1 (InputLayer)             (None, 256, 512, 3)   0                                            
    ____________________________________________________________________________________________________
    conv2d_1 (Conv2D)                (None, 256, 512, 40)  1120        input_1[0][0]                    
    ____________________________________________________________________________________________________
    conv2d_2 (Conv2D)                (None, 256, 512, 40)  14440       conv2d_1[0][0]                   
    ____________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)   (None, 256, 256, 40)  0           conv2d_2[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_3 (Conv2D)                (None, 256, 256, 40)  14440       max_pooling2d_1[0][0]            
    ____________________________________________________________________________________________________
    conv2d_4 (Conv2D)                (None, 256, 256, 40)  14440       conv2d_3[0][0]                   
    ____________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)   (None, 128, 128, 40)  0           conv2d_4[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_5 (Conv2D)                (None, 128, 128, 40)  14440       max_pooling2d_2[0][0]            
    ____________________________________________________________________________________________________
    conv2d_6 (Conv2D)                (None, 128, 128, 40)  14440       conv2d_5[0][0]                   
    ____________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)   (None, 64, 64, 40)    0           conv2d_6[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_7 (Conv2D)                (None, 64, 64, 40)    14440       max_pooling2d_3[0][0]            
    ____________________________________________________________________________________________________
    conv2d_8 (Conv2D)                (None, 64, 64, 40)    14440       conv2d_7[0][0]                   
    ____________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)   (None, 32, 32, 40)    0           conv2d_8[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_9 (Conv2D)                (None, 32, 32, 40)    14440       max_pooling2d_4[0][0]            
    ____________________________________________________________________________________________________
    conv2d_10 (Conv2D)               (None, 32, 32, 40)    14440       conv2d_9[0][0]                   
    ____________________________________________________________________________________________________
    max_pooling2d_5 (MaxPooling2D)   (None, 16, 16, 40)    0           conv2d_10[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_11 (Conv2D)               (None, 16, 16, 40)    14440       max_pooling2d_5[0][0]            
    ____________________________________________________________________________________________________
    conv2d_12 (Conv2D)               (None, 16, 16, 40)    14440       conv2d_11[0][0]                  
    ____________________________________________________________________________________________________
    max_pooling2d_6 (MaxPooling2D)   (None, 8, 8, 40)      0           conv2d_12[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_13 (Conv2D)               (None, 8, 8, 40)      14440       max_pooling2d_6[0][0]            
    ____________________________________________________________________________________________________
    conv2d_14 (Conv2D)               (None, 8, 8, 40)      14440       conv2d_13[0][0]                  
    ____________________________________________________________________________________________________
    up_sampling2d_1 (UpSampling2D)   (None, 16, 16, 40)    0           conv2d_14[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_1 (Concatenate)      (None, 16, 16, 80)    0           up_sampling2d_1[0][0]            
                                                                       conv2d_12[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_15 (Conv2D)               (None, 16, 16, 40)    28840       concatenate_1[0][0]              
    ____________________________________________________________________________________________________
    conv2d_16 (Conv2D)               (None, 16, 16, 40)    14440       conv2d_15[0][0]                  
    ____________________________________________________________________________________________________
    up_sampling2d_2 (UpSampling2D)   (None, 32, 32, 40)    0           conv2d_16[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_2 (Concatenate)      (None, 32, 32, 80)    0           up_sampling2d_2[0][0]            
                                                                       conv2d_10[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_17 (Conv2D)               (None, 32, 32, 40)    28840       concatenate_2[0][0]              
    ____________________________________________________________________________________________________
    conv2d_18 (Conv2D)               (None, 32, 32, 40)    14440       conv2d_17[0][0]                  
    ____________________________________________________________________________________________________
    up_sampling2d_3 (UpSampling2D)   (None, 64, 64, 40)    0           conv2d_18[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_3 (Concatenate)      (None, 64, 64, 80)    0           up_sampling2d_3[0][0]            
                                                                       conv2d_8[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_19 (Conv2D)               (None, 64, 64, 40)    28840       concatenate_3[0][0]              
    ____________________________________________________________________________________________________
    conv2d_20 (Conv2D)               (None, 64, 64, 40)    14440       conv2d_19[0][0]                  
    ____________________________________________________________________________________________________
    up_sampling2d_4 (UpSampling2D)   (None, 128, 128, 40)  0           conv2d_20[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_4 (Concatenate)      (None, 128, 128, 80)  0           up_sampling2d_4[0][0]            
                                                                       conv2d_6[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_21 (Conv2D)               (None, 128, 128, 40)  28840       concatenate_4[0][0]              
    ____________________________________________________________________________________________________
    conv2d_22 (Conv2D)               (None, 128, 128, 40)  14440       conv2d_21[0][0]                  
    ____________________________________________________________________________________________________
    up_sampling2d_5 (UpSampling2D)   (None, 256, 256, 40)  0           conv2d_22[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_5 (Concatenate)      (None, 256, 256, 80)  0           up_sampling2d_5[0][0]            
                                                                       conv2d_4[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_23 (Conv2D)               (None, 256, 256, 40)  28840       concatenate_5[0][0]              
    ____________________________________________________________________________________________________
    conv2d_24 (Conv2D)               (None, 256, 256, 40)  14440       conv2d_23[0][0]                  
    ____________________________________________________________________________________________________
    up_sampling2d_6 (UpSampling2D)   (None, 256, 512, 40)  0           conv2d_24[0][0]                  
    ____________________________________________________________________________________________________
    concatenate_6 (Concatenate)      (None, 256, 512, 80)  0           up_sampling2d_6[0][0]            
                                                                       conv2d_2[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_25 (Conv2D)               (None, 256, 512, 40)  28840       concatenate_6[0][0]              
    ____________________________________________________________________________________________________
    conv2d_26 (Conv2D)               (None, 256, 512, 40)  14440       conv2d_25[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_27 (Conv2D)               (None, 256, 512, 3)   123         conv2d_26[0][0]                  
    ====================================================================================================
    Total params: 448,643
    Trainable params: 448,643
    Non-trainable params: 0
    ____________________________________________________________________________________________________
    


```python
# For a multi-class classification problem
frame_prediction_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00005),
#model.compile(optimizer=keras.optimizers.adam(),
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Was ran on the above simple model, not pre-trained Inceptionv3 run.
epoch = 0
#batches_validation_per_epoch = 50
#batches_training_per_epoch = 400
batch_size = 20
batches_training_per_epoch = 200
batches_validation_per_epoch = 40
print("Batch size %i: %i training batches, %i validation batches" % (batch_size, batches_training_per_epoch, batches_validation_per_epoch) )
WEIGHTS_FNAME = '.\\Models\\weights_%i.hdf'
MODELS_FNAME = '.\\Models\\models_%i.h5'

for i in range(100):
    frame_prediction_model.fit_generator(
        TrainBatchGen(batch_size, training=training),
        batches_training_per_epoch,
        epochs=epoch+1,
        verbose=1,
        callbacks=None,
        class_weight=None,
        max_q_size=50,
        workers=50,
        validation_data=TrainBatchGen(batch_size, validation),
        validation_steps = batches_validation_per_epoch,
        pickle_safe=False,
        initial_epoch=epoch)
    
    epoch += 1
    
    # Plot occasional validation data plot
    if i % 5 == 0:
        plot_validate(TrainBatchGen(batch_size, validation), frame_prediction_model, 50, "Epoch validation results %i" % epoch)
    
    # Save the model
    frame_prediction_model.save_weights(WEIGHTS_FNAME % epoch)
    frame_prediction_model.save(MODELS_FNAME % epoch)
    print(("Wrote model to " + WEIGHTS_FNAME )  % epoch)


```

    Batch size 20: 200 training batches, 40 validation batches
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=0, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=1)`
    

    Epoch 1/1
    200/200 [==============================] - 801s - loss: 0.3753 - mean_squared_error: 0.3753 - val_loss: 0.2601 - val_mean_squared_error: 0.2601
    


![png](output_7_3.png)



![png](output_7_4.png)



![png](output_7_5.png)



![png](output_7_6.png)



![png](output_7_7.png)



![png](output_7_8.png)


    Wrote model to .\Models\weights_1.hdf
    Epoch 2/2
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=1, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=2)`
    

    200/200 [==============================] - 800s - loss: 0.1880 - mean_squared_error: 0.1880 - val_loss: 0.1684 - val_mean_squared_error: 0.1684
    Wrote model to .\Models\weights_2.hdf
    Epoch 3/3
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=2, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=3)`
    

    200/200 [==============================] - 798s - loss: 0.1127 - mean_squared_error: 0.1127 - val_loss: 0.0990 - val_mean_squared_error: 0.0990
    Wrote model to .\Models\weights_3.hdf
    Epoch 4/4
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=3, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=4)`
    

    200/200 [==============================] - 801s - loss: 0.0853 - mean_squared_error: 0.0853 - val_loss: 0.0853 - val_mean_squared_error: 0.0853
    Wrote model to .\Models\weights_4.hdf
    Epoch 5/5
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=4, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=5)`
    

    200/200 [==============================] - 794s - loss: 0.0739 - mean_squared_error: 0.0739 - val_loss: 0.0728 - val_mean_squared_error: 0.0728
    Wrote model to .\Models\weights_5.hdf
    Epoch 6/6
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=5, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=6)`
    

    200/200 [==============================] - 799s - loss: 0.0675 - mean_squared_error: 0.0675 - val_loss: 0.0613 - val_mean_squared_error: 0.0613
    


![png](output_7_20.png)



![png](output_7_21.png)



![png](output_7_22.png)



![png](output_7_23.png)



![png](output_7_24.png)



![png](output_7_25.png)


    Wrote model to .\Models\weights_6.hdf
    Epoch 7/7
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=6, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=7)`
    

    200/200 [==============================] - 796s - loss: 0.0629 - mean_squared_error: 0.0629 - val_loss: 0.0671 - val_mean_squared_error: 0.0671
    Wrote model to .\Models\weights_7.hdf
    Epoch 8/8
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=7, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=8)`
    

    200/200 [==============================] - 797s - loss: 0.0564 - mean_squared_error: 0.0564 - val_loss: 0.0515 - val_mean_squared_error: 0.0515
    Wrote model to .\Models\weights_8.hdf
    Epoch 9/9
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=8, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=9)`
    

    200/200 [==============================] - 795s - loss: 0.0527 - mean_squared_error: 0.0527 - val_loss: 0.0502 - val_mean_squared_error: 0.0502
    Wrote model to .\Models\weights_9.hdf
    Epoch 10/10
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=9, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=10)`
    

    200/200 [==============================] - 799s - loss: 0.0507 - mean_squared_error: 0.0507 - val_loss: 0.0460 - val_mean_squared_error: 0.0460
    Wrote model to .\Models\weights_10.hdf
    Epoch 11/11
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=10, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=11)`
    

    200/200 [==============================] - 797s - loss: 0.0470 - mean_squared_error: 0.0470 - val_loss: 0.0496 - val_mean_squared_error: 0.0496
    


![png](output_7_37.png)



![png](output_7_38.png)



![png](output_7_39.png)



![png](output_7_40.png)



![png](output_7_41.png)



![png](output_7_42.png)


    Wrote model to .\Models\weights_11.hdf
    Epoch 12/12
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=11, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=12)`
    

    200/200 [==============================] - 796s - loss: 0.0472 - mean_squared_error: 0.0472 - val_loss: 0.0411 - val_mean_squared_error: 0.0411
    Wrote model to .\Models\weights_12.hdf
    Epoch 13/13
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=12, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=13)`
    

    200/200 [==============================] - 797s - loss: 0.0442 - mean_squared_error: 0.0442 - val_loss: 0.0423 - val_mean_squared_error: 0.0423
    Wrote model to .\Models\weights_13.hdf
    Epoch 14/14
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=13, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=14)`
    

    200/200 [==============================] - 798s - loss: 0.0423 - mean_squared_error: 0.0423 - val_loss: 0.0398 - val_mean_squared_error: 0.0398
    Wrote model to .\Models\weights_14.hdf
    Epoch 15/15
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=14, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=15)`
    

    200/200 [==============================] - 795s - loss: 0.0397 - mean_squared_error: 0.0397 - val_loss: 0.0384 - val_mean_squared_error: 0.0384
    Wrote model to .\Models\weights_15.hdf
    Epoch 16/16
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=15, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=16)`
    

    200/200 [==============================] - 795s - loss: 0.0395 - mean_squared_error: 0.0395 - val_loss: 0.0345 - val_mean_squared_error: 0.0345
    


![png](output_7_54.png)



![png](output_7_55.png)



![png](output_7_56.png)



![png](output_7_57.png)



![png](output_7_58.png)



![png](output_7_59.png)


    Wrote model to .\Models\weights_16.hdf
    Epoch 17/17
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=16, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=17)`
    

    200/200 [==============================] - 794s - loss: 0.0387 - mean_squared_error: 0.0387 - val_loss: 0.0370 - val_mean_squared_error: 0.0370
    Wrote model to .\Models\weights_17.hdf
    Epoch 18/18
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=17, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=18)`
    

    200/200 [==============================] - 796s - loss: 0.0393 - mean_squared_error: 0.0393 - val_loss: 0.0346 - val_mean_squared_error: 0.0346
    Wrote model to .\Models\weights_18.hdf
    Epoch 19/19
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=18, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=19)`
    

    200/200 [==============================] - 797s - loss: 0.0359 - mean_squared_error: 0.0359 - val_loss: 0.0349 - val_mean_squared_error: 0.0349
    Wrote model to .\Models\weights_19.hdf
    Epoch 20/20
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=19, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=20)`
    

    200/200 [==============================] - 792s - loss: 0.0356 - mean_squared_error: 0.0356 - val_loss: 0.0341 - val_mean_squared_error: 0.0341
    Wrote model to .\Models\weights_20.hdf
    Epoch 21/21
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=20, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=21)`
    

    200/200 [==============================] - 793s - loss: 0.0353 - mean_squared_error: 0.0353 - val_loss: 0.0390 - val_mean_squared_error: 0.0390
    


![png](output_7_71.png)



![png](output_7_72.png)



![png](output_7_73.png)



![png](output_7_74.png)



![png](output_7_75.png)



![png](output_7_76.png)


    Wrote model to .\Models\weights_21.hdf
    Epoch 22/22
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=21, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=22)`
    

    200/200 [==============================] - 797s - loss: 0.0344 - mean_squared_error: 0.0344 - val_loss: 0.0321 - val_mean_squared_error: 0.0321
    Wrote model to .\Models\weights_22.hdf
    Epoch 23/23
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=22, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=23)`
    

    200/200 [==============================] - 795s - loss: 0.0336 - mean_squared_error: 0.0336 - val_loss: 0.0366 - val_mean_squared_error: 0.0366
    Wrote model to .\Models\weights_23.hdf
    Epoch 24/24
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=23, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=24)`
    

    200/200 [==============================] - 794s - loss: 0.0329 - mean_squared_error: 0.0329 - val_loss: 0.0306 - val_mean_squared_error: 0.0306
    Wrote model to .\Models\weights_24.hdf
    Epoch 25/25
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=24, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=25)`
    

    200/200 [==============================] - 799s - loss: 0.0330 - mean_squared_error: 0.0330 - val_loss: 0.0415 - val_mean_squared_error: 0.0415
    Wrote model to .\Models\weights_25.hdf
    Epoch 26/26
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=25, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=26)`
    

    200/200 [==============================] - 795s - loss: 0.0315 - mean_squared_error: 0.0315 - val_loss: 0.0299 - val_mean_squared_error: 0.0299
    


![png](output_7_88.png)



![png](output_7_89.png)



![png](output_7_90.png)



![png](output_7_91.png)



![png](output_7_92.png)



![png](output_7_93.png)


    Wrote model to .\Models\weights_26.hdf
    Epoch 27/27
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=26, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=27)`
    

    200/200 [==============================] - 794s - loss: 0.0320 - mean_squared_error: 0.0320 - val_loss: 0.0310 - val_mean_squared_error: 0.0310
    Wrote model to .\Models\weights_27.hdf
    Epoch 28/28
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=27, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=28)`
    

    200/200 [==============================] - 792s - loss: 0.0309 - mean_squared_error: 0.0309 - val_loss: 0.0286 - val_mean_squared_error: 0.0286
    Wrote model to .\Models\weights_28.hdf
    Epoch 29/29
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=28, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=29)`
    

    200/200 [==============================] - 796s - loss: 0.0302 - mean_squared_error: 0.0302 - val_loss: 0.0357 - val_mean_squared_error: 0.0357
    Wrote model to .\Models\weights_29.hdf
    Epoch 30/30
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=29, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=30)`
    

    200/200 [==============================] - 793s - loss: 0.0295 - mean_squared_error: 0.0295 - val_loss: 0.0317 - val_mean_squared_error: 0.0317
    Wrote model to .\Models\weights_30.hdf
    Epoch 31/31
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=30, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=31)`
    

    200/200 [==============================] - 793s - loss: 0.0303 - mean_squared_error: 0.0303 - val_loss: 0.0303 - val_mean_squared_error: 0.0303
    


![png](output_7_105.png)



![png](output_7_106.png)



![png](output_7_107.png)



![png](output_7_108.png)



![png](output_7_109.png)



![png](output_7_110.png)


    Wrote model to .\Models\weights_31.hdf
    Epoch 32/32
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=31, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=32)`
    

    200/200 [==============================] - 794s - loss: 0.0306 - mean_squared_error: 0.0306 - val_loss: 0.0274 - val_mean_squared_error: 0.0274
    Wrote model to .\Models\weights_32.hdf
    Epoch 33/33
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=32, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=33)`
    

    200/200 [==============================] - 793s - loss: 0.0291 - mean_squared_error: 0.0291 - val_loss: 0.0272 - val_mean_squared_error: 0.0272
    Wrote model to .\Models\weights_33.hdf
    Epoch 34/34
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=33, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=34)`
    

    200/200 [==============================] - 794s - loss: 0.0294 - mean_squared_error: 0.0294 - val_loss: 0.0289 - val_mean_squared_error: 0.0289
    Wrote model to .\Models\weights_34.hdf
    Epoch 35/35
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=34, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=35)`
    

    200/200 [==============================] - 793s - loss: 0.0290 - mean_squared_error: 0.0290 - val_loss: 0.0287 - val_mean_squared_error: 0.0287
    Wrote model to .\Models\weights_35.hdf
    Epoch 36/36
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=35, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=36)`
    

    200/200 [==============================] - 794s - loss: 0.0291 - mean_squared_error: 0.0291 - val_loss: 0.0291 - val_mean_squared_error: 0.0291
    


![png](output_7_122.png)



![png](output_7_123.png)



![png](output_7_124.png)



![png](output_7_125.png)



![png](output_7_126.png)



![png](output_7_127.png)


    Wrote model to .\Models\weights_36.hdf
    Epoch 37/37
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=36, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=37)`
    

    200/200 [==============================] - 796s - loss: 0.0290 - mean_squared_error: 0.0290 - val_loss: 0.0309 - val_mean_squared_error: 0.0309
    Wrote model to .\Models\weights_37.hdf
    Epoch 38/38
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=37, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=38)`
    

    200/200 [==============================] - 792s - loss: 0.0276 - mean_squared_error: 0.0276 - val_loss: 0.0270 - val_mean_squared_error: 0.0270
    Wrote model to .\Models\weights_38.hdf
    Epoch 39/39
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=38, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=39)`
    

    200/200 [==============================] - 790s - loss: 0.0283 - mean_squared_error: 0.0283 - val_loss: 0.1222 - val_mean_squared_error: 0.1222
    Wrote model to .\Models\weights_39.hdf
    Epoch 40/40
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=39, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=40)`
    

    200/200 [==============================] - 794s - loss: 0.0273 - mean_squared_error: 0.0273 - val_loss: 0.0285 - val_mean_squared_error: 0.0285
    Wrote model to .\Models\weights_40.hdf
    Epoch 41/41
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=40, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=41)`
    

    200/200 [==============================] - 793s - loss: 0.0280 - mean_squared_error: 0.0280 - val_loss: 0.0253 - val_mean_squared_error: 0.0253
    


![png](output_7_139.png)



![png](output_7_140.png)



![png](output_7_141.png)



![png](output_7_142.png)



![png](output_7_143.png)



![png](output_7_144.png)


    Wrote model to .\Models\weights_41.hdf
    Epoch 42/42
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=41, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=42)`
    

    200/200 [==============================] - 794s - loss: 0.0273 - mean_squared_error: 0.0273 - val_loss: 0.0290 - val_mean_squared_error: 0.0290
    Wrote model to .\Models\weights_42.hdf
    Epoch 43/43
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=42, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=43)`
    

    200/200 [==============================] - 791s - loss: 0.0278 - mean_squared_error: 0.0278 - val_loss: 0.0306 - val_mean_squared_error: 0.0306
    Wrote model to .\Models\weights_43.hdf
    Epoch 44/44
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=43, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=44)`
    

    200/200 [==============================] - 794s - loss: 0.0266 - mean_squared_error: 0.0266 - val_loss: 0.0254 - val_mean_squared_error: 0.0254
    Wrote model to .\Models\weights_44.hdf
    Epoch 45/45
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=44, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=45)`
    

    200/200 [==============================] - 795s - loss: 0.0272 - mean_squared_error: 0.0272 - val_loss: 0.0268 - val_mean_squared_error: 0.0268
    Wrote model to .\Models\weights_45.hdf
    Epoch 46/46
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=45, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=46)`
    

    200/200 [==============================] - 794s - loss: 0.0268 - mean_squared_error: 0.0268 - val_loss: 0.0233 - val_mean_squared_error: 0.0233
    


![png](output_7_156.png)



![png](output_7_157.png)



![png](output_7_158.png)



![png](output_7_159.png)



![png](output_7_160.png)



![png](output_7_161.png)


    Wrote model to .\Models\weights_46.hdf
    Epoch 47/47
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=46, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=47)`
    

    200/200 [==============================] - 792s - loss: 0.0267 - mean_squared_error: 0.0267 - val_loss: 0.0248 - val_mean_squared_error: 0.0248
    Wrote model to .\Models\weights_47.hdf
    Epoch 48/48
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=47, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=48)`
    

    200/200 [==============================] - 792s - loss: 0.0261 - mean_squared_error: 0.0261 - val_loss: 0.0266 - val_mean_squared_error: 0.0266
    Wrote model to .\Models\weights_48.hdf
    Epoch 49/49
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=48, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=49)`
    

    200/200 [==============================] - 791s - loss: 0.0257 - mean_squared_error: 0.0257 - val_loss: 0.0229 - val_mean_squared_error: 0.0229
    Wrote model to .\Models\weights_49.hdf
    Epoch 50/50
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=49, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=50)`
    

    200/200 [==============================] - 795s - loss: 0.0264 - mean_squared_error: 0.0264 - val_loss: 0.0249 - val_mean_squared_error: 0.0249
    Wrote model to .\Models\weights_50.hdf
    Epoch 51/51
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=50, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=51)`
    

    200/200 [==============================] - 792s - loss: 0.0249 - mean_squared_error: 0.0249 - val_loss: 0.0249 - val_mean_squared_error: 0.0249
    


![png](output_7_173.png)



![png](output_7_174.png)



![png](output_7_175.png)



![png](output_7_176.png)



![png](output_7_177.png)



![png](output_7_178.png)


    Wrote model to .\Models\weights_51.hdf
    Epoch 52/52
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=51, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=52)`
    

    200/200 [==============================] - 796s - loss: 0.0264 - mean_squared_error: 0.0264 - val_loss: 0.0268 - val_mean_squared_error: 0.0268
    Wrote model to .\Models\weights_52.hdf
    Epoch 53/53
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=52, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=53)`
    

    200/200 [==============================] - 797s - loss: 0.0251 - mean_squared_error: 0.0251 - val_loss: 0.0250 - val_mean_squared_error: 0.0250
    Wrote model to .\Models\weights_53.hdf
    Epoch 54/54
    

    C:\Anaconda\envs\py35\lib\site-packages\ipykernel\__main__.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<__main__...., 200, callbacks=None, verbose=1, use_multiprocessing=False, initial_epoch=53, validation_data=<__main__...., workers=50, validation_steps=40, max_queue_size=50, class_weight=None, epochs=54)`
    

    173/200 [========================>.....] - ETA: 81s - loss: 0.0244 - mean_squared_error: 0.0244


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    C:\Anaconda\envs\py35\lib\site-packages\keras\engine\training.py in fit_generator(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
       2041                                                sample_weight=sample_weight,
    -> 2042                                                class_weight=class_weight)
       2043 
    

    C:\Anaconda\envs\py35\lib\site-packages\keras\engine\training.py in train_on_batch(self, x, y, sample_weight, class_weight)
       1761         self._make_train_function()
    -> 1762         outputs = self.train_function(ins)
       1763         if len(outputs) == 1:
    

    C:\Anaconda\envs\py35\lib\site-packages\keras\backend\tensorflow_backend.py in __call__(self, inputs)
       2272                               feed_dict=feed_dict,
    -> 2273                               **self.session_kwargs)
       2274         return updated[:len(self.outputs)]
    

    C:\Anaconda\envs\py35\lib\site-packages\tensorflow\python\client\session.py in run(self, fetches, feed_dict, options, run_metadata)
        894       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 895                          run_metadata_ptr)
        896       if run_metadata:
    

    C:\Anaconda\envs\py35\lib\site-packages\tensorflow\python\client\session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
       1123       results = self._do_run(handle, final_targets, final_fetches,
    -> 1124                              feed_dict_tensor, options, run_metadata)
       1125     else:
    

    C:\Anaconda\envs\py35\lib\site-packages\tensorflow\python\client\session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1320       return self._do_call(_run_fn, self._session, feeds, fetches, targets,
    -> 1321                            options, run_metadata)
       1322     else:
    

    C:\Anaconda\envs\py35\lib\site-packages\tensorflow\python\client\session.py in _do_call(self, fn, *args)
       1326     try:
    -> 1327       return fn(*args)
       1328     except errors.OpError as e:
    

    C:\Anaconda\envs\py35\lib\site-packages\tensorflow\python\client\session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
       1305                                    feed_dict, fetch_list, target_list,
    -> 1306                                    status, run_metadata)
       1307 
    

    KeyboardInterrupt: 

    
    During handling of the above exception, another exception occurred:
    

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-8-f158bf0d57f8> in <module>()
         29         validation_steps = batches_validation_per_epoch,
         30         pickle_safe=False,
    ---> 31         initial_epoch=epoch)
         32 
         33     epoch += 1
    

    C:\Anaconda\envs\py35\lib\site-packages\keras\legacy\interfaces.py in wrapper(*args, **kwargs)
         85                 warnings.warn('Update your `' + object_name +
         86                               '` call to the Keras 2 API: ' + signature, stacklevel=2)
    ---> 87             return func(*args, **kwargs)
         88         wrapper._original_function = func
         89         return wrapper
    

    C:\Anaconda\envs\py35\lib\site-packages\keras\engine\training.py in fit_generator(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
       2087         finally:
       2088             if enqueuer is not None:
    -> 2089                 enqueuer.stop()
       2090 
       2091         callbacks.on_train_end()
    

    C:\Anaconda\envs\py35\lib\site-packages\keras\utils\data_utils.py in stop(self, timeout)
        618                     thread.terminate()
        619                 else:
    --> 620                     thread.join(timeout)
        621 
        622         if self._use_multiprocessing:
    

    C:\Anaconda\envs\py35\lib\threading.py in join(self, timeout)
       1052 
       1053         if timeout is None:
    -> 1054             self._wait_for_tstate_lock()
       1055         else:
       1056             # the behavior of a negative timeout isn't documented, but
    

    C:\Anaconda\envs\py35\lib\threading.py in _wait_for_tstate_lock(self, block, timeout)
       1068         if lock is None:  # already determined that the C code is done
       1069             assert self._is_stopped
    -> 1070         elif lock.acquire(block, timeout):
       1071             lock.release()
       1072             self._stop()
    

    KeyboardInterrupt: 

