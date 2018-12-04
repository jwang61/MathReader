# MathReader

Extract and classify numbers and math operators hand-written on a template document.
Train a convolutional neural network using TensorFlow on Python, and process images using OpenCV.

# Requirements

- Numpy >= 1.15.2
- OpenCV >= 3.4
- TensorFlow >= 1.11.0

# Scripts

The `train` folder contains the scripts to train the CNN: 
- `imageresize.py` is used to batch resize all images and save it to a new location.
- `createdata.py` is used to compile all the images into a single .pkl file, making future loading easier.
- `trainmodel.py` trains a image classification model using tensorflow and outputs a frozen protobuf.
- `removedropout.py` manually removes the dropout nodes in the model, as they are not used for inference.

To read an image of the template document, run the following command:
```
python3 read_img.py _imagename_ --model _frozenmodel_
```

# Other
A sample template `template.png` and sample image with numbers `captured_img.jpg` can be found in the directory.
Possible data sets to train on include the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and a kaggle dataset of handwritten numbers [here](https://www.kaggle.com/xainano/handwrittenmathsymbols)
