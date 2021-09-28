# Instructions

There are 3 files with the extension `.npy` in this folder.
They can be read with `numpy`.

- `class_a.npy`: Sample images from class A
- `class_b.npy`: Sample images from class B
- `field.npy`: Sample images that are similar to the images that will be used for evaluation. Class labels are not provided.

Please write in python:

1. a training program that generates a model to be used by the classifier program,
2. a classifier program that uses the model above to classify images as class A or class B, and
3. an evaluation program that evaluates the performance of the classifier program above.

# Results

I trained my model using the following properties:

- Adadelta optimization algorithm.
- Learning rate 0.001.
- Batch size 128.
- 50 epochs.
- 50% dropout.

As we can see my simple CNN model was able to achieve an accuracy of 100%.