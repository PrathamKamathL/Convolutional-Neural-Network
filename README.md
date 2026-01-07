Convolutional Neural Network using MNIST Dataset

This project demonstrates the basic working of a Convolutional Neural Network (CNN) using TensorFlow and Keras. The model is trained on the MNIST dataset, which consists of handwritten digit images from 0 to 9. The goal of this project is to understand how CNNs work both conceptually and practically, including how convolution, pooling, and prediction are performed.

Project objective

The main objectives of this project are:
- To understand how images are processed by a CNN
- To learn how convolution and pooling work with simple mathematical intuition
- To implement a CNN model using TensorFlow and Keras
- To train and evaluate the model on the MNIST dataset
- To visualize how the trained model predicts a digit from an image

About the dataset

The MNIST dataset contains grayscale images of handwritten digits.
Each image is of size 28 by 28 pixels.
The dataset contains 60,000 training images and 10,000 test images.
The dataset is loaded directly using TensorFlow’s built-in MNIST loader.

CNN architecture used

The model follows a simple and standard CNN architecture.
First, the input image is passed through a convolution layer that learns basic features like edges and curves.
Then a max pooling layer reduces the image size while keeping important information.
Another convolution and pooling layer is used to learn more complex patterns.
After that, the feature maps are flattened and passed to fully connected dense layers.
Finally, a softmax layer predicts the probability of each digit from 0 to 9.

How convolution works

Convolution works by sliding a small matrix called a kernel over the image.
At each position, the kernel values are multiplied with the image values and summed.
This operation helps the network detect features such as edges and shapes.
The same kernel is applied across the entire image, which reduces the number of parameters.

How pooling works

Pooling reduces the size of the feature map.
In max pooling, the maximum value from a small region is selected.
This helps reduce computation and makes the model more robust to small changes in the image.
Pooling keeps important features while discarding unnecessary details.

How prediction is made

The final layer of the model outputs probabilities for each digit.
The digit with the highest probability is selected as the predicted output.
This is done using the argmax operation.

How to run the project

First, install the required libraries:
pip install tensorflow matplotlib

Then run the Python file:
python cnn_mnist.py

The script will train the model, evaluate it on test data, and display a sample image along with its predicted digit.

Results

The model achieves around 98 to 99 percent accuracy on the MNIST test dataset.
It performs well on unseen handwritten digits and generalizes effectively.

Who this project is for

This project is suitable for:
- Beginners learning deep learning
- Students studying machine learning or artificial intelligence
- Anyone who wants a clear and simple introduction to CNNs

Future improvements

The project can be extended by:
- Adding dropout and batch normalization
- Visualizing convolution filters
- Using more complex datasets like CIFAR-10
- Deploying the model as a web application

Conclusion

This project provides a clear and practical understanding of Convolutional Neural Networks. It focuses on simplicity and clarity, making it easy for beginners to learn how CNNs work and how they are implemented in real applications.
