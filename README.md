# ResCNN: A novel CNN Implementation for bigger and more resilient batch processing and learning from Images.

*This work has been submitted to [UPCON 2021](http://upcon2021.in/) and is currently under review.*  
*Authors: [Avirup Dey](https://avirupju.github.io/) and [Sarosij Bose](https://sarosijbose.github.io/)*

## Description:-
Here we implement ResCNN, a Convolutional neural network featured as an viable alternative to traditional CNNs for Deep learning. We first convert the traditional convolution operation into a purely matrix based form which is based on the theory of [unrolling](https://hal.inria.fr/inria-00112631/document) convolution. Such an implementation allows us to employ some Image processing techniques mainly focusing on compression of images. One such technique used here is [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition), which is applied on entire dataset/Images and then fed into our convolutional network. We further experiment with various sets of hyperparameters to show that our network is able to learn at a much higher rate without collapsing like a standard CNN Model whose detailed comparisons are done below. Further, we also show that such improvement can be brought about without significant cost overheads or employment of GPUs. 

## Demo:-
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L2O6UziUIK2IlX7_kGxqPWAZtXA5kmw1?usp=sharing)

## Setup:-  

1. It is recommended to setup a fresh virtual environment first.  
```bash
python -m venv fastcnn
source activate env/bin/activate
```

2. Then install the required dependencies.  
```bash
pip install -r requirements.txt
```

3. Run ResCNN to obtain the results.
```python 
python ResCNN.py
```
The resulting output should look like this:-  

4. Run the normal CNN to compare results with ResCNN.
```python
python CNN.py
```
The resulting output should look like this:-  


For both the implementations, the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) Handwritten Digit Recognition Dataset has been used. 

## Observations:-  

**Coming soon.**

## Acknowledgements:-  
Parts of the code in this codebase have been adopted from [Pytorch.org](Pytorch.org) and other miscellaneous sources. We are grateful to the respective individuals/organizations for making their work publicly available.
