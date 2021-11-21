# ResCNN: A novel CNN Implementation for bigger and more resilient batch processing and learning from Images.

*This work has been accepted at [UPCON 2021](http://upcon2021.in/). The publication will be availabale on IEEE Xplore Library very soon.*  
*Authors: [Avirup Dey](https://avirupju.github.io/) and [Sarosij Bose](https://sarosijbose.github.io/)*

## Description:-
Here we implement ResCNN, a Convolutional neural network featured as an viable alternative to traditional CNNs for Deep learning. We first convert the traditional convolution operation into a purely matrix based form which is based on the theory of [unrolling](https://hal.inria.fr/inria-00112631/document) convolution. Such an implementation allows us to employ some Image processing techniques mainly focusing on compression of images. One such technique used here is [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition), which is applied on entire dataset/Images and then fed into our convolutional network. We further experiment with various sets of hyperparameters to show that our network is able to learn at a much higher rate without collapsing like a standard CNN Model whose detailed comparisons are done below. Further, we also show that such improvement can be brought about without significant cost overheads or employment of GPUs. 
Slides of the conference proceedings are available [here](https://sarosijbose.github.io/files/ResCNN_UPCON_2021_Slides.pdf)

## Setup:-  

1. It is recommended to setup a fresh virtual environment first.  
```bash
python -m venv rescnn
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
We fared a conventional CNN against our prototype. Our model could retain its stability even at higher learning rates while the conventional
CNN showed significant drop in accuracy. Beyond a critical learning rate(around 0.06) it could not converge to a minima most of the times. The model had 
10 trial runs at each learning rate and the results show the average behaviour over all the runs.<br>

<img src = "https://github.com/AvirupJU/Fast_Convolution/blob/main/Results/acc_lr.jpg" height="300px" width="450px">

At higher learning rate the conventional CNN "skips" the minima and settles on some sub-optimal point and this adversely affects the model performance.
But our model does not suffer from the same problem within the practical limits of choosing a leaning rate, given a specific model.
Since our prototype was tested against a shallow neural network we started from learning rate = 0.01 went upto rates as high as 0.1.<br>

<p float="left">
  <img src="https://github.com/AvirupJU/Fast_Convolution/blob/main/Results/lr_0.01.jpg" width="400px" />
  <img src="https://github.com/AvirupJU/Fast_Convolution/blob/main/Results/lr_0.07.jpg" width="420px" /> 
</p><br>

We also studied the behaviour of our model with varying degrees of singular value decompostion(SVD).


## Acknowledgements:-  
Parts of the code in this codebase have been adopted from [Pytorch.org](Pytorch.org) and other miscellaneous sources. We are grateful to the respective individuals/organizations for making their work publicly available.
