# FastCNN: A novel CNN Implementation for faster batch processing and learning from Images.

*This work has been submitted to [UPCON 2021](http://upcon2021.in/) and is currently under review.*  
*Authors: [Avirup Dey](https://avirupju.github.io/) and [Sarosij Bose](https://sarosijbose.github.io/)*

## Description:-
Here we implement FastCNN, a Convolutional neural network featured as an viable alternative to traditional CNNs for Deep learning. We first convert the traditional convolution operation into a purely matrix based form which is based on the theory of [unrolling](https://hal.inria.fr/inria-00112631/document) convolution. Such an implementation allows us to employ some Image processing techniques mainly focusing on compression of images. One such technique used here is [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition), which is applied on entire dataset/Images and then fed into our convolutional network. We further experiment with various sets of hyperparameters to how that our network is able to learn at a much higher rate without collapsing like a standard CNN Model whose detailed comparisons are done below. Further, we also show that such improvement can be brought about without significant cost overheads or employment of GPUs. 

## Setup:-  

**Coming soon.**

## Observations:-  

**Coming soon.**

## Acknowledgements:-  
Parts of the code in this codebase have been adopted from [Pytorch.org](Pytorch.org) and other miscellaneous sources. We are grateful to the respective individuals/organizations for making their work publicly available.
