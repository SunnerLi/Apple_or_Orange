# Apple or Orange    
    
This repo show the different model to classify the apple and orange.   
    
<p align="center">
  <img src="http://cdn.mysitemyway.com/etc-mysitemyway/icons/legacy-previews/icons/magic-marker-icons-food-beverage/115460-magic-marker-icon-food-beverage-food-apple1-sc44.png" height=256 width=256/>
  <img src="https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcSLbpDCLDjV4Ij6nSB618pGM6NilaoGobqVnssPgUUz7pO_pWzt" />
</p>
<p align="left">
                       
                       
                       

</p>
    
    
Motivation
----------------------
It's a long history to discuss the object detection and the object recognition. The neural network is usually used to complete this task. After the professor Alex Krizhevsky won the championship, the deep learning is discussed frequently. This project adopt some model to do the similar task.    
    
    
Abstract    
----------------------    
This project use Keras module to construct the neural network structures. The structures include CIFAR10, VGG16, Perceptron, small perceptron and sunnerNet.     
    
    
Parameter    
----------------------    
* The parameters of CIFAR10 is reference the tensorflow tutorial.     
* The parameters of perceptron is tuned by myself.    
* The parameters of VGG-16 is reference from the keras example.    
    
    
SunnerNet
----------------------
The sunnerNet is the structure designed by myself, which is revised from the AlexNet. The sunnerNet has 4 Convolutional layers. The following layer is a FC layer and the Softmax layer. The structure is shwon below. The detail parameter can refer my code.     
```
    # 1st
        conv1 -> relu1 -> maxpool1 -> norm1

    # 2nd
     -> conv2 -> relu2 -> maxpool2 -> norm2

    # 3rd
     -> conv3 -> relu3 -> maxpool3

    # 4th
     -> conv4 -> relu4 -> maxpool4

    # FC & output
     -> FC1 -> tanh1 -> FC2 -> softmax
```
    
    
Result
----------------------
(Skip)
    
    
Training Data
----------------------
The images are collected from Bing and google. Some images are from Fruit Image Data set(FIDS). You can also make or extend your image set. See this. 