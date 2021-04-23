# MobilenetXT-tf
Reproduce mobilenetxt with tensorflow

## INTRODUCTION
  
#### A non-official Tensorflow-version implementation of MobileNeXt model from paper Rethinking Bottleneck Structure for Efficient Mobile Network Design(https://arxiv.org/pdf/2007.02269.pdf)
#### The artchitecture of this model is:
 
### ![](https://github.com/carolchenyx/MobilenetXT-tf/blob/main/images/mobilenetxt.jpg)
 
 
## RUN
#### If you want to run the demo code for classification, you need to prepare the environment fisrt according to the requirement.
 
    $ pip install -r requirement.txt
    $ python run.py

#### After training of this model, the weight of the model is more larger than mobinetv2. The channels of this network are too large. 
