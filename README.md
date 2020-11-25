# malaria-classifier
In this repository, I solved a real-life image classification problem related to medicine.
The dataset was obtained from the National Library of Medicine: https://lhncbc.nlm.nih.gov/publication/pub9932

I used a convolutional neural network with mainly ReLU activation except for the last layer, which used a sigmoid function. The design of the network (number of layers, types of layers, etc.) is arbitrary for the most part and is refined through some trial and error. 

Layers: 
Conv2D: Easily summarize the image and reduce spatial capacity
MaxPool2D: Further decrease size of the image
Dropout: Prevent overfitting by randomization
Dense: Final layers that provide the classification value

Stopped training at 30 epochs as further iterations through the data set were unnecessary 
