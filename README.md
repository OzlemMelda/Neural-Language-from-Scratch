# Neural-Language-from-Scratch

I train a neural language model using a multi-layer perceptron given below. This network receives 3 consecutive words as the input and aims to predict the next word. This model is trained using cross-entropy loss function, which corresponds to maximizing the probability of the target word.

![image](https://user-images.githubusercontent.com/53811688/124980563-36af0a00-e03d-11eb-915c-c434227afd4b.png)

The network consists of a 16 dimensional embedding layer, a 128 dimensional hidden layer and one output layer. The input consists of a sequence of 3 consecutive words, provided as
integer valued indices representing a word in our 250-word dictionary. Hidden layer has sigmoid activation function and the output layer is a softmax over the 250 words in our dictionary. In the hidden layer before activation function, 3 word embeddings are concatenated and fed to the sigmoid.

NOTE: I did not use any deep learning libraries such as tensorflow, and pytorch but used Python libraries and functions, such as numpy, matplotlib.

## Mathematical Representation of The Network

### Representation of Variables
![image](https://user-images.githubusercontent.com/53811688/124980744-72e26a80-e03d-11eb-8f5c-86bc98ca818f.png)

* Forward Propagation
![image](https://user-images.githubusercontent.com/53811688/124980883-9ad1ce00-e03d-11eb-8c81-061e9577d822.png)

* Gradients to be computed in Backward Propagation
![image](https://user-images.githubusercontent.com/53811688/124981004-bb9a2380-e03d-11eb-93cc-37188254935d.png)
![image](https://user-images.githubusercontent.com/53811688/124981039-c6ed4f00-e03d-11eb-821a-182332b1df37.png)

## File Descriptions
* Network.py which has a Network class, where I have the forward, backward propagation, and the activation functions. Matrix-vector operations are used.
* main.py, where I load the dataset, shuffle the training data and divide it into mini-batches, write the loop for the epoch and iterations, and evaluate the model on validation set during training.
* eval.py, where I load the learned network parameters and evaluate the model on test data.

## How to Run
* Execute "$python main.py" to train model for multiple parameters (learning rate, learning rate decay, batch size) and save the best model according to validation accuracy as "model.pk". To save you some time, I already defined parameters with best validation accuracy. Through training, you will see training and validation accuracy at the last batch of each epoch.
* Execute "$python eval.py" to load trained model and get test accuracy.

## Results
* Model is trained for multiple parameters (learning rate, learning rate decay, batch size) and the best model is selected according to validation accuracy. To save you some time, I already defined parameters in main.py with best validation accuracy I observed. (learning rate = 0.012, learning rate decay = 0.95, number of iterations to decay learning rate = 75, batch size = 900) Training accuracies at last batch of each epoch is 0.257, 0.256, 0.273, 0.266, 0.24 in order. Validation accuracies of each epoch is 0.243, 0.247, 0.249, 0.247, 0.249 in order. You can see small decrease in training accuracy since I printed out the accuracies of last batch of each epoch. I also observed that there is no much change in training and validation accuracies after few epoches.
* Test accuracy of the trained model is 0.248

IMPORTANT NOTE: Since I did not use any deep learning libraries such as tensorflow, and pytorch but used only Python libraries and functions like numpy for matrix-vector operations, I was able to increase the performance of the model to some point. 
