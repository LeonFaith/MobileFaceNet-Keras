# MobileFaceNet-Keras
A Keras implementation of MobileFaceNet from https://arxiv.org/abs/1804.07573. It is the 1st time I uploaded my own work. If the license or citation is wrong, please inform me and I will change it ASAP. The Keras model for inference is ~14 Mb, as well as the TensorFlow model in optimized protobuf format, so I might not follow all the details in the original work (should be around 5 Mb). 
  
1. Data Preprocessing Strategy:  
(1) Use the celebrity & msra datasets from the Trillion Pairs dataset: http://trillionpairs.deepglint.com/data.  
(2) For each identity folder:  
a. n > 350  
Randomly pick 350 pics from the origin data set  
b. 200 < n <= 350  
Keep all the pics  
c. 90 < n <= 200  
Keep all the pics & Transfer them to HSV (Double the data)  
d. 30 < n <= 90  
Keep all the pics, Transfer them to HSV, Add Gaussian noise to them & Add Salt & Pepper noise to them  
e. n <= 30  
Drop the folder  
(3) Crop & resize:  
a. Use MTCNN to locate the face (which has already been aligned) & Crop it out  
b. Resize the image to 112 x 112  

2. Training Strategy:  
At first I did not finish the data preprocessing step and use parts of the processed data for training with 512 as the mini batch size. Since I only bought 1 Nvidia 1080 Ti GPU myself and its memory is not enough when the dataset becomes larger (there is one intermediate layer whose size is related to the number of labels). So for now the training has been paused and I am waiting for upgrading my hardware when I got money.  
  
3. Improvement for training step in progress.  
