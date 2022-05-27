# Linear-Generative-Adversarial-Networks - Project Overview 
This project is built using pytorch to regenerate the famous MNIST dataset using Linear Generative Adverserial Networks. The model is implemeneted with simple BCE loss and reduces the loss to *0.203*. 

# Generator and Discriminator Architecture 
Each of the generator and the discriminator consists of constant linear blocks, 4 and 3 block respectively. For the generator, 
  - Linear Layer
  - BatchNormalization Layer
  - ReLU Activation Layer
  
For the discriminator, 
  - Linear Layer
  - LeakyReLU Activation to avoid the dying ReLU problem

# Smaple output 

![smaple output](https://github.com/ahmedheakl/Linear-Generative-Adversarial-Networks/blob/main/imgs/output.gif)
