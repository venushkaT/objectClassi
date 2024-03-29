# Supervised and unsupervised based object classification

combines supervised and unsupervised architectures in one model to achieve a classification task.
The task is achieved using two different network architectures. 

## Usage
### Dependencies

 - Python3.6+
 - matplotlib
 - pytorch1.3
 - torchvision 0.4.1
 - tensorboard
 
### Train
To train model (Autoencoder)

    python autoEncorder.py --train
    
To see the Output of Auto Encoder 

    python autoEncorder.py --valid
To train CNN model  

    python cnnClassification.py --train

To test CNN model
    
    python cnnClassification.py --valid
    
### Visualization
To Autoencorder 

    tensorboard --logdir="logs2"
    
To Classification model

    tensorboard --logdir="logs"



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
