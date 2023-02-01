# LFighter: Defending against Label-flipping Attacks in Federated Learning.
This repository contains PyTorch implementation of the paper ''LFighter: Defending against Label-flipping Attacks in Federated Learning''.

## Paper 

[LFighter: Defending against Label-flipping Attacks in Federated Learning]()

## Content
The repository contains one jupyter notebook for each benchmark which can be used to re-produce the experiments reported in the paper for that benchmark. The notebooks contain clear instructions on how to run the experiments. 

## Data sets
[MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) will be automatically downloaded.
However, [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) requires a manual download using this [link](https://drive.google.com/file/d/1X86CyTJW77a1CCkAFPvN6pqceN63q2Tx/view?usp=sharing). 
After downloading [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/), please save it as imdb.csv in the data folder inside the folder IMDB.


## Dependencies

[Python 3.6](https://www.anaconda.com/download)

[PyTorch 1.6](https://pytorch.org/)

[TensorFlow 2](https://www.tensorflow.org/)


## Results

### Robustness
*The table below shows LFighter's robustness to the label-flipping attack with 40% attackers.* </br></br>
<img src="figures/main_results.PNG" width="100%">




### Accuracy stability
*The figure below shows the source class stability under the label-flipping attack with 40% attackers for the CIFAR10-ResNet18-non-IID and IMDB-BiLSTM benchmarks.* </br></br>
<img src="figures/stability_all.PNG" width="100%">






## Citation 



## Funding
This research was funded by the European Commission (projects H2020-871042 ``SoBigData++'' and H2020-101006879 ``MobiDataLab''), the Government of Catalonia (ICREA Acad\`emia Prizes to J.Domingo-Ferrer and to D. S\'anchez, FI grant to N. Jebreel), and MCIN/AEI/ 10.13039/501100011033 and ``ERDF A way of making Europe'' under grant PID2021-123637NB-I00 ``CURLING''. 

