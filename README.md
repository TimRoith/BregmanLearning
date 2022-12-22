# 📈 BregmanLearning
Implementation of the inverse scale space training algorithms for sparse neural networks, proposed in **A Bregman Learning Framework for Sparse Neural Networks** [[1]](#1).
Feel free to use it and please refer to our paper when doing so.
```
@article{JMLR:v23:21-0545,
  author  = {Leon Bungert and Tim Roith and Daniel Tenbrinck and Martin Burger},
  title   = {A Bregman Learning Framework for Sparse Neural Networks},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {192},
  pages   = {1--43},
  url     = {http://jmlr.org/papers/v23/21-0545.html}
}
```
<p align="center">
      <img src="https://user-images.githubusercontent.com/44805883/120522872-99294080-c3d5-11eb-9b9d-48809054be15.png" width="600">
</p>

## 💡 Method Description
Our Bregman learning framework aims at training sparse neural networks in an inverse scale space manner, starting with very few parameters and gradually adding only relevant parameters during training. We train a neural network <img src="https://latex.codecogs.com/svg.latex?f_\theta:\mathcal{X}\rightarrow\mathcal{Y}" title="net"/> parametrized by weights <img src="https://latex.codecogs.com/svg.latex?\theta" title="weights"/> using the simple baseline algorithm
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\begin{cases}v\gets\,v-\tau\hat{\nabla}\mathcal{L}(\theta),\\\theta\gets\mathrm{prox}_{\delta\,J}(\delta\,v),\end{cases}" title="Update" />
</p>

where 
* <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}" title="loss"/> denotes a loss function with stochastic gradient <img src="https://latex.codecogs.com/svg.latex?\hat{\nabla}\mathcal{L}" title="stochgrad"/>,
* <img src="https://latex.codecogs.com/svg.latex?J" title="J"/> is a sparsity-enforcing functional, e.g., the <img src="https://latex.codecogs.com/svg.latex?\ell_1" title="ell1"/>-norm,
* <img src="https://latex.codecogs.com/svg.latex?\mathrm{prox}_{\delta\,J}" title="prox"/> is the proximal operator of <img src="https://latex.codecogs.com/svg.latex?J" title="J"/>.

Our algorithm is based on linearized Bregman iterations [[2]](#2) and is a simple extension of stochastic gradient descent which is recovered choosing <img src="https://latex.codecogs.com/svg.latex?J=0" title="Jzero"/>. We also provide accelerations of our baseline algorithm using momentum and Adam [[3]](#3). 

The variable <img src="https://latex.codecogs.com/svg.latex?v" title="v"/> is a subgradient of <img src="https://latex.codecogs.com/svg.latex?\theta" title="weights"/> with respect to the *elastic net* functional 

<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?J_\delta(\theta)=J(\theta)+\frac1\delta\|\theta\|^2" title="el-net"/>
</p>

and stores the information which parameters are non-zero.

## 🎲 Initialization

We use a *sparse initialization strategy* by initializing parameters non-zero with a small probability.
Their variance is chosen to avoid vanishing or exploding gradients, generalizing Kaiming-He or Xavier initialization.

## 🔬 Experiments
The different experiments can be executed as Jupyter notebooks in the 
[notebooks folder](https://github.com/TimRoith/BregmanLearning/tree/main/notebooks).

### Classification

<p align="center">
      <img src="https://user-images.githubusercontent.com/44805883/120520997-bdd0e880-c3d4-11eb-9743-166b097fe70b.png" width="700">
</p>

#### Mulit Layer Perceptron
In this experiment we consider the MNIST classification task using a simple multilayer perceptron.
We compare the LinBreg optimizer to standard SGD and proximal descent. The respective notebook can be found at 
[MLP-Classification](https://github.com/TimRoith/BregmanLearning/blob/main/notebooks/MLP-Classification.ipynb).

<p align="center">
      <img src="https://user-images.githubusercontent.com/44805883/126574731-ad93c4fe-72e1-43c8-8e82-082045a312c0.png" width="700">
</p>


#### Convolutions and Group Sparsity
In this experiment we consider the Fashion-MNIST classification task using a simple convolutional net. The experiment can be excecuted as a notebook, 
namely via the file [ConvNet-Classification](https://github.com/TimRoith/BregmanLearning/blob/main/notebooks/ConvNet-Classification.ipynb).

#### ResNet
In this experiment we consider the CIFAR10 classification task using a ResNet. The experiment can be excecuted as a notebook, 
namely via the file [ResNet-Classification](https://github.com/TimRoith/BregmanLearning/blob/main/notebooks/ResNet-Classification.ipynb).


### NAS

<p align="center">
      <img src="https://user-images.githubusercontent.com/44805883/120520730-70547b80-c3d4-11eb-94f8-df36e24ad778.png" width="700">
</p>

This experiment implements the neural architecture search as proposed in  [[4]](#4). 

The corresponding notebooks are 
[DenseNet](https://github.com/TimRoith/BregmanLearning/blob/main/notebooks/DenseNet.ipynb) and 
[Skip-Encoder](https://github.com/TimRoith/BregmanLearning/blob/main/notebooks/Skip-Encoder.ipynb).

## :point_up: Miscellaneous

The notebooks will throw errors if the datasets cannot be found. You can change the default configuration ```'download':False``` to ```'download':True``` in order to automatically download the necessary dataset and store it in the appropriate folder.

If you want to run the code on your CPU you should replace ```'use_cuda':True, 'num_workers':4``` by ```'use_cuda':False, 'num_workers':0``` in the configuration of the notebook.

## 📝 References
<a id="1">[1]</a> Leon Bungert, Tim Roith, Daniel Tenbrinck, Martin Burger. "A Bregman Learning Framework for Sparse Neural Networks." Journal of Machine Learning Research 23.192 (2022): 1-43. https://www.jmlr.org/papers/v23/21-0545.html

<a id="2">[2]</a> Woatao Yin, Stanley Osher, Donald Goldfarb, Jerome Darbon. "Bregman iterative algorithms for \ell_1-minimization with applications to compressed sensing." SIAM Journal on Imaging sciences 1.1 (2008): 143-168.

<a id="3">[3]</a> Diederik Kingma, Jimmy Lei Ba. "Adam: A Method for Stochastic Optimization." arXiv preprint arXiv:1412.6980 (2014). https://arxiv.org/abs/1412.6980

<a id="4">[4]</a> Leon Bungert, Tim Roith, Daniel Tenbrinck, Martin Burger. "Neural Architecture Search via Bregman Iterations." arXiv preprint arXiv:2106.02479 (2021). https://arxiv.org/abs/2106.02479
