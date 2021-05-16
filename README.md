# BregmanLearning
Implementation of the inverse scale space training algorithms for sparse neural networks, proposed in **A Bregman Learning Framework for Sparse Neural Networks** [[1]](#1).
Feel free to use it and please refer to our paper when doing so.
```
@misc{bungert2021bregman,
      title={A Bregman Learning Framework for Sparse Neural Networks}, 
      author={Leon Bungert and Tim Roith and Daniel Tenbrinck and Martin Burger},
      year={2021},
      eprint={2105.04319},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Method Description
Our Bregman learning framework aims at training sparse neural networks in an inverse scale space manner, starting with very few parameters and gradually adding only relevant parameters during training. We train a neural network 
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;f_\theta:\mathcal{X}\rightarrow\mathcal{Y}" title="net"/> 
</p>

parametrized by weights <img src="https://latex.codecogs.com/svg.latex?\theta" title="weights"/> using the simple baseline algorithm
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;v = \theta," title="Subgradient update" />
</p>

where
* <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}=\{(x_i,y_i)\}_{i=1}^N\subset\mathcal{X}\times\mathcal{Y}" title="training set"/> denotes the training set, 
* <img src="https://latex.codecogs.com/svg.latex?l(\cdot,\cdot)" title="loss"/> denotes a loss function.

The Lipschitz constant of the net w.r.t. the input space variable is defined as
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathrm{Lip}(f_\theta)=\sup_{x,x^\prime\in\mathcal{X}}\frac{|f_\theta(x)-f_\theta(x^\prime)|}{|x-x^\prime|}." title="Lipschitz Constant" />
</p>

In the algorithm we approximate this constant via difference quotients on a finite subset
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{X}_{\mathrm{Lip}}\subset\mathcal{X}\times\mathcal{X}." title="Lipschitz Constant" />
</p>

In order to ensure that the tuples in this set correspond to point with that realize a high difference quotient the set is updated iteratively 
via a gradient ascent scheme.

## References
<a id="1">[1]</a> Leon Bungert, Tim Roith, Daniel Tenbrinck, Martin Burger. "A Bregman Learning Framework for Sparse Neural Networks." arXiv preprint arXiv:2105.04319 (2021). https://arxiv.org/abs/2105.04319

<a id="2">[2]</a> The Pytorch website https://pytorch.org/, see also their git https://github.com/pytorch/pytorch

<a id="3">[3]</a> The MNIST dataset http://yann.lecun.com/exdb/mnist/

<a id="4">[4]</a> The Fashion-MNIST dataset  https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/
