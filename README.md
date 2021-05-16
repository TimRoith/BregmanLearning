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
      <img src="https://latex.codecogs.com/svg.latex?\begin{cases}v\gets\,v-\tau\mathcal{L}(\theta),\\\theta\gets\mathrm{prox}_{\delta\,J}(\delta\,v),\end{cases}" title="Update" />
</p>

where 
* <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}" title="loss"/> denotes a loss function,
* <img src="https://latex.codecogs.com/svg.latex?\mathrm{prox}" title="prox"/> is the proximal operator, and
* <img src="https://latex.codecogs.com/svg.latex?\mathrm{prox}" title="J"/> is a sparsity-enforcing functional, e.g., the <img src="https://latex.codecogs.com/svg.latex?\ell_1" title="ell1"/>-norm.

## References
<a id="1">[1]</a> Leon Bungert, Tim Roith, Daniel Tenbrinck, Martin Burger. "A Bregman Learning Framework for Sparse Neural Networks." arXiv preprint arXiv:2105.04319 (2021). https://arxiv.org/abs/2105.04319

<a id="2">[2]</a> The Pytorch website https://pytorch.org/, see also their git https://github.com/pytorch/pytorch

<a id="3">[3]</a> The MNIST dataset http://yann.lecun.com/exdb/mnist/

<a id="4">[4]</a> The Fashion-MNIST dataset  https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/
