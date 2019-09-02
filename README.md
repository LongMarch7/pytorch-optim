# PyTorch optim

[![Made with Python](https://img.shields.io/badge/Made_with-Python-blue.svg)](https://img.shields.io/badge/Made_with-Python-blue.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This notebook demonstrates how to use the `optim` package to prevent updating the weights manually. It is based on the [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) post.

## Optimization Algorithms
The optimization algorithm (or optimizer) is the main approach used today for training a machine learning model to minimize its error rate. There are two metrics to determine the efficacy of an optimizer: speed of convergence (the process of reaching a global optimum for gradient descent); and generalization (the model’s performance on new data). Popular algorithms such as Adaptive Moment Estimation (Adam) or Stochastic Gradient Descent (SGD) can capably cover one or the other metric, but researchers can’t have it both ways.

SGD is a variant of gradient descent. Instead of performing computations on the whole dataset - which is redundant and inefficient - SGD only computes on a small subset or random selection of data examples. SGD produces the same performance as regular gradient descent when the learning rate is low.

Essentially Adam is an algorithm for gradient-based optimization of stochastic objective functions. It combines the advantages of two SGD extensions - Root Mean Square Propagation (RMSProp) and Adaptive Gradient Algorithm (AdaGrad) - and computes individual adaptive learning rates for different parameters.

Despite the widespread popularity of Adam, recent research papers have noted that it can fail to converge to an optimal solution under specific settings. The paper, [Improving Generalization Performance by Switching from Adam to SGD](https://arxiv.org/pdf/1712.07628.pdf), demonstrates that adaptive optimization techniques such as Adam generalize poorly compared to SGD.

The paper, [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://openreview.net/pdf?id=Bkg3g2R9FX), proposes an optimizer, AdaBound, that trains as fast as Adam and as good as SGD. Luckily, we have [the implementation](https://github.com/Luolc/AdaBound) compatible with PyTorch.

## Todos
 - Unfortunately, after I worked on some projects from Stanford's [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) course, I found getting your hands dirty with DNN libraries, such as PyTorch, is a faster way to learn rather than joining a class or reading a lot of papers. Just go to [PyTorch Tutorials](https://pytorch.org/tutorials/) and code.
 - Build a question answering system for competitions.

## License
[PyTorch optim](https://github.com/yungshun317/pytorch-optim) is released under the [MIT License](https://opensource.org/licenses/MIT) by [yungshun317](https://github.com/yungshun317).