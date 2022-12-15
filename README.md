# RecurrentWhisperer - A general class template for training recurrent neural networks using Tensorflow.

Written for Python 3.6.9.

RecurrentWhisperer is a base class for training recurrent neural networks or other deep learning models using TensorFlow. RecurrentWhisperer provides functionality for:

1) Training a recurrent neural network using modern techniques for
encouraging stable training, such as adaptive learning rate management and adaptive gradient norm clipping. RecurrentWhisperer handles common tasks like splitting training data into batches, making gradient steps based on individual batches of training data, periodically evaluating validation data, and periodically saving model checkpoints.

2) Managing Tensorboard visualizations of training progress.

3) Managing a directory structure for maintaining many different variants
of a model (i.e., with different hyperparameter settings). Previously
saved models can be readily restored from checkpoints, and training runs
can be readily resumed if their execution was interrupted or preempted.

**If you are using RecurrentWhisperer in research to be published, please cite our accompanying paper in your publication:**

Golub and Sussillo (2018), "FixedPointFinder: A Tensorflow toolbox for identifying and characterizing fixed points in recurrent neural networks," *Journal of Open Source Software*, 3(31), 1003, https://doi.org/10.21105/joss.01003 .

[![DOI](http://joss.theoj.org/papers/10.21105/joss.01003/status.svg)](https://doi.org/10.21105/joss.01003)


## Recommended Installation

The recommended installation is to assemble all dependencies in a virtual environment. 

To create a new virtual environment, enter at the command line:
```bash
$ python3 -m venv --system-site-packages your-virtual-env-name
```
where `your-virtual-env-name` is a path to the the virtual environment you would like to create (e.g.: `/home/rwhisp`). Then activate your new virtual environment:
```bash
$ source your-virtual-env-name/bin/activate
```

Next, install all dependencies in your virtual environment. This step will depend on whether you require Tensorflow with GPU support.

For GPU-enabled TensorFlow, use:

```bash
$ pip install -e git+https://github.com/mattgolub/recurrent-whisperer.git@master#egg=v1.5.0[gpu]
```

For CPU-only TensorFlow, use:

```bash
$ pip install -e git+https://github.com/mattgolub/recurrent-whisperer.git@master#egg=v1.5.0[cpu]
```

When you are finished working in your virtual environment, enter:

```bash
$ deactivate
```

## Advanced Users

Advanced Python users may skip the Recommended Installation, opting to instead clone this repository and ensure that compatible versions of the following prerequisites are available:

* **TensorFlow** (requires at least version 1.14) ([install](https://www.tensorflow.org/install/))
* **NumPy, SciPy and Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains both)
* **PyYaml** ([install](https://pyyaml.org))

## Getting started

See [FlipFlop.py](https://github.com/mattgolub/fixed-point-finder/blob/master/example/FlipFlop.py) for an example subclass that inherits from RecurrentWhisperer for the purposes of training an RNN to implement an N-bit memory.
