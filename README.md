# RecurrentWhisperer - A general class template for training recurrent neural networks using Tensorflow.

Written using Python 2.7.6.

## Recommended Installation

The recommended installation is to assemble all dependencies in a virtual environment. 

To create a new virtual environment, enter at the command line:
```bash
$ virtualenv your-virtual-env-name
```
where `your-virtual-env-name` is a path to the the virtual environment you would like to create (e.g.: `/home/rwhisp`). Then activate your new virtual environment:
```bash
$ source your-virtual-env-name/bin/activate
```

Next, install all dependencies in your virtual environment. This step will depend on whether you require Tensorflow with GPU support.

For GPU-enabled TensorFlow, use:

```bash
$ pip install -e git+https://github.com/mattgolub/recurrent-whisperer.git@master#egg=v1.0.0[gpu]
```

For CPU-only TensorFlow, use:

```bash
$ pip install -e git+https://github.com/mattgolub/recurrent-whisperer.git@master#egg=v1.0.0[cpu]
```

When you are finished working in your virtual environment, enter:

```bash
$ deactivate
```

## Advanced Users

Advanced Python users may skip the Recommended Installation, opting to instead clone this repository and ensure that compatible versions of the following prerequisites are available:

* **TensorFlow** version 1.10 ([install](https://www.tensorflow.org/install/))
* **NumPy, SciPy and Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains both)
* **PyYaml** ([install](https://pyyaml.org))

## Getting started

See [FlipFlop.py](https://github.com/mattgolub/fixed-point-finder/blob/master/example/FlipFlop.py) for an example subclass that inherits from RecurrentWhisperer for the purposes of training an RNN to implement an N-bit memory.
