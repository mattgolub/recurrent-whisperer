from setuptools import setup, find_packages

setup(
    name = 'recurrent-whisperer',
    version = '1.0.0',
    url = 'https://github.com/mattgolub/recurrent-whisperer.git',
    author = 'Matt Golub',
    author_email = 'mgolub@stanford.edu',
    description = 'A general class template for training recurrent neural networks using Tensorflow',
    packages = find_packages(),
    install_requires = ['numpy >= 1.15.2', 'scipy >= 1.1.0', 'pyyaml >= 3.13'],
)
