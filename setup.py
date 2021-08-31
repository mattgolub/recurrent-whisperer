from setuptools import setup, find_packages

setup(
    name = 'recurrent-whisperer',
    version = '1.4.0',
    url = 'https://github.com/mattgolub/recurrent-whisperer',
    author = 'Matt Golub',
    author_email = 'mgolub@stanford.edu',
    description = 'A general class template for training recurrent neural networks using Tensorflow',
    license='Apache 2.0',
    packages = find_packages(),
    install_requires = [
        'numpy >= 1.13.3',
        'scipy >= 1.1.0',
        'matplotlib >= 2.2.3',
        'pyyaml >= 3.13'],
    extras_require={
        'cpu': ['tensorflow==1.14.0'],
        'gpu': ['tensorflow-gpu==1.14.0'],},
)
