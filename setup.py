from setuptools import find_packages
from distutils.core import setup

# todo: add license and pip install support
setup(
    name='shifu',
    version='0.1.0',
    author='Jaylon',
    license="None for now",
    packages=find_packages(),
    author_email='wangjilong@sensetime.com',
    description='A lightweight robotic simulation builder for robot learning research',
    install_requires=[
        'torch'
    ]
)
