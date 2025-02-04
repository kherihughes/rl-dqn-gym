from setuptools import setup, find_packages

setup(
    name="dqn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "gymnasium>=0.29.0",
        "matplotlib>=3.5.0"
    ],
) 