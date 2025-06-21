from setuptools import setup, find_packages

setup(
    name='mps',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "monai",
        "nibabel",
        "pandas",
    ],
    author='jaebeombme',
    description='MRI Pulse Sequence Classification Model',
)