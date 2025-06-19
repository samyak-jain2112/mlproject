from setuptools import find_packages,setup

setup(
    name='mlproject',
    version='0.0.1',
    author='samyak',
    author_email='samyakjain2112gmail.com',
    packages=find_packages(),
    install_requires=['pandas','numpy','seaborn']
)