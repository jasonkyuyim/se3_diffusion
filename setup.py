from setuptools import setup

setup(
    name="protein_diffusion",
    packages=[
        'data',
        'analysis',
        'model',
        'experiments',
        'scripts',
        'openfold'
    ],
    package_dir={
        'data': './data',
        'analysis': './analysis',
        'model': './model',
        'experiments': './experiments',
        'scripts': './scripts',
        'openfold': './openfold',
    },
)
