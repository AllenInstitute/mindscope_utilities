from setuptools import setup

setup(name='mindscope_utilities',
    version='0.1.0',
    description='Utilities for loading, manipulating and visualizing data from the Allen Institute Mindscope program',
    url='https://github.com/AllenInstitute/mindscope_utilities',
    author='Allen Institute',
    author_email='marinag, iryna.yavorska, kater, dougo, @alleninstitute.org',
    license='Allen Institute',
    install_requires=[
        'flake8',
        'pytest',
        'allensdk==2.11.2',
    ],
)