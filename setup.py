from setuptools import setup

setup(name='mindscope_utilities',
    version='0.1.7',
    packages=['mindscope_utilities'],
    include_package_data = True,
    description='Utilities for loading, manipulating and visualizing data from the Allen Institute Mindscope program',
    url='https://github.com/AllenInstitute/mindscope_utilities',
    author='Allen Institute',
    author_email='marinag@alleninstitute.org, iryna.yavorska@alleninstitute.org, kater@alleninstitute.org, dougo@alleninstitute.org,',
    license='Allen Institute',
    install_requires=[
        'flake8',
        'pytest',
        'allensdk==2.11.2',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: Other/Proprietary License', # Allen Institute Software License
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8'
  ],
)