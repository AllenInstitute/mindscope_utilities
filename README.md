# Deprecation warning:
mindscope_utilities is now a deprecated repo that will no longer be supported. It's functionality and many of it's individual functions have, and will continue to be moved over to the brain_observatory_utilities repo: 
https://github.com/AllenInstitute/brain_observatory_utilities


# mindscope_utilities
Utilities for loading, manipulating and visualizing data from the Allen Institute Mindscope program.

Functions in this repository depend on the AllenSDK
https://github.com/AllenInstitute/AllenSDK

# Installation

Set up a dedicated conda environment:

```
conda create -n mindscope_utilities python=3.8 
```

Activate the new environment:

```
conda activate mindscope_utilities
```

Make the new environment visible in the Jupyter 
```
pip install ipykernel
python -m ipykernel install --user --name mindscope_utilities
```

Install mindscope_utilities
```
pip install mindscope_utilities
```

Or if you intend to edit the source code, install in developer mode:
```
git clone https://github.com/AllenInstitute/mindscope_utilities.git
cd mindscope_utilities
pip install -e .
```

# Testing

Tests are run on CircleCI on every github commit.

Tests can be run locally by running the following at the command line:
```
flake8 mindscope_utilities
pytest
```

# Level of Support

We are no longer supporting this repo. We are supporting the brain_observatory_utilities repo which provides much of the same functionality as this repo
