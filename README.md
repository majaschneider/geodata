# geodata

Library for working with points and routes based on geographic coordinates.
Provides convenience methods for importing and preprocessing de4l (https://de4l.io) specific datasets.


# Installation

Install this package in your repository via pip:
```bash
pip install git+https://github.com/majaschneider/geodata.git
```
This assumes you already have PyTorch installed with whatever requirements you need (see [here](https://pytorch.org/get-started/locally/) for further info).

To also get PyTorch with whatever CUDA version it ships naturally you can use:
```bash
pip install git+ssh://git@git.informatik.uni-leipzig.de/scads/de4l/privacy/de4l-geodata.git#egg=de4l_geodata[torch]
```

# Development

For development you can use

``` bash
poetry install -E torch
```

You can then e.g. run the tests locally via:

``` bash
poetry run python -m unittest discover -v
```

For more info about poetrys virtual enviroment see [here](https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment)
