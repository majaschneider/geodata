# de4l-geodata

Library for working with points and routes based on geographic coordinates.
Provides convenience methods for importing and preprocessing de4l (https://de4l.io) specific datasets.

Install this package in your repository via pip:
```python
pip install git+ssh://git@git.informatik.uni-leipzig.de/scads/de4l/privacy/de4l-geodata.git
```
This assumes you already have PyTorch installed with whatever requirements you need (see [here](https://pytorch.org/get-started/locally/) for further info).

To also get PyTorch with whatever CUDA version it ships naturally you can use:
```python
pip install git+ssh://git@git.informatik.uni-leipzig.de/scads/de4l/privacy/de4l-geodata.git#egg=de4l_geodata[torch]
```
