# measurements
I made a python package, mainly for personal use for my physics class for quantitative evaluation.
With this package, you can calculate error spread, with the error spread formula:

$\Delta f(x_1,\dots, x_n) = \sqrt{\sum_{i}(\frac{df}{dx_i}\cdot\Delta x_i)^2},$

using symbolic derivatives. It can also fit any function on a given dataset.
There are a few things I plan to add in the future, like fit specific functions, calculate covariance matrix, or even addig a GUI.
If you want to install this as a python package, make sure you have `wheel`, then `cd` to `some/path/to/measurements/measurements/dist` and
type `pip install measurements-0.0.1-py3-none-any.whl`, it should be added to path. Then, you can just `import measurements` and be ready.
