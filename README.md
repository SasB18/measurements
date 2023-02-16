# Measurements
I made a python package, mainly for personal use for my physics lab courses to evaluate measured data.
With this package, you can calculate average, standard deviaton and even error spread, with the error spread formula:

$\Delta f(x_1,\dots, x_n) = \sqrt{\sum_{i}(\frac{df}{dx_i}\cdot\Delta x_i)^2},$

using symbolic derivatives.
There are a few things I plan to add in the future, like fit specific functions or even addig a GUI.
If you want to install this as a python package, make sure you have `wheel`, then `cd` to `some/path/to/measurements/measurements/dist` and
type `pip install measurements-0.0.1-py3-none-any.whl`, it should be added to path. Then, you can just `import measurements` and be ready.
