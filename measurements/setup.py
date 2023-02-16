from setuptools import setup, find_packages

setup(name='measurements',
      version='0.0.5',
      packages=find_packages(),
      description='A Python library for qualitative evaluation',
      author='Balázs Sashalmi',
      install_requires=['numpy', 'sympy', 'scipy', 'matplotlib'],
     )
