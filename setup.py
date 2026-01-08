from setuptools import setup, find_packages

setup(
  name = 'tab-transformer-pytorch',
  packages = find_packages(),
  version = '0.6.1',
  license='MIT',
  description = 'Tab Transformer - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/tab-transformer-pytorch',
  keywords = [
    'artificial intelligence',
    'transformers',
    'attention mechanism',
    'tabular data'
  ],
  install_requires=[
    'einops>=0.8',
    'discrete-continuous-embed-readout>=0.0.32',
    'hyper-connections>=0.3.12',
    'torch>=2.3',
    'x-mlps-pytorch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
