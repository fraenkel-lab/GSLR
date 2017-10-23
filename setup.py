from setuptools import setup

setup(
    name='gslr',
    packages=['gslr'],
    version='0.0.1',
    url='https://github.com/fraenkel-lab/GSLR',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6'],
    license='MIT',
    author='zfrenchee ludwigschmidt',
    author_email='alex@lenail.org',
    description='',
    install_requires=[
        'numpy',
        'numba',
        'pcst_fast',
    ],
)

