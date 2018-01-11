from setuptools import setup

setup(
    name='tdv_postagger',
    version='0.1',
    packages=['tdv_postagger', 'tdv_postagger.ml', 'tdv_postagger.test', 'tdv_postagger.config',
              'tdv_postagger.featurizer'],
    url='',
    license='',
    author='Danilo S. Carvalho',
    author_email='danilo@jaist.ac.jp',
    description='TDV feature-based LSTM POS-tagger',
    install_requires=[
        'saf',
        'wikt_morphodecomp',
        'numpy',
        'keras'
    ]
)
