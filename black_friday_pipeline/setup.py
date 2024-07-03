from setuptools import find_packages, setup

setup(
    name='black_friday_pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tfx',
        'apache-beam[gcp]',
        'tensorflow',
        'tensorflow-transform'
    ]
)