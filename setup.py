from setuptools import setup, find_packages

setup(
    name='state_estimation_python',
    version='0.1',
    description='State estimation toolbox with python',
    url='https://github.com/frknayk/state_estimation_python',
    author='Furkan Ayik',
    author_email='furkanayik@outlook.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],
    install_requires=['numpy','pyunitreport'],
)

