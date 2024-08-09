from setuptools import setup, find_packages

setup(
    name='pytential',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # sympy
    ],
    author='Mike Welland',
    author_email='mike@mikewelland.com',
    description='A pacakge to represent and manipulate potential energy functions',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mwelland/pytential',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU LESSER GENERAL PUBLIC LICENSE',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)