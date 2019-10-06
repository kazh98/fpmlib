from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fpmlib",
    version="0.0.1",
    author="Kazuhiro HISHINUMA",
    author_email="kaz@cs.meiji.ac.jp",
    description="A fork of fixed point quasiconvex subgradient method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazh98/fpmlib",
    packages=find_packages(include=["fpmlib", "fpmlib.*"]),
    install_requires=[
        "numpy>=1.17.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.7',
)
