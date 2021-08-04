import setuptools
import os


_PATH_ROOT = os.path.dirname(__file__)

path_readme = os.path.join(_PATH_ROOT, "README.md")
with open(path_readme, "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="guided-conv",
    version="0.1.4",
    author="itsikad",
    author_email="itsik.adiv@gmail.com",
    description="Guided convolution/Dynamic filtering layers",
    long_description=long_description,
    url="https://github.com/itsikad/guided-convolution",
    keywords=['deep learning', 'pytorch', 'AI'],
    packages=setuptools.find_packages(where="guided_conv"),
    package_dir={"": "guided_conv"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
