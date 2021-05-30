import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="guided_conv",
    version="0.1.0",
    author="itsikad",
    author_email="itsik.adiv@gmail.com",
    description="Guided convolution/Dynamic filtering layers",
    long_description=long_description,
    url="https://github.com/itsikad/guided_convolution.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
