from setuptools import find_packages, setup


setup(
    name="nv",
    version="1.0",
    author="khaykingleb",
    package_dir={"": "nv"},
    packages=find_packages("nv"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)  
