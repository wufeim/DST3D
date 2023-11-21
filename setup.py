from setuptools import find_packages
from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    author="Wufei Ma, Qihao Liu, Jiahao Wang",
    author_email="wufeim@gmail.com",
    name="dst3d",
    version="0.1.0",
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="DST-3D data generation.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    packages=find_packages(include=["dst3d", "dst3d.*"]),
    url="https://github.com/wufeim/dst3d",
    zip_safe=False,
)
