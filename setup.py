import os
from setuptools import setup, find_packages


with open(os.path.join("src", "nn", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__ = "):
            version = str(line.split()[-1].strip('"'))
            break


requirements_base = [
    "numpy>=1.24.3",
    "tqdm>=4.65.0",
]


setup(
    name="nn",
    version=version,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements_base,
)
