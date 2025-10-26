"""Setup script for temporal-steering package."""

from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=["tests*", "venv*", "data*", "activations*", "probes*", "results*"]),
    include_package_data=True,
)
