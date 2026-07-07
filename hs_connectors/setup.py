from setuptools import find_packages, setup

setup(
    name="hs_connectors",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"hs_connectors": ["py.typed"]},
    python_requires=">=3.10",
    install_requires=[
        "safetensors",
        "torch",
    ],
)
