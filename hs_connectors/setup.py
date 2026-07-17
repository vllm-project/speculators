from setuptools import find_packages, setup

setup(
    name="hs_connectors",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "mooncake-transfer-engine",
    ],
    extras_require={
        "vllm": ["vllm"],
    },
)
