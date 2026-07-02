from setuptools import find_packages, setup

setup(
    name="hs_connectors",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        # >=0.3.10.post1 for force-delete support (remove(key, force=True)),
        # required to reclaim store capacity as samples are consumed.
        "mooncake-transfer-engine>=0.3.10.post1",
    ],
    extras_require={
        "vllm": ["vllm"],
    },
)
