from setuptools import setup, find_packages

setup(
    name="lob_rl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "train": [
            "gymnasium>=0.29",
            "stable-baselines3>=2.0",
        ],
    },
)
