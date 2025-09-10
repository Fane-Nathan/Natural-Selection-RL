from setuptools import setup, find_packages

setup(
    name="natural-selection-rl",
    version="0.1.0",
    description="Natural Selection with Reinforcement Learning - Baldwin Effect Simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "torch>=1.12.0",
        "gymnasium>=0.26.0",
    ],
)