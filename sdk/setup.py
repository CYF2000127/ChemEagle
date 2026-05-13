from setuptools import setup, find_packages

setup(
    name="chemeagle-client",
    version="0.1.0",
    description="Python client for the ChemEagle public API",
    author="ChemEagle",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=["requests>=2.28"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
