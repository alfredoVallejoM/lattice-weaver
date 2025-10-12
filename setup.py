from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lattice-weaver",
    version="5.0.0",
    author="LatticeWeaver Team",
    author_email="team@latticeweaver.dev",
    description="Framework universal para modelar fenómenos complejos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/latticeweaver/lattice-weaver",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "dash>=2.9.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "mypy>=1.3.0",
        ],
        "viz": [
            "dash-cytoscape>=0.3.0",
            "dash-bootstrap-components>=1.4.0",
        ],
        "all": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "mypy>=1.3.0",
            "dash-cytoscape>=0.3.0",
            "dash-bootstrap-components>=1.4.0",
        ],
    },
)
