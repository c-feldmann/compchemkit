from setuptools import setup

setup(
    name="compchemkit",
    version="0.1",
    author="Christian Feldmann",
    license="BSD",
    packages=[
        "compchemkit",
    ],
    author_email="cfeldmann@bit.uni-bonn.de",
    description="Classes and functions useful for chemoinformatics",
    python_requires=">=3.8",
    install_requires=[
        "scipy",
        "bidict",
        "numpy",
        "scikit-learn",
        "pandas",
        "seaborn",
        "matplotlib",
    ],
)
