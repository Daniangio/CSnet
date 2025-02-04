from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "csnet/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="CSnet",
    version=version,
    author="Daniele Angioletti, ---",
    description="CSnet.",
    python_requires=">=3.8",
    packages=find_packages(include=["csnet", "csnet.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "csnet-build = csnet.scripts.build_dataset:main",
        ]
    },
    install_requires=[
        "MDAnalysis",
        "pandas",
        "plotly",
        "gpytorch",
    ],
    zip_safe=True,
)