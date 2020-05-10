#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["numpy>=1.18", "matplotlib>=3.1", "scipy>=1.4", "pandas>=1.0.3", "tqdm>=4.46.0"]

setup_requirements = [
    "pytest-runner",
]

test_requirements = ["pytest>=5.4", "pytest-cov>=2.8"]

setup(
    author="Germán Riaño",
    author_email="griano@germanriano.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Epidemics Models with Random Infectious Period",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="epistoch",
    name="epistoch",
    packages=find_packages(where="src", include=["epistoch", "epistoch.*", "pyphase"]),
    package_dir={"": "src"},
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/griano/epistoch",
    version="0.1.6",
    zip_safe=False,
)
