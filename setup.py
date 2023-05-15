import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boar", # Replace with your own username
    version="0.0.1",
    author="Vincent Le Corre, Larry Lueer",
    author_email="larry.lueer@fau.de",
    description="High throughput parameter extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.hte.group/lecorrev/boar",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    setup_requires=["wheel"]
)