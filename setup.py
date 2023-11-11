import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boar_pv", # Replace with your own username
    version="1.0.6",
    author="Vincent Le Corre, Larry Lueer",
    author_email="vincent.le.corre@fau.de",
    description="High throughput parameter extraction and experimental design with Bayesian optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD 3-Clause License',
    url="https://github.com/i-MEET/boar",
    download_url="https://github.com/i-MEET/boar/archive/refs/tags/v1.0.4.tar.gz",
    packages=setuptools.find_packages(),
    readme = "README.md",
    keywords=['Bayesian optimization', 'parameter extraction', 'experimental design', 'high throughput', 'solar cells'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires='>=3.9',
    install_requires = [
        'numpy>=1.2',
        'pandas>=1.4',
        'matplotlib>=3.5',
        'notebook>=6.0',
        'jupyterlab>=3.4',
        'ipympl>=0.9',
        'seaborn>=0.11',
        'scipy>=1.0',
        'scikit-optimize>=0.9',
        'tqdm>=4.65',
        'parmap>=1.5',
        'sqlalchemy>=1.4',
        'torch>=2.0',
        'torchvision>=0.15',
        'torchaudio>=2.0',
        'ax-platform>=0.3',
    ],
    extras_require = {
        'dev': [
            'pytest',
            'twine',
        ],
    },
    include_package_data=True,
        
)
