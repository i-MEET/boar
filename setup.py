import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boar_pv", # Replace with your own username
    version="1.0.0",
    author="Vincent Le Corre, Larry Lueer",
    author_email="vincent.le.corre@fau.de",
    description="High throughput parameter extraction and experimental design with Bayesian optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD 3-Clause License',
    url="https://github.com/i-MEET/boar",
    packages=setuptools.find_packages(),
    readme = "README.md",
    keywords=['Bayesian optimization', 'parameter extraction', 'experimental design', 'high throughput', 'solar cells'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = [
        'numpy',
        'pandas',
        'matplotlib',
        'notebook',
        'jupyterlab',
        'ipympl',
        'seaborn',
        'scipy',
        'scikit-optimize',
        'tqdm',
        'parmap',
        'sqlalchemy',
        'pyodbc',
        'openpyxl',
        'futures',
        'torch',
        'torchvision',
        'torchaudio',
        'ax-platform',
    ],
    extras_require = {
        'dev': [
            'pytest',
            'twine',
        ],
    },

        
)
