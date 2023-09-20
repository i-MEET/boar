import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boar", # Replace with your own username
    version="1.0.0",
    author="Vincent Le Corre, Larry Lueer",
    author_email="vincent.le.corre@fau.de",
    description="High throughput parameter extraction and experimental design with Bayesian optimization",
    long_description=long_description,
    url="https://github.com/i-MEET/boar",
    packages=setuptools.find_packages(),
    readme = "README.md",
    keywords=['Bayesian optimization', 'parameter extraction', 'experimental design', 'high throughput', 'solar cells'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Status :: Beta",
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
        'futures'
    ],
        
)
