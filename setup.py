from setuptools import setup, find_packages

setup(
    name = 'lncDC',
    version = '1.3',
    author = 'Minghua Li',
    author_email = 'lim74@miamioh.edu',
    description = 'a tool to predict the probability of a transcript being a long noncoding rna',
    url = '',
    license = 'MIT',
    packages = find_packages(),
    install_requires = [
        'pandas>=1.4.1',
        'numpy>=1.22.2',
        'scikit-learn>=1.0.2',
        'xgboost>=1.5.2',
        'imbalanced-learn>=0.9.0',
        'biopython>=1.79',
        'tqdm>=4.62.3'
    ],
    package_data = {
        'data':['*.csv', '*.pkl'],
        'test':['*.fasta'],},
    include_package_data = True,
    python_requires = '>=3.9'
)
