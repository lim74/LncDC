from setuptools import setup, find_packages

setup(
    name = 'lncDC',
    version = '1.3.1',
    author = 'Minghua Li',
    author_email = 'lim74@miamioh.edu',
    description = 'A tool for predicting the probability of a transcript being a long noncoding rna',
    url = '',
    license = 'MIT',
    packages = find_packages(),
    install_requires = [
        'pandas>=1.4.2',
        'numpy>=1.23',
        'scikit-learn>=1.1.1',
        'xgboost>=1.6.1',
        'imbalanced-learn>=0.9.1',
        'biopython>=1.79',
        'tqdm>=4.64.0'
    ],
    package_data = {
        'data':['*.csv', '*.pkl'],
        'test':['*.fasta'],},
    include_package_data = True,
    python_requires = '>=3.9'
)
