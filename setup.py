from setuptools import setup, find_packages

setup(
    name = 'lncDC',
    version = '1.3.5',
    author = 'Minghua Li',
    author_email = 'lim74@miamioh.edu',
    description = 'A tool for predicting the probability of a transcript being a long noncoding rna',
    url = 'https://github.com/lim74/LncDC',
    license = 'MIT',
    packages = find_packages(),
    install_requires = [
        'pandas>=1.5',
        'numpy>=1.23',
        'scikit-learn==1.1.3',
        'xgboost==1.7.1',
        'imbalanced-learn>=0.9.1',
        'biopython>=1.79',
        'tqdm>=4.64'
    ],
    package_data = {
        'data':['*.csv', '*.pkl'],
        'test':['*.fasta'],},
    include_package_data = True,
    python_requires = '>=3.9'
)
