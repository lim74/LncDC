from setuptools import setup, find_packages

setup(
    name = 'lncDC',
    version = '1.2',
    author = 'Minghua Li',
    author_email = 'lim74@miamioh.edu',
    description = 'a tool to predict the probability of a transcript being a long noncoding rna',
    url = '',
    license = 'MIT',
    packages = find_packages(),
    install_requires = [
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'imbalanced-learn',
        'biopython'
    ],
    package_data = {
        'data':['*.csv', '*.pkl'],
        'test':['*.fasta'],},
    include_package_data = True,
    python_requires = '>=3.9'
)
