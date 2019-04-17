import setuptools


readme = open('README.md').read()
history = open('HISTORY.md').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').read().splitlines()

setuptools.setup(name='npshmex',
                 author='Jelle Aalbers',
                 version='0.0.3',
                 description=('ProcessPoolExecutor that passes numpy'
                              'arrays through shared memory'),
                 py_modules=['multihist'],
                 long_description=readme + '\n\n' + history,
                 install_requires=['SharedArray'],
                 setup_requires=['pytest-runner'],
                 tests_require=requirements + ['pytest'],
                 zip_safe=False)
