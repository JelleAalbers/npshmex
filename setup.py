import setuptools

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [x.strip().split('=')[0]
                for x in f.readlines()]

setuptools.setup(name='npshmex',
                 version='0.0.3',
                 description=('ProcessPoolExecutor that passes numpy'
                              'arrays through shared memory'),
                 py_modules=['multihist'],
                 install_requires=['SharedArray'],
                 setup_requires=['pytest-runner'],
                 tests_require=requires + ['pytest'],
                 zip_safe=False)
