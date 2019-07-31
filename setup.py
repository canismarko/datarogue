from setuptools import setup

setup(name='datarogue',
      version='0.1',
      description='Tools for extracting data from plots.',
      url='http://github.com/canismarko/datarogue',
      author='Mark Wolfman',
      author_email='canismarko@gmail.com',
      entry_points={
          'console_scripts': ['datarogue-train=datarogue.training:train_all_networks'],
      },
      license='GPLv3',
      packages=['datarogue'],
      zip_safe=False,
)
