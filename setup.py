from setuptools import setup, find_packages


setup(
      name='mm',
      version='v0.10',
      author='Ji Won Park',
      packages=find_packages(),
      license='LICENSE.md',
      description='Methods for inference of external convergence',
      long_description=open("README.md").read(),
      long_description_content_type='text/markdown',
      url='https://github.com/jiwoncpark/gaussian-mixture-models',
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      keywords='statistics'
      )
