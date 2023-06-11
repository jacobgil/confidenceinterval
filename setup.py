import setuptools

with open('README.md', mode='r', encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name='confidenceinterval',
    version='1.0.3',
    author='Jacob Gildenblat',
    author_email='jacob.gildenblat@gmail.com',
    description='Confidence Intervals in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jacobgil/confidenceinterval',
    project_urls={
        'Bug Tracker': 'https://github.com/jacobgil/confidenceinterval/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=requirements)
