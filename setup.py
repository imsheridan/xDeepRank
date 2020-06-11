import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xdeeprank",
    version="0.0.1",
    author="deepdeliver",
    author_email="xiaoyudong0512@gmail.com",
    description="An eXtensible package of deep learning based ranking models for large-scale industrial recommender system with tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imsheridan/xDeepRank",
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    keywords=['ctr', 'click through rate', 'recommender system', 'industrial', 'deep learning', 'tensorflow'],
)