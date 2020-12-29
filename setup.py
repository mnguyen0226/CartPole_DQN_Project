from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="CartPole",
    version="0.1",
    author="Minh Nguyen",
    author_email="mnguyen0226@vt.edu",
    description="Reinforcement Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    license="GNU",
    python_requires=">=3.5.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "pandas",
        "gym",
        "torch",
        "Pillow",
        "matplotlib",
        "torchvision"
    ],
    include_package_data=True,
)
