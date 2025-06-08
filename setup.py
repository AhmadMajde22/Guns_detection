from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line and not line.startswith("--")]

setup(
    name="Guns_Detection",
    version="0.1.0",
    author="Ahmad Majdi",
    packages=find_packages(),
    install_requires=requirements,
)
