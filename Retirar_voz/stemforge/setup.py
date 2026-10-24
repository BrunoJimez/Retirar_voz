from setuptools import setup, find_packages

setup(
    name="stemforge",
    version="1.0.0",
    description="Separação profissional de stems musicais com IA aberta",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24", "scipy>=1.10", "soundfile>=0.12", "librosa>=0.10",
        "pyloudnorm>=0.1.1", "psutil>=5.9", "demucs>=4.0.1",
        "noisereduce>=3.0",
    ],
    entry_points={"console_scripts": ["stemforge=stemforge.cli:main"]},
)
