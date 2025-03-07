from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="gpt2-training-project",
    version="0.1.0",
    author="Student",
    author_email="student@example.com",
    description="GPT2 Training and Fine-tuning Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gpt2-training-project",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-gpt2=gpt2_training:main",
            "finetune-gpt2=lora_finetune:main",
        ],
    },
)
