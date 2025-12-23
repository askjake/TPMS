from setuptools import setup, find_packages

setup(
    name="tpms-tracker",
    version="1.0.0",
    description="Intelligent TPMS Vehicle Pattern Recognition System",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.28.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'plotly>=5.17.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.0',
    ],
    python_requires='>=3.9',
)
