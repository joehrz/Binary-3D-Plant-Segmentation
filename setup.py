from setuptools import setup, find_packages

setup(
    name='plant_point_cloud_segmentation',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'torchvision',
        'torchaudio',
        'pointnet2',
        'open3d',
        'matplotlib',
        'scikit-learn',
        'PyYAML',
        'tqdm'
    ],
)