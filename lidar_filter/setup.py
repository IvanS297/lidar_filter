from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lidar_filter'

setup(
    name=package_name,
    version='1.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ivan',
    maintainer_email='ivan.ser202@gmail.com',
    description='Package for patching 2d laser scan from glass errors. See more: https://github.com/IvanS297/lidar_filter and https://www.youtube.com/channel/UC8cQtZ49KFYy3E-_9oM-eKg/',
    license='GNU',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "glass_node=lidar_filter.node:main",
        ],
    },
)
