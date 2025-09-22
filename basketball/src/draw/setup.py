from setuptools import find_packages, setup

package_name = 'draw'

setup(
    name=package_name,
    version='0.5.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','opencv-python'],
    zip_safe=True,
    maintainer='alkaid',
    maintainer_email='xinlai.gao2006@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'draw = draw.draw:main',
        ],
    },
)
