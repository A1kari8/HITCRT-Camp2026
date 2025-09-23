from setuptools import find_packages, setup

package_name = 'assets'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['best-blur-video.pt']),
        ('share/' + package_name, ['calibration.json']),
        ('share/' + package_name, ['mytracker.yaml']),
        ('share/' + package_name, ['v1.pt']),
        ('share/' + package_name, ['pytorch_model.bin']),
        ('share/' + package_name, ['config.toml']),
        ('share/' + package_name + '/test1', ['videos/test1/rgb.mp4']),
        ('share/' + package_name + '/test2', ['videos/test2/rgb.mp4']),
        ('share/' + package_name + '/test3', ['videos/test3/rgb.mp4']),
        ('share/' + package_name + '/test4', ['videos/test4/rgb.mp4']),
        ('share/' + package_name + '/test5', ['videos/test5/rgb.mp4']),
    ],
    install_requires=['setuptools'],
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
        ],
    },
)
