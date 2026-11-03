from pathlib import Path
from setuptools import setup

package_name = 'kinova_marker_pick'
data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml', 'README.md', 'marker_id7.png', 'marker_id8.png']),
]
for directory in ['launch', 'config', 'models', 'worlds']:
    for path in sorted(Path(directory).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            data_files.append(('share/' + package_name + '/' + str(path.parent), [str(path)]))
setup(
    name=package_name, version='0.2.0', packages=[package_name], data_files=data_files,
    install_requires=['setuptools'], zip_safe=True, maintainer='research',
    maintainer_email='research@example.com', license='Apache-2.0',
    description='Kinova Gen3 Lite Gazebo marker-guided cable peg pickup and insertion',
    tests_require=['pytest'],
    entry_points={'console_scripts':['marker_pick = kinova_marker_pick.marker_pick:main']},
)
