from setuptools import setup, find_packages

setup(
    name='video_subs',
    version='1.0.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'video_subs=video_subs.main:main',  # Correct this line as necessary
        ],
    },
    # Other setup parameters...
)
