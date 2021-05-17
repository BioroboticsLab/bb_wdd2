from distutils.core import setup

def parse_requirements(filename):
    with open(filename, "r") as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith("#")]

reqs = parse_requirements("requirements.txt")

setup(
    name='bb_wdd',
    version='2.0.0',
    description='',
    entry_points={
        'console_scripts': [
            'bb_wdd = wdd.scripts.bb_wdd:main',
        ]
    },
    install_requires=reqs,
    extras_require={
        'Flea3': ['PyCapture2'],
    },
    packages=[
        'wdd',
        'wdd.scripts',
    ],
)
