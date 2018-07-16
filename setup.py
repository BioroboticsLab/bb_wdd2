from distutils.core import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]

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
    dependency_links=dep_links,
    extras_require={
        'Flea3': ['PyCapture2'],
    },
    packages=[
        'wdd',
        'wdd.scripts',
    ],
)
