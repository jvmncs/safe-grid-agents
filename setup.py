from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools

    use_setuptools()
    import setuptools

setuptools.setup(
    name="safe-grid-agents",
    version="0.1.0dev",
    description="Training (hopefully) safe agents in gridworlds.",
    long_description=open("README.md").read(),
    url="https://github.com/jvmancuso/safe-grid-agents/",
    author="Jason Mancuso",
    author_email="jason@manc.us",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: Console :: Curses",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Games/Entertainment :: Arcade",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Testing",
    ],
    keywords=(
        "ai "
        "artificial intelligence "
        "gridworld "
        "gym "
        "rl "
        "reinforcement learning "
    ),
    install_requires=["safe-grid-gym", "pyyaml", "moviepy", "tensorboardX"],
    dependency_links=[
        "https://github.com/david-lindner/safe-grid-gym/tarball/merge-toys#egg=safe-grid-gym-0.1"
    ],
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
    # test_suite="safe_grid_agents.tests",
)
