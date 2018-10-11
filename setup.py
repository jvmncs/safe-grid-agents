from setuptools import setup, find_packages

setup(
    name="safe_grid_agents",
    version="0.0.1-rc",
    packages=["ai_safety_gridworlds", "safe_grid_agents"],
    package_dir={"ai_safety_gridworlds": "ai-safety-gridworlds/ai_safety_gridworlds"},
    install_requires=["numpy==1.14.5", "pycolab", "absl-py", "pyyaml", "moviepy"],
    license="",
    long_description=open("README.md").read(),
)
