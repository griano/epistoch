from setuptools import setup

setup(
    name="EpiStoch",
    version="1.0",
    packages=["pyphase", "epi_stoch", "epi_stoch.utils", "epi_stoch.experimental"],
    package_dir={"": "src"},
    url="",
    license="MIT",
    author="Germán Riaño",
    author_email="griano@jmarkov.org",
    description="Epidemiology Models with General Random Distribution",
)
