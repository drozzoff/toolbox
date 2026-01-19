from setuptools import setup, find_packages

DEPENDENCIES = [
	'xtrack',
	'pandas',
	'rich',
	'numpy',
	'matplotlib',
	'plotly',
	'scikit-learn',
	'seaborn',
	'flask-compress'
]

setup(
	name = "toolbox",
	version = "0.0.1-alpha",
	description = "Some functionality to assist Synchroton analysis",
	author = "Andrii Pastushenko",
	url = "to add",
	python_requires = ">=3.7",
	license = "MIT",
	
	packages = ["toolbox"],
	install_requires = DEPENDENCIES,
	classifiers = [
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Natural Language :: English",
		"Programming Language :: Python",
		"Programming Language :: Python :: 3 :: Only",
		"Programming Language :: Python :: 3.7",
		"Topic :: Scientific/Engineering :: Physics",
	],
)