from setuptools import setup, find_packages

DEPENDENCIES = [
	'xsuite',
	'pandas',
	'rich',
	'numpy',
	'matplotlib',
	'scikit-learn',
	'scikit-image',
	'seaborn',
	'h5py',
	'ipython',
	'ipywidgets',
	'scipy',
	'tqdm',
]

OPTIONAL_DEPENDENCIES = {
	"dashboard": [
		"profile-dashboard @ git+https://github.com/drozzoff/dashboard.git",
	],
}

setup(
	name = "toolbox",
	version = "0.0.1",
	description = "Some functionality aimed to be used with xsuite",
	author = "Andrii Pastushenko",
	url = "https://github.com/drozzoff/toolbox",
	python_requires = ">=3.10",
	license = "MIT",
	
	packages = find_packages(include = ["toolbox", "toolbox.*"]),
	package_data = {"toolbox.visualisation": ["default_style.json"]},
	install_requires = DEPENDENCIES,
	extras_require = OPTIONAL_DEPENDENCIES,
	classifiers = [
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Natural Language :: English",
		"Programming Language :: Python",
		"Programming Language :: Python :: 3 :: Only",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Topic :: Scientific/Engineering :: Physics",
	],
)
