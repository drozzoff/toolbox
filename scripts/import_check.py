import importlib
import pkgutil

import toolbox

optional_packages = (
	"toolbox.dashboard_profiles",
)

def is_optional(module_name):
	return any(module_name == package or module_name.startswith(package + ".") for package in optional_packages)

module_names = [
	module.name
	for module in pkgutil.walk_packages(toolbox.__path__, prefix = f"{toolbox.__name__}.")
	if not is_optional(module.name)
]

for module_name in module_names:
	importlib.import_module(module_name)
	print(f"Imported {module_name}")

print(f"Successfully imported toolbox and {len(module_names)} core submodules")