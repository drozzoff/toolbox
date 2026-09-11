import importlib
import pkgutil

import toolbox.dashboard_profiles


module_names = [
    module.name
    for module in pkgutil.walk_packages(
        toolbox.dashboard_profiles.__path__,
        prefix = f"{toolbox.dashboard_profiles.__name__}.",
    )
]

for module_name in module_names:
    importlib.import_module(module_name)
    print(f"Imported {module_name}")

print(f"Successfully imported {len(module_names)} dashboard profile modules")