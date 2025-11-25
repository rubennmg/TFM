import pkgutil
import importlib
import pathlib

package_path = pathlib.Path(__file__).parent
package_name = __name__

for module in pkgutil.walk_packages([str(package_path)], prefix=f"{package_name}."):
    importlib.import_module(module.name)
