"""
CSL-Core Venom: host-wide agent discovery, policy workbench and mapping test.

Imported lazily by the `cslcore setup` / `cslcore venom` commands only; `import
chimera_core` never loads this package.
"""

from .. import __version__ as VENOM_VERSION  # one version for the package

__all__ = ["VENOM_VERSION"]
