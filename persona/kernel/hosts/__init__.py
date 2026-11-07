"""Hosts: adapters that plug the kernel into somebody else's machinery.

The kernel never imports from this package. A host imports the kernel and hands
it a ``CapabilityBundle``. That direction is the whole point - the day the kernel
needs a host, it stops being a kernel.
"""
from .companion import bundle_for_companion

__all__ = ["bundle_for_companion"]