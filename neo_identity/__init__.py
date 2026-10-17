"""Neo Identity: Krea 2 Identity Edit V2 for Forge Neo.

The package is the extension's engine; ``scripts/neo_identity_edit.py`` is the
always-on script that drives it from the txt2img and img2img tabs.

Nothing here imports the WebUI at module scope. Every host interaction goes
through a function that imports what it needs when it is called, so the package
can be imported, read and tested without Forge present, and so a host that is
not Forge Neo fails the compatibility check instead of failing an import.
"""

__version__ = "1.0.0"
