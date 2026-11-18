# Edited Browser-Use provenance

Restored from `anasalsawy/sawii-persona` commit
`74ca8a9724fd01f2437eb254e300e4c03ebc1559`, including its
`vendor/persona-lanes` shim and `scripts/install_browser_sdk.py`.
`SOURCE.json` records the original patch source, exact Browser-Use commit,
and SHA-256 checksums. The installer verifies them before applying patches.

Run `python scripts/install_browser_sdk.py` with Python 3.11+, Git, and Chrome
available. It installs the pinned patched source, not stock browser-use.
The clone lives in ignored `.browser-use-sdk/`; the installed SDK retains
upstream's license. Use a clean destination for a reinstall.

Hands uses only the patched browser session/capture/tool layer. It does not
instantiate `PersonaBrowserAgent` or add another inference owner. Persona
still owns L/R decisions. Installing this SDK is not a security audit of its
transitive dependencies.
