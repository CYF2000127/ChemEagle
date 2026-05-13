"""
ChemEagle Python SDK.

Install (editable, alongside the server)::

    pip install -e api/sdk

Then::

    from chemeagle_client import ChemEagleClient
    client = ChemEagleClient(base_url="https://app.chemeagle.net", api_key="ce_...")
    result = client.process_image("scheme.png", sync=True)
"""

from .client import ChemEagleClient, ChemEagleError

__all__ = ["ChemEagleClient", "ChemEagleError"]
__version__ = "0.1.0"
