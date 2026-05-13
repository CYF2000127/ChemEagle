# chemeagle-client

Tiny Python SDK for the ChemEagle public API.

## Install

```bash
pip install -e .   # from inside api/sdk/
```

(Or just copy `chemeagle_client/client.py` into your project — its only
dependency is `requests`.)

## Quick start

```python
from chemeagle_client import ChemEagleClient

client = ChemEagleClient(
    base_url="https://app.chemeagle.net",
    api_key="ce_xxxxxxxxxxxx",   # or set $CHEMEAGLE_API_KEY
)

# 1) Single image, synchronous
result = client.process_image("scheme.png", sync=True)
print(result["reactions"][0]["smiles"])

# 2) PDF, asynchronous (recommended — PDFs take minutes)
task_id = client.process_pdf("paper.pdf")["task_id"]
final = client.wait_for_task(task_id, poll=3.0, max_wait=1800)
for img_result in final["results"]:
    print(img_result["image_name"], len(img_result.get("reactions", [])))

# 3) From a URL
client.process_url("https://example.org/paper.pdf", kind="pdf")
```

## Errors

All non-success responses raise `ChemEagleError`:

```python
from chemeagle_client import ChemEagleError
try:
    client.process_image("x.png", sync=True)
except ChemEagleError as e:
    print(e.error_code, e.message, e.http_status)
```

See the full endpoint reference in `../docs/api.md`.
