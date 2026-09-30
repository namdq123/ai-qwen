"""Keep API tests from downloading model weights."""

import os

os.environ["QWEN_DOWNLOAD_ON_START"] = "false"
