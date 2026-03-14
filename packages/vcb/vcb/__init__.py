import os

# By default, set the logging level to INFO.
if os.environ.get("LOGURU_LEVEL") is None:
    os.environ["LOGURU_LEVEL"] = "INFO"
