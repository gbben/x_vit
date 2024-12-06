from datetime import datetime


def formatted_now() -> str:
    """Generated a string-formatted datetime for right now."""
    return datetime.now().strftime("%y-%m-%d__%H-%M")
