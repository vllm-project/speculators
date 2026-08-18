from typing import Any


def get_image_ref(item: dict[str, Any]) -> Any | None:
    """Extract a serializable image reference from a multimodal content item."""
    if item.get("type") not in ("image", "image_url", "input_image"):
        return None

    for key in ("image", "path", "url"):
        image_ref = item.get(key)
        if image_ref is not None:
            return image_ref

    image_url = item.get("image_url")
    if isinstance(image_url, dict):
        return image_url.get("url")
    return image_url
