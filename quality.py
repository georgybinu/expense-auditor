from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def get_blur_score(image_path: Path) -> float:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_image_blurry(image_path: Path, threshold: float = 100.0) -> bool:
    return get_blur_score(image_path) < threshold
