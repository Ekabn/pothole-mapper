import re
import cv2
import numpy as np
import pytesseract

CROP_X_FRAC = 0.35
CROP_Y_FRAC = 0.09

GPS_REGEX = re.compile(
    r"(\d{3,4}\.\d{3,4}),?\s*([NS]).{0,15}?(\d{4,5}\.\d{3,4}),?\s*([EW])"
)

def crop_overlay(frame):
    h, w = frame.shape[:2]
    return frame[0:int(h*CROP_Y_FRAC), 0:int(w*CROP_X_FRAC)]

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, (gray.shape[1]*3, gray.shape[0]*3), interpolation=cv2.INTER_CUBIC)
    _, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    return gray

def nmea_to_decimal(value, direction):
    value = float(value)
    degrees = int(value / 100)
    minutes = value - degrees * 100
    decimal = degrees + minutes / 60.0
    if direction in ("S", "W"):
        decimal = -decimal
    return decimal

def read_gps_from_frame(frame):
    crop = crop_overlay(frame)
    processed = preprocess_for_ocr(crop)
    text = pytesseract.image_to_string(
        processed,
        config="--psm 6 -c tessedit_char_whitelist=0123456789.,NSEWkmhKM/: "
    )

    match = GPS_REGEX.search(text)
    if not match:
        return None

    lat_raw, lat_dir, lon_raw, lon_dir = match.groups()
    try:
        lat = nmea_to_decimal(lat_raw, lat_dir)
        lon = nmea_to_decimal(lon_raw, lon_dir)
    except ValueError:
        return None

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None

    return lat, lon
