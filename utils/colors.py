"""Color utility functions for website analysis."""

import re
import colorsys
from typing import Tuple, Optional
import numpy as np
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color, sRGBColor

def hex_from_rgb(r: int, g: int, b: int) -> str:
    """Convert RGB values (0-255) to hex color string."""
    return f"#{r:02x}{g:02x}{b:02x}"

def rgb_from_css(col: str) -> Optional[Tuple[int, int, int]]:
    """Convert CSS color string to RGB tuple (0-255)."""
    # Handle hex colors
    if col.startswith('#'):
        col = col.lstrip('#')
        if len(col) == 3:
            col = ''.join(c + c for c in col)
        try:
            return tuple(int(col[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return None
    
    # Handle rgb/rgba colors
    rgb_match = re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', col)
    if rgb_match:
        return tuple(int(x) for x in rgb_match.groups())
    
    # Handle hsl/hsla colors
    hsl_match = re.match(r'hsla?\(\s*(\d+)\s*,\s*(\d+)%\s*,\s*(\d+)%', col)
    if hsl_match:
        h, s, l = map(int, hsl_match.groups())
        h /= 360
        s /= 100
        l /= 100
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return tuple(int(x * 255) for x in (r, g, b))
    
    return None

def hsl_distance(c1: str, c2: str) -> float:
    """Calculate HSL distance between two colors, wrapping hue at 360°."""
    def to_hsl(color: str) -> Tuple[float, float, float]:
        rgb = rgb_from_css(color)
        if not rgb:
            return (0, 0, 0)
        r, g, b = [x/255 for x in rgb]
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return (h * 360, s * 100, l * 100)
    
    h1, s1, l1 = to_hsl(c1)
    h2, s2, l2 = to_hsl(c2)
    
    # Skip if either color is too light
    if l1 > 95 or l2 > 95:
        return 0
    
    # Calculate hue difference, wrapping at 360°
    h_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
    
    # Calculate saturation and lightness differences
    s_diff = abs(s1 - s2)
    l_diff = abs(l1 - l2)
    
    # Weighted combination (hue is most important)
    return h_diff * 0.7 + s_diff * 0.2 + l_diff * 0.1

def delta_e(color1: str, color2: str) -> float:
    """Calculate color difference using CIEDE2000."""
    def to_lab(color: str) -> LabColor:
        rgb = rgb_from_css(color)
        if not rgb:
            return LabColor(0, 0, 0)
        r, g, b = [x/255 for x in rgb]
        rgb_color = sRGBColor(r, g, b)
        return convert_color(rgb_color, LabColor)
    
    lab1 = to_lab(color1)
    lab2 = to_lab(color2)
    return delta_e_cie2000(lab1, lab2)

def is_neutral(color: str, threshold: float = 5.0) -> bool:
    """Check if a color is neutral (close to white, black, or grey)."""
    rgb = rgb_from_css(color)
    if not rgb:
        return True
    
    # Check distance from white
    if delta_e(color, "#FFFFFF") < threshold:
        return True
    
    # Check distance from black
    if delta_e(color, "#000000") < threshold:
        return True
    
    # Check if it's a grey (all RGB components within 10% of each other)
    r, g, b = rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    if max_val - min_val < 25:  # 10% of 255
        return True
    
    return False 