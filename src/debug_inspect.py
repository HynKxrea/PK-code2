#!/usr/bin/env python3
import sys, os
# Ensure src on path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
try:
    import util.util as u
except Exception:
    import importlib
    u = importlib.import_module('util.util')

from PIL import Image
import numpy as np
import cv2

if len(sys.argv) < 2:
    print('Usage: python src/debug_inspect.py <image_path>')
    sys.exit(1)

path = sys.argv[1]
if not os.path.exists(path):
    print('File not found:', path)
    sys.exit(1)

print('Inspecting:', path)

# Run analyze_image
try:
    red_pixels, used_red_pixels, piece_count = u.analyze_image(path)
    print('analyze_image -> red_pixels:', red_pixels, 'used_red_pixels:', used_red_pixels, 'piece_count:', piece_count)
except Exception as e:
    print('analyze_image raised:', repr(e))

# Compute HSV stats
im = Image.open(path).convert('RGBA')
arr = np.array(im)
R = arr[:,:,0]; G = arr[:,:,1]; B = arr[:,:,2]; A = arr[:,:,3]
bgr = np.dstack((B,G,R))
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H = hsv[:,:,0].ravel()
S = hsv[:,:,1].ravel()
V = hsv[:,:,2].ravel()
print('HSV min/median/max H,S,V:')
print(int(H.min()), int(np.median(H)), int(H.max()))
print(int(S.min()), int(np.median(S)), int(S.max()))
print(int(V.min()), int(np.median(V)), int(V.max()))

# Try various red threshold combos
print('\nSample counts for red thresholds (h_tol, s_min, v_min):')
for ht in (6,10,15,20,30):
    for smin in (20,30,40,50):
        for vmin in (20,30,40,50):
            lower1 = (H <= ht) & (S >= smin) & (V >= vmin)
            lower2 = (H >= 180-ht) & (S >= smin) & (V >= vmin)
            c = int(np.count_nonzero(lower1 | lower2))
            if c>0:
                print(f'h={ht:2d} s>={smin:2d} v>={vmin:2d} -> {c}')

# Save small preview of masks using the current defaults
try:
    red_mask_vis = cv2.imread(os.path.join(os.getcwd(),'src','util','dummy.png'))
except Exception:
    pass

print('\nDone')
