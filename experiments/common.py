import sys, types, cv2, numpy as np
pa = types.ModuleType("pyautogui"); pa.size = lambda: (1920, 1080)
sys.modules["pyautogui"] = pa
import virtual_mouse_improved as vm  # the project's final code
import mediapipe as mp

NAMES = ["call","dislike","fist","four","like","mute","grabbing","grip",
         "ok","one","palm","peace","peace_inv","rock","point","pinkie",
         "stop","stop_inv","three","three2","two_up","two_up_inv","mid_finger","three3",
         "gun","thumb_index"]

def tiles():
    im = cv2.imread("gestures.png"); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    m = cv2.morphologyEx((g < 235).astype(np.uint8), cv2.MORPH_OPEN, np.ones((25,25),np.uint8))
    _, _, st, _ = cv2.connectedComponentsWithStats(m)
    boxes = sorted([tuple(s[:4]) for s in st[1:] if s[2] > 200 and s[3] > 200], key=lambda b:(round(b[1]/100), b[0]))
    out = {}
    for name, (x,y,w,h) in zip(NAMES, boxes):
        out[name] = im[y:y+h, x:x+w]
    return out

def webcam_canvas(tile, size=260, W=640, H=480, dx=0, dy=0):
    """Place a hand tile in a 640x480 frame, roughly the scale of a hand seen by a webcam."""
    t = cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA)
    border = np.median(np.concatenate([t[0], t[-1], t[:,0], t[:,-1]]), axis=0)
    c = np.zeros((H, W, 3), np.uint8); c[:] = border
    x0, y0 = (W-size)//2 + dx, (H-size)//2 + dy
    c[y0:y0+size, x0:x0+size] = t
    # feather the seam
    return c

def detector(static=True):
    return mp.solutions.hands.Hands(static_image_mode=static, max_num_hands=1, model_complexity=0,
                                    min_detection_confidence=0.7, min_tracking_confidence=0.6)

def landmarks(det, frame):
    r = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return r.multi_hand_landmarks[0].landmark if r.multi_hand_landmarks else None
