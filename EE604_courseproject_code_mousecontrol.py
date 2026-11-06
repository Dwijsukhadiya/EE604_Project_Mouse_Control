"""
Virtual Mouse using Hand Gesture Recognition (improved version)

Gestures
--------
  Index finger up (middle down)       -> move cursor
  Pinch thumb + index (quick)         -> left click
  Pinch thumb + index (hold > 0.45 s) -> drag (mouse held down until release)
  Pinch thumb + middle                -> right click
  Pinch thumb + ring                  -> screenshot (timestamped file)
  Index + middle up, move hand        -> scroll (up/down, proportional)
  Index + middle + ring up, move hand -> zoom in / out
  Fist / anything else                -> idle (nothing happens)

Press 'q' or ESC in the video window to quit.
Every action is logged to gesture_log.csv with its latency, for evaluation.
Dark frames are brightened with adaptive gamma correction before hand detection.
"""
import csv
import math
import os
import platform
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ----------------------------- Configuration -----------------------------
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480
FRAME_MARGIN = 0.15      # ignore this fraction at each frame edge (active region)
SMOOTHING = 0.30         # EMA weight of the new position (lower = smoother, laggier)
PINCH_ON = 0.30          # pinch starts when tip distance < 0.30 x hand size
PINCH_OFF = 0.42         # pinch ends when tip distance > 0.42 x hand size (hysteresis)
FOLD_RATIO = 1.25        # a pinched finger must reach out: tip-wrist > 1.25 x knuckle-wrist
STABLE_FRAMES = 3        # frames a gesture must persist before it counts
ACTION_COOLDOWN = 0.40   # seconds between discrete actions (clicks, screenshots)
DRAG_HOLD = 0.45         # seconds a pinch must be held to become a drag
SCROLL_GAIN = 2500       # scroll units per unit of normalised vertical motion
SCROLL_DEADZONE = 0.004  # ignore tiny vertical jitter while scrolling
ZOOM_STEP = 0.06         # normalised vertical motion per zoom step
POINTER_LANDMARK = 5     # index MCP: stays still during pinches, so clicks don't shift the cursor
SCREENSHOT_DIR = "screenshots"
LOG_FILE = "gesture_log.csv"

pyautogui.PAUSE = 0      # default is 0.1 s after EVERY call, which adds lag to moveTo
pyautogui.FAILSAFE = True  # keep the safety stop; the cursor is clamped away from corners
MOD = "command" if platform.system() == "Darwin" else "ctrl"
SCREEN_W, SCREEN_H = pyautogui.size()

# MediaPipe landmark ids
WRIST, THUMB_TIP = 0, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_PIP, PINKY_TIP = 18, 20


# ------------------------------- Helpers ---------------------------------
def enhance_if_dark(frame, dark_threshold=90, target_mean=120.0):
    """Image processing step: if the frame is dark, brighten it with an adaptive
    gamma curve. Gamma is chosen so the mean brightness maps to target_mean:
        gamma = ln(target / 255) / ln(mean / 255)
    The curve is global and monotonic, so it keeps the relative contrast and
    shading of the hand, which the landmark network relies on."""
    mean = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
    if mean >= dark_threshold:
        return frame
    gamma = math.log(target_mean / 255.0) / math.log(max(mean, 1.0) / 255.0)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(frame, lut)


def dist(a, b, w, h):
    """Euclidean distance in pixels (x and y use the same scale)."""
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def fingers_up(lm):
    """A finger is 'up' when its tip is above its PIP joint (image y grows downward)."""
    return {
        "index": lm[INDEX_TIP].y < lm[INDEX_PIP].y,
        "middle": lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y,
        "ring": lm[RING_TIP].y < lm[RING_PIP].y,
        "pinky": lm[PINKY_TIP].y < lm[PINKY_PIP].y,
    }


def to_screen(nx, ny):
    """Map the active region of the frame to the full screen, clamped off the corners."""
    m = FRAME_MARGIN
    sx = min(max((nx - m) / (1 - 2 * m), 0.0), 1.0)
    sy = min(max((ny - m) / (1 - 2 * m), 0.0), 1.0)
    return 1 + sx * (SCREEN_W - 3), 1 + sy * (SCREEN_H - 3)


class Pinch:
    """Pinch detector with hysteresis so it doesn't flicker at the threshold."""

    def __init__(self):
        self.active = False

    def update(self, ratio):
        if self.active and ratio > PINCH_OFF:
            self.active = False
        elif not self.active and ratio < PINCH_ON:
            self.active = True
        return self.active


class Debouncer:
    """A gesture is accepted only after STABLE_FRAMES consecutive frames."""

    def __init__(self, n):
        self.n, self.candidate, self.count = n, None, 0
        self.value, self.onset = "none", time.perf_counter()

    def update(self, g, t):
        if g == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count, self.cand_onset = g, 1, t
        if self.count >= self.n and self.value != g:
            self.value, self.onset = g, self.cand_onset  # onset = first frame it was seen
        return self.value


def classify(lm, w, h, pinches):
    """Return one raw gesture label for this frame (mutually exclusive)."""
    size = dist(lm[WRIST], lm[MIDDLE_MCP], w, h) + 1e-6  # hand size -> distance-invariant
    ratios = {
        "pinch_index": dist(lm[THUMB_TIP], lm[INDEX_TIP], w, h) / size,
        "pinch_middle": dist(lm[THUMB_TIP], lm[MIDDLE_TIP], w, h) / size,
        "pinch_ring": dist(lm[THUMB_TIP], lm[RING_TIP], w, h) / size,
    }
    # Fold gating: when the thumb merely rests on a finger folded into the palm
    # (as in a pointing pose), that finger's tip is close to the wrist. A real
    # pinch is made with the finger reaching out, so its tip is well beyond its knuckle.
    finger = {"pinch_index": (INDEX_TIP, INDEX_MCP), "pinch_middle": (MIDDLE_TIP, MIDDLE_MCP),
              "pinch_ring": (RING_TIP, RING_MCP)}
    reaching = lambda tip, mcp: (dist(lm[tip], lm[WRIST], w, h)
                                 > FOLD_RATIO * dist(lm[mcp], lm[WRIST], w, h))
    active = [k for k, r in ratios.items() if pinches[k].update(r) and reaching(*finger[k])]
    if active:  # if several are close, the tightest pinch wins
        return min(active, key=ratios.get)

    up = fingers_up(lm)
    if up["index"] and up["middle"] and up["ring"] and not up["pinky"]:
        return "zoom"
    if up["index"] and up["middle"] and not up["ring"] and not up["pinky"]:
        return "scroll"
    if up["index"] and not up["middle"]:
        return "move"
    return "idle"


# --------------------------------- Main ----------------------------------
def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    log_f = open(LOG_FILE, "a", newline="")
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(["timestamp", "event", "latency_ms", "fps"])

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,               # one hand: avoids mixing landmarks from two hands
        model_complexity=0,            # faster model, fine for this task
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    draw = mp.solutions.drawing_utils
    hand_conn = mp.solutions.hands.HAND_CONNECTIONS

    pinches = {k: Pinch() for k in ("pinch_index", "pinch_middle", "pinch_ring")}
    stable = Debouncer(STABLE_FRAMES)
    prev_gesture = "none"
    cur_x = cur_y = None
    pinch_start, dragging = None, False
    last_action = 0.0
    ref_y, scroll_acc, zoom_acc = None, 0.0, 0.0
    fps, t_prev = 0.0, time.perf_counter()

    def record(event):
        lat = (time.perf_counter() - stable.onset) * 1000
        log.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event, f"{lat:.1f}", f"{fps:.1f}"])
        print(f"{event:12s} latency {lat:6.1f} ms")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not received, exiting.")
                break
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            frame = cv2.flip(frame, 1)
            frame = enhance_if_dark(frame)
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            raw, lm = "none", None
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand, hand_conn)
                lm = hand.landmark
                raw = classify(lm, w, h, pinches)
            gesture = stable.update(raw, now)
            entered = gesture != prev_gesture

            # ---- Cursor movement (also during index pinch, so drag works) ----
            if lm is not None and gesture in ("move", "pinch_index"):
                tx, ty = to_screen(lm[POINTER_LANDMARK].x, lm[POINTER_LANDMARK].y)
                if cur_x is None:
                    cur_x, cur_y = tx, ty
                cur_x += SMOOTHING * (tx - cur_x)
                cur_y += SMOOTHING * (ty - cur_y)
                pyautogui.moveTo(cur_x, cur_y)

            # ---- Left click / drag ----
            if gesture == "pinch_index":
                if pinch_start is None:
                    pinch_start = now
                elif not dragging and now - pinch_start > DRAG_HOLD:
                    pyautogui.mouseDown()
                    dragging = True
                    record("drag_start")
            elif pinch_start is not None:  # pinch just released
                if dragging:
                    pyautogui.mouseUp()
                    record("drag_end")
                elif now - last_action > ACTION_COOLDOWN:
                    pyautogui.click()
                    last_action = now
                    record("left_click")
                pinch_start, dragging = None, False

            # ---- Edge-triggered actions: fire once when the gesture starts ----
            if entered and now - last_action > ACTION_COOLDOWN:
                if gesture == "pinch_middle":
                    pyautogui.rightClick()
                    last_action = now
                    record("right_click")
                elif gesture == "pinch_ring":
                    path = os.path.join(SCREENSHOT_DIR, time.strftime("shot_%Y%m%d_%H%M%S.png"))
                    pyautogui.screenshot().save(path)
                    last_action = now
                    record("screenshot")

            # ---- Continuous scroll / zoom from vertical hand motion ----
            if gesture in ("scroll", "zoom") and lm is not None:
                y = lm[MIDDLE_MCP].y
                if entered or ref_y is None:
                    ref_y, scroll_acc, zoom_acc = y, 0.0, 0.0
                dy = ref_y - y  # positive when the hand moves up
                ref_y = y
                if gesture == "scroll" and abs(dy) > SCROLL_DEADZONE:
                    scroll_acc += dy * SCROLL_GAIN
                    n = int(scroll_acc)
                    if n:
                        pyautogui.scroll(n)
                        scroll_acc -= n
                        record("scroll_up" if n > 0 else "scroll_down")
                elif gesture == "zoom":
                    zoom_acc += dy
                    while zoom_acc > ZOOM_STEP:
                        pyautogui.hotkey(MOD, "=")  # '=' is the unshifted '+' key
                        zoom_acc -= ZOOM_STEP
                        record("zoom_in")
                    while zoom_acc < -ZOOM_STEP:
                        pyautogui.hotkey(MOD, "-")
                        zoom_acc += ZOOM_STEP
                        record("zoom_out")
            else:
                ref_y = None

            prev_gesture = gesture

            # ---- HUD ----
            m = FRAME_MARGIN
            cv2.rectangle(frame, (int(m * w), int(m * h)), (int((1 - m) * w), int((1 - m) * h)),
                          (255, 0, 255), 1)
            cv2.putText(frame, f"{gesture}{' (drag)' if dragging else ''}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS {fps:.0f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Virtual Mouse", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if dragging:
            pyautogui.mouseUp()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        log_f.close()


if __name__ == "__main__":
    main()
