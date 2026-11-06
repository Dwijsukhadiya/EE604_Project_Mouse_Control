# Experiments (Section 4 of the report)

These scripts reproduce the results tables and figures of the report on sample images from the public HaGRID dataset (https://github.com/hukenovs/hagrid). They import the gesture logic directly from the project's program, so the results reflect the actual code.

## Setup

Run these commands inside the `experiments` folder (Python 3.9 to 3.12).

Windows (PowerShell):

```powershell
python -m pip install mediapipe==0.10.21 opencv-python matplotlib numpy
copy ..\EE604_courseproject_code_mousecontrol.py virtual_mouse_improved.py
curl.exe -L -o gestures.png https://raw.githubusercontent.com/hukenovs/hagrid/master/images/gestures.png
mkdir ..\fig
```

macOS / Linux:

```bash
python3 -m pip install mediapipe==0.10.21 opencv-python matplotlib numpy
cp ../EE604_courseproject_code_mousecontrol.py virtual_mouse_improved.py
curl -L -o gestures.png https://raw.githubusercontent.com/hukenovs/hagrid/master/images/gestures.png
mkdir -p ../fig
```

## Run

| Command | Report section | Output |
|---|---|---|
| `python exp1_logic.py` | 4.2 Gesture interpretation | Table 2 numbers, `exp1_rows.json` (per-gesture table in the appendix) |
| `python exp2b.py` | 4.3 Robustness to low light | Detection rates for all five preprocessing methods (Table 3) |
| `python exp_final.py` | 4.3 to 4.5 Low light, cursor jitter, processing time | `results.json` |
| `python make_figs.py` | Figures 2, 4 and 5 | Figure files in `../fig` (Figure 3 is drawn in LaTeX from the Table 3 data) |

## Notes

- PyAutoGUI is replaced by a stub, so the scripts need no display and never move your mouse.
- `results.json` and `exp1_pergesture.json` hold the results reported in the paper, for comparison.
- Timing results depend on the CPU; the report used an Intel Xeon at 2.1 GHz without a GPU.
- The test frames are built from the HaGRID example image `gestures.png`: 26 single-hand classes, each placed at three sizes and three positions in a 640x480 frame (234 frames).
