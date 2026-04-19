"""
Data diagnostic script — analyzes training data to find why loss is stuck.
Run: python diagnose_data.py
"""
import csv
import os
import sys
import cv2
import numpy as np
from collections import Counter

from config import DATASETS_DIR, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT
from trainer import crop_and_resize

def analyze_dataset(dataset_dir):
    csv_path = os.path.join(dataset_dir, "data.csv")
    if not os.path.isfile(csv_path):
        print(f"  No data.csv found!")
        return

    lefts = []
    rights = []
    rows = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            left = float(row["left"])
            right = float(row["right"])
            lefts.append(left)
            rights.append(right)
            rows.append(row)

    lefts = np.array(lefts)
    rights = np.array(rights)

    total = len(lefts)
    print(f"  Total frames: {total}")
    print()

    # ── Command Distribution ──
    idle = np.sum((np.abs(lefts) < 5) & (np.abs(rights) < 5))
    forward = np.sum((lefts > 5) & (rights > 5) & (np.abs(lefts - rights) < 20))
    backward = np.sum((lefts < -5) & (rights < -5))
    turn_left = np.sum((rights - lefts) > 15)
    turn_right = np.sum((lefts - rights) > 15)

    print(f"  === COMMAND DISTRIBUTION ===")
    print(f"  Idle (both ~0):    {idle:4d} ({100*idle/total:.1f}%)")
    print(f"  Forward:           {forward:4d} ({100*forward/total:.1f}%)")
    print(f"  Backward:          {backward:4d} ({100*backward/total:.1f}%)")
    print(f"  Turn Left:         {turn_left:4d} ({100*turn_left/total:.1f}%)")
    print(f"  Turn Right:        {turn_right:4d} ({100*turn_right/total:.1f}%)")
    other = total - idle - forward - backward - turn_left - turn_right
    print(f"  Mixed/Other:       {other:4d} ({100*other/total:.1f}%)")
    print()

    # ── Value Ranges ──
    print(f"  === VALUE RANGES ===")
    print(f"  Left motor:  min={lefts.min():.0f}  max={lefts.max():.0f}  mean={lefts.mean():.1f}  std={lefts.std():.1f}")
    print(f"  Right motor: min={rights.min():.0f}  max={rights.max():.0f}  mean={rights.mean():.1f}  std={rights.std():.1f}")
    print()

    # ── Check for highly similar consecutive frames ──
    same_cmd_streak = 0
    max_streak = 0
    for i in range(1, total):
        if lefts[i] == lefts[i-1] and rights[i] == rights[i-1]:
            same_cmd_streak += 1
            max_streak = max(max_streak, same_cmd_streak)
        else:
            same_cmd_streak = 0

    unique_cmds = len(set(zip(lefts.tolist(), rights.tolist())))
    print(f"  === DIVERSITY ===")
    print(f"  Unique command pairs: {unique_cmds}")
    print(f"  Longest same-command streak: {max_streak} frames")
    print()

    # ── Key Problem Detection ──
    print(f"  === DIAGNOSIS ===")
    problems = []

    if idle / total > 0.3:
        problems.append(f"TOO MANY IDLE FRAMES ({100*idle/total:.0f}%) - car was stopped most of the time")

    if forward / total > 0.7:
        problems.append(f"MOSTLY FORWARD ({100*forward/total:.0f}%) - not enough turns for model to learn steering")

    if turn_left + turn_right < total * 0.15:
        problems.append(f"TOO FEW TURNS ({100*(turn_left+turn_right)/total:.0f}%) - model can't learn steering")

    if unique_cmds < 10:
        problems.append(f"VERY FEW UNIQUE COMMANDS ({unique_cmds}) - data lacks variety")

    if lefts.std() < 15 or rights.std() < 15:
        problems.append(f"LOW VARIANCE (L std={lefts.std():.0f}, R std={rights.std():.0f}) - commands too uniform")

    if max_streak > total * 0.1:
        problems.append(f"LONG IDENTICAL STREAKS ({max_streak} frames) - redundant data")

    if total < 1500:
        problems.append(f"SMALL DATASET ({total} frames) - need at least 3000+ for decent training")

    if not problems:
        print("  No obvious problems detected. Data looks reasonable.")
    else:
        for p in problems:
            print(f"  !! {p}")

    print()

    # ── Save sample grid for visual inspection ──
    images_dir = os.path.join(dataset_dir, "images")
    if os.path.isdir(images_dir):
        sample_indices = np.linspace(0, total-1, min(16, total), dtype=int)
        grid_imgs = []

        for idx in sample_indices:
            img_path = os.path.join(dataset_dir, rows[idx]["image_path"])
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Show what the model sees (cropped)
                    cropped = crop_and_resize(img)
                    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

                    # Add command overlay
                    l, r = int(float(rows[idx]["left"])), int(float(rows[idx]["right"]))
                    cv2.putText(cropped_bgr, f"L:{l} R:{r}", (5, 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                    grid_imgs.append(cropped_bgr)

        if grid_imgs:
            # Create grid (4 columns)
            cols = 4
            rows_n = (len(grid_imgs) + cols - 1) // cols
            while len(grid_imgs) < rows_n * cols:
                grid_imgs.append(np.zeros_like(grid_imgs[0]))

            grid_rows = []
            for r in range(rows_n):
                row_imgs = grid_imgs[r*cols:(r+1)*cols]
                grid_rows.append(np.hstack(row_imgs))
            grid = np.vstack(grid_rows)

            out_path = os.path.join(dataset_dir, "diagnostic_grid.jpg")
            cv2.imwrite(out_path, grid)
            print(f"  Saved visual grid: {out_path}")
            print(f"  (Open this image to see what the model sees after cropping)")


if __name__ == "__main__":
    print("=" * 60)
    print("  DATA DIAGNOSTIC")
    print("=" * 60)

    if not os.path.exists(DATASETS_DIR):
        print("No datasets directory found!")
        sys.exit(1)

    sessions = [d for d in os.listdir(DATASETS_DIR)
                if os.path.isdir(os.path.join(DATASETS_DIR, d))]

    if not sessions:
        print("No recorded sessions found!")
        sys.exit(1)

    for session in sorted(sessions):
        print(f"\n--- Session: {session} ---")
        analyze_dataset(os.path.join(DATASETS_DIR, session))

    print("=" * 60)
    print("DONE")
