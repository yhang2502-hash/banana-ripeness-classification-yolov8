import cv2 as cv
import numpy as np
import time
import os
import csv
from collections import deque, Counter
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from rembg import remove, new_session

# =========================================================
# USER SETTINGS
# =========================================================
CAMERA_INDEX = 1
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
PROCESS_W = 1280
PROCESS_H = 720
IMG_SIZE = 224

METHOD1_MODEL_PATH = r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model1_Raw_50epoch\weights\best.pt"
METHOD2_MODEL_PATH = r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model2_Masked_CLAHE_50epoch\weights\best.pt"
METHOD3_MODEL_PATH = r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model3_Isolated_CLAHE_50epoch\weights\best.pt"

# Keep the original trained order internally.
CLASS_NAMES = ["unripe", "ripe", "overripe", "rotten"]

# Display the corrected class naming in the GUI.
DISPLAY_NAME_MAP = {
    "Class A": "Green",
    "Class B": "Partially ripe",
    "Class C": "Ripe",
    "Class D": "Overripe",
    "A": "Green",
    "B": "Partially ripe",
    "C": "Ripe",
    "D": "Overripe",
    "unripe": "Green",
    "ripe": "Partially ripe",
    "overripe": "Ripe",
    "rotten": "Overripe",
    "green": "Green",
    "partially ripe": "Partially ripe",
    "unknown": "Unknown",
    "Unknown": "Unknown",
}


SHORT_DISPLAY_NAME_MAP = {
    "Green": "Green",
    "Partially ripe": "Partially ripe",
    "Ripe": "Ripe",
    "Overripe": "Overripe",
    "Unknown": "Unknown",
}


SAVE_DIR = "snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(SAVE_DIR, "record_result_log.csv")

# =========================================================
# PANEL / DISPLAY SETTINGS
# =========================================================
PANEL_W = 360
IMAGE_H = 270
HEADER_H = 215
SEPARATOR_H = 4
BOTTOM_BAR_H = 58
DISPLAY_SCALE = 1.0
VIDEO_TOTAL_H = HEADER_H + SEPARATOR_H + IMAGE_H + BOTTOM_BAR_H

# =========================================================
# GUI COLOURS
# =========================================================
BG_APP = "#eef3f8"
BG_DARK = (15, 18, 24)
SEPARATOR_COLOR = (38, 44, 56)
# Blue/orange in BGR -> warm blue-ish look in canvas? adjust below
DEFAULT_TEXT_COLOR_BGR = (255, 170, 50)
DEFAULT_TEXT_COLOR_BGR = (255, 190, 80)
DEFAULT_TEXT_COLOR_BGR = (255, 210, 90)
DEFAULT_TEXT_COLOR_BGR = (255, 191, 0)
DEFAULT_TEXT_COLOR_BGR = (255, 180, 0)
DEFAULT_TEXT_COLOR_BGR = (255, 170, 0)
DEFAULT_TEXT_COLOR_BGR = (255, 160, 0)
DEFAULT_TEXT_COLOR_BGR = (255, 150, 0)
# Use a clear blue in BGR.
DEFAULT_TEXT_COLOR_BGR = (255, 140, 0)
GREEN_TEXT_COLOR_BGR = (0, 220, 80)
WHITE_TEXT_COLOR_BGR = (255, 255, 255)
DEFAULT_TEXT_COLOR_TK = "#1f5fa8"
GREEN_TEXT_COLOR_TK = "#1b8f3a"
MUTED_TEXT_COLOR_TK = "#263238"
ACC_GOOD_THRESHOLD = 80.0

# =========================================================
# STABILITY SETTINGS
# =========================================================
HISTORY_LEN = 10

# =========================================================
# LOAD MODELS
# =========================================================
print("Loading YOLO models...")
model1 = YOLO(METHOD1_MODEL_PATH)
model2 = YOLO(METHOD2_MODEL_PATH)
model3 = YOLO(METHOD3_MODEL_PATH)

print("Loading rembg session (silueta)...")
session = new_session("silueta")
print("All models loaded successfully.")


# =========================================================
# CSV LOGGING
# =========================================================
def init_csv_log():
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "snapshot_path",
                "lighting_mode",
                "actual_class",

                "method1_current_label", "method1_current_conf", "method1_is_correct", "method1_recorded_accuracy",
                "method1_majority_label", "method1_majority_conf", "method1_stability", "method1_top2_label", "method1_gap",

                "method2_current_label", "method2_current_conf", "method2_is_correct", "method2_recorded_accuracy",
                "method2_majority_label", "method2_majority_conf", "method2_stability", "method2_top2_label", "method2_gap",

                "method3_current_label", "method3_current_conf", "method3_is_correct", "method3_recorded_accuracy",
                "method3_majority_label", "method3_majority_conf", "method3_stability", "method3_top2_label", "method3_gap",

                "recommended_method",
                "detected_class",
                "best_method_accuracy",
                "system_status"
            ])


def append_csv_log(row):
    with open(CSV_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# =========================================================
# PREPROCESSING METHODS
# =========================================================
def preprocess_method1(frame):
    return frame.copy()


def preprocess_method2_live(frame):
    img = frame.copy()
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    lower_yellow = np.array([15, 30, 30])
    upper_yellow = np.array([45, 255, 255])

    lower_brown_dark = np.array([0, 10, 0])
    upper_brown_dark = np.array([30, 255, 200])

    mask_green = cv.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask_brown_dark = cv.inRange(hsv, lower_brown_dark, upper_brown_dark)

    mask = cv.bitwise_or(mask_green, mask_yellow)
    mask = cv.bitwise_or(mask, mask_brown_dark)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    masked_img = cv.bitwise_and(img, img, mask=mask)

    lab = cv.cvtColor(masked_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    enhanced_lab = cv.merge((l_clahe, a, b))
    enhanced_img = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)

    result = img.copy()
    result[mask > 0] = enhanced_img[mask > 0]
    return result


def preprocess_method3_live(frame):
    img = frame.copy()

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced = cv.cvtColor(cv.merge((l_clahe, a, b)), cv.COLOR_LAB2BGR)

    output = remove(enhanced, session=session)

    if len(output.shape) == 3 and output.shape[2] == 4:
        rgba_planes = cv.split(output)
        mask_ai = rgba_planes[3]
    else:
        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        _, mask_ai = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv.morphologyEx(mask_ai, cv.MORPH_OPEN, kernel)
    mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(
        mask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask_clean)

    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(
            final_mask, [largest_contour], -1, 255, thickness=cv.FILLED)
    else:
        final_mask = mask_clean

    final_img = cv.bitwise_and(enhanced, enhanced, mask=final_mask)
    return final_img


# =========================================================
# INFERENCE
# =========================================================
def display_name(label):
    return DISPLAY_NAME_MAP.get(str(label), str(label))


def short_display_name(label):
    return SHORT_DISPLAY_NAME_MAP.get(label, label)


def predict_frame(model, image):
    results = model.predict(source=image, imgsz=IMG_SIZE, verbose=False)

    if not results or results[0].probs is None:
        return {
            "top1_label": "Unknown",
            "top1_conf": 0.0,
            "top2_label": "Unknown",
            "top2_conf": 0.0,
            "gap": 0.0
        }

    probs_obj = results[0].probs
    probs = probs_obj.data.cpu().numpy()

    sorted_idx = np.argsort(probs)[::-1]
    top1_id = int(sorted_idx[0])
    top2_id = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_id

    if hasattr(results[0], "names"):
        names = results[0].names
        top1_label = names[top1_id]
        top2_label = names[top2_id]
    else:
        top1_label = CLASS_NAMES[top1_id] if top1_id < len(
            CLASS_NAMES) else str(top1_id)
        top2_label = CLASS_NAMES[top2_id] if top2_id < len(
            CLASS_NAMES) else str(top2_id)

    top1_label = display_name(top1_label)
    top2_label = display_name(top2_label)

    top1_conf = float(probs[top1_id])
    top2_conf = float(probs[top2_id])
    gap = top1_conf - top2_conf

    return {
        "top1_label": top1_label,
        "top1_conf": top1_conf,
        "top2_label": top2_label,
        "top2_conf": top2_conf,
        "gap": gap
    }


# =========================================================
# STABILITY / HISTORY
# =========================================================
def update_history(label_hist, conf_hist, new_label, new_conf):
    label_hist.append(new_label)
    conf_hist.append(new_conf)


def compute_majority_and_stability(label_hist, conf_hist):
    if len(label_hist) == 0:
        return "Unknown", 0.0, 0.0

    counts = Counter(label_hist)
    majority_label, majority_count = counts.most_common(1)[0]

    matching_confs = [c for l, c in zip(
        label_hist, conf_hist) if l == majority_label]
    avg_conf = float(sum(matching_confs) / len(matching_confs)
                     ) if matching_confs else 0.0
    stability = (majority_count / len(label_hist)) * 100.0
    return majority_label, avg_conf, stability


# =========================================================
# DRAWING / PANELS
# =========================================================
def draw_panel_lines(img, lines, colors, start_y=34, x=16, scale=0.72, thickness=2, line_gap=27):
    y = start_y
    for line, color in zip(lines, colors):
        cv.putText(img, line, (x, y), cv.FONT_HERSHEY_SIMPLEX,
                   scale, color, thickness, cv.LINE_AA)
        y += line_gap
    return img


def draw_info_text(img, line_items, default_color=(255, 140, 0), start_y=34, x=16, scale=0.72, thickness=2, line_gap=27):
    lines = []
    colors = []
    for item in line_items:
        if isinstance(item, tuple):
            line, color = item
        else:
            line, color = item, default_color
        lines.append(line)
        colors.append(color)
    return draw_panel_lines(img, lines, colors, start_y=start_y, x=x, scale=scale, thickness=thickness, line_gap=line_gap)


def make_panel(image, title, info, actual_class=None):
    header = np.zeros((HEADER_H, PANEL_W, 3), dtype=np.uint8)
    header[:] = (20, 12, 12)

    BLUE = (255, 140, 0)
    GREEN = (0, 255, 0)

    title_color = BLUE
    current_class_color = BLUE
    accuracy_color = BLUE

    selected = str(actual_class).strip()
    current = str(info["current_label"]).strip()
    is_good_acc = (float(info["current_conf"]) * 100.0 >= 80.0)

    if selected and current == selected and is_good_acc:
        title_color = GREEN
        current_class_color = GREEN
        accuracy_color = GREEN

    line_items = [
        (title, title_color),
        (f"Current Class: {info['current_label']}", current_class_color),
        (f"Accuracy: {info['current_conf'] * 100:.2f}%", accuracy_color),
        (f"Majority: {info['majority_label']}", BLUE),
        (f"Stability: {info['stability']:.1f}%", BLUE),
        (f"Top-2: {info['top2_label']}", BLUE),
        (f"Gap: {info['gap'] * 100:.2f}%", BLUE),
    ]
    draw_info_text(header, line_items, default_color=BLUE)

    image_panel = cv.resize(image, (PANEL_W, IMAGE_H))
    separator = np.zeros((SEPARATOR_H, PANEL_W, 3), dtype=np.uint8)
    separator[:] = (45, 45, 45)

    panel = np.vstack([header, separator, image_panel])
    return panel


def add_bottom_bar(canvas, fps, lighting_mode):
    bar = np.zeros((BOTTOM_BAR_H, canvas.shape[1], 3), dtype=np.uint8)
    bar[:] = (24, 24, 24)
    text = f"Lighting: {lighting_mode}   |   FPS: {fps:.2f}"
    cv.putText(bar, text, (18, 38), cv.FONT_HERSHEY_SIMPLEX,
               0.86, WHITE_TEXT_COLOR_BGR, 2, cv.LINE_AA)
    return np.vstack([canvas, bar])


# =========================================================
# GUI APP
# =========================================================
class BananaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Banana Ripeness Classification System")
        self.root.configure(bg=BG_APP)
        self.root.geometry("1500x980")
        self.root.minsize(1500, 800)

        self.lighting_mode = "Unknown"
        self.running = False
        self.cap = None
        self.prev_time = time.time()

        self.last_canvas = None
        self.last_info1 = None
        self.last_info2 = None
        self.last_info3 = None

        self.actual_class = tk.StringVar(value="")

        self.m1_label_hist = deque(maxlen=HISTORY_LEN)
        self.m1_conf_hist = deque(maxlen=HISTORY_LEN)
        self.m2_label_hist = deque(maxlen=HISTORY_LEN)
        self.m2_conf_hist = deque(maxlen=HISTORY_LEN)
        self.m3_label_hist = deque(maxlen=HISTORY_LEN)
        self.m3_conf_hist = deque(maxlen=HISTORY_LEN)

        self.recorded_method1_acc = 0.0
        self.recorded_method2_acc = 0.0
        self.recorded_method3_acc = 0.0

        self.build_ui()

    def build_ui(self):
        title_label = tk.Label(
            self.root,
            text="Banana Ripeness Classification System",
            font=("Arial", 30, "bold"),
            bg=BG_APP,
            fg="black"
        )
        title_label.pack(pady=(12, 8))

        top_frame = tk.Frame(self.root, bg=BG_APP)
        top_frame.pack(pady=(0, 8))

        button_style = {
            "font": ("Arial", 13, "bold"),
            "width": 14,
            "height": 2,
            "bg": "#e8edf3",
            "activebackground": "#d8e5f2",
            "relief": "raised",
            "bd": 2,
        }

        tk.Button(top_frame, text="START CAMERA", command=self.start_camera,
                  **button_style).grid(row=0, column=0, padx=6)
        tk.Button(top_frame, text="STOP CAMERA", command=self.stop_camera,
                  **button_style).grid(row=0, column=1, padx=6)
        tk.Button(top_frame, text="WARM", command=lambda: self.set_lighting(
            "Warm"), **button_style).grid(row=0, column=2, padx=6)
        tk.Button(top_frame, text="COOL", command=lambda: self.set_lighting(
            "Cool"), **button_style).grid(row=0, column=3, padx=6)
        tk.Button(top_frame, text="QUIT", command=self.close_app,
                  **button_style).grid(row=0, column=4, padx=6)

        self.status_label = tk.Label(
            self.root,
            text=f"Lighting Mode: {self.lighting_mode}",
            font=("Arial", 17, "bold"),
            bg=BG_APP,
            fg="black",
        )
        self.status_label.pack(pady=(2, 8))

        separator_line = tk.Frame(self.root, bg="black", height=3)
        separator_line.pack(fill="x", padx=10, pady=(0, 10))

        main_frame = tk.Frame(self.root, bg=BG_APP)
        main_frame.pack(anchor="n", padx=8, pady=(2, 12))

        left_frame = tk.Frame(main_frame, bg=BG_APP)
        left_frame.pack(side="left", padx=(0, 8), anchor="n")

        self.video_label = tk.Label(
            left_frame, bg="black", bd=2, relief="sunken")
        self.video_label.pack(anchor="n")

        right_frame = tk.Frame(main_frame, bg=BG_APP,
                               width=420, height=VIDEO_TOTAL_H)
        right_frame.pack(side="left", anchor="n")
        right_frame.pack_propagate(False)

        box_common = {
            "font": ("Arial", 13, "bold"),
            "bg": BG_APP,
            "padx": 16,
            "pady": 12,
            "bd": 2,
            "relief": "groove",
        }

        eval_frame = tk.LabelFrame(
            right_frame, text="Real-Time Evaluation", height=232, **box_common)
        eval_frame.pack(fill="x", anchor="n", pady=(0, 12))
        eval_frame.pack_propagate(False)

        tk.Label(eval_frame, text="Actual Class:", font=("Arial", 14, "bold"),
                 bg=BG_APP, fg=MUTED_TEXT_COLOR_TK).pack(anchor="w", pady=(0, 8))

        radio_classes = [
            ("Green", "Green"),
            ("Partially ripe", "Partially ripe"),
            ("Ripe", "Ripe"),
            ("Overripe", "Overripe"),
        ]
        for label_text, label_value in radio_classes:
            tk.Radiobutton(
                eval_frame,
                text=label_text,
                variable=self.actual_class,
                value=label_value,
                font=("Arial", 12),
                bg=BG_APP,
                fg="black",
                activebackground=BG_APP,
                anchor="w"
            ).pack(anchor="w", pady=0)

        tk.Button(
            eval_frame,
            text="RECORD RESULT",
            width=18,
            height=2,
            font=("Arial", 12, "bold"),
            bg="#dfead8",
            activebackground="#cfe0c4",
            command=self.record_result
        ).pack(pady=(12, 0))

        summary_frame = tk.LabelFrame(
            right_frame, text="Final Summary", height=VIDEO_TOTAL_H - 205 - 6, **box_common)
        summary_frame.pack(fill="both", anchor="n")
        summary_frame.pack_propagate(False)

        self.recommended_method_label = tk.Label(
            summary_frame,
            text="Recommended Method:\n-",
            font=("Arial", 15, "bold"),
            fg="black",
            bg=BG_APP,
            anchor="w",
            justify="left",
            wraplength=350,
        )
        self.recommended_method_label.pack(anchor="w", pady=(2, 10))

        self.detected_class_label = tk.Label(
            summary_frame,
            text="Detected Class: Unknown",
            font=("Arial", 12),
            fg="black",
            bg=BG_APP,
            anchor="w",
            justify="left",
            wraplength=350,
        )
        self.detected_class_label.pack(anchor="w", pady=0)

        self.best_accuracy_label = tk.Label(
            summary_frame,
            text="Best Method Accuracy: 0.00%",
            font=("Arial", 12),
            fg="black",
            bg=BG_APP,
            anchor="w",
            justify="left",
            wraplength=350,
        )
        self.best_accuracy_label.pack(anchor="w", pady=0)

        self.system_status_label = tk.Label(
            summary_frame,
            text="System Status: Review Needed",
            font=("Arial", 15, "bold"),
            fg="red",
            bg=BG_APP,
            anchor="w",
            justify="left",
            wraplength=350,
        )
        self.system_status_label.pack(anchor="w", pady=(8, 8))

        self.method1_acc_label = tk.Label(summary_frame, text="Method 1 Accuracy: 0.00%", font=(
            "Arial", 12), fg="black", bg=BG_APP, anchor="w", justify="left", wraplength=350)
        self.method1_acc_label.pack(anchor="w", pady=0)
        self.method2_acc_label = tk.Label(summary_frame, text="Method 2 Accuracy: 0.00%", font=(
            "Arial", 12), fg="black", bg=BG_APP, anchor="w", justify="left", wraplength=350)
        self.method2_acc_label.pack(anchor="w", pady=0)
        self.method3_acc_label = tk.Label(summary_frame, text="Method 3 Accuracy: 0.00%", font=(
            "Arial", 12), fg="black", bg=BG_APP, anchor="w", justify="left", wraplength=350)
        self.method3_acc_label.pack(anchor="w", pady=0)

    def set_lighting(self, mode):
        self.lighting_mode = mode
        self.status_label.config(text=f"Lighting Mode: {self.lighting_mode}")

    def start_camera(self):
        if self.running:
            return

        self.cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.running = True
        self.prev_time = time.time()
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def save_snapshot_for_record(self):
        if self.last_canvas is None:
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SAVE_DIR, f"record_{timestamp}.png")
        cv.imwrite(save_path, self.last_canvas)
        return save_path

    @staticmethod
    def label_color_tk(value):
        return GREEN_TEXT_COLOR_TK if value >= ACC_GOOD_THRESHOLD else DEFAULT_TEXT_COLOR_TK

    def update_summary_from_record(self, actual_class):
        m1_label = self.last_info1["current_label"]
        m2_label = self.last_info2["current_label"]
        m3_label = self.last_info3["current_label"]

        m1_conf = self.last_info1["current_conf"] * 100.0
        m2_conf = self.last_info2["current_conf"] * 100.0
        m3_conf = self.last_info3["current_conf"] * 100.0

        self.recorded_method1_acc = m1_conf if m1_label == actual_class else 0.0
        self.recorded_method2_acc = m2_conf if m2_label == actual_class else 0.0
        self.recorded_method3_acc = m3_conf if m3_label == actual_class else 0.0

        method_scores = {
            "Method 1": self.recorded_method1_acc,
            "Method 2": self.recorded_method2_acc,
            "Method 3": self.recorded_method3_acc,
        }

        best_method = max(method_scores, key=method_scores.get)
        best_accuracy = method_scores[best_method]

        if best_accuracy > 0:
            if best_method == "Method 1":
                detected_class = self.last_info1["current_label"]
            elif best_method == "Method 2":
                detected_class = self.last_info2["current_label"]
            else:
                detected_class = self.last_info3["current_label"]
            system_status = "Acceptable"
            status_color = GREEN_TEXT_COLOR_TK
        else:
            detected_class = "Unknown"
            system_status = "Review Needed"
            status_color = "red"
            best_method = "-"
            best_accuracy = 0.0

        recommended_color = self.label_color_tk(best_accuracy)
        self.recommended_method_label.config(
            text=f"Recommended Method:\n{best_method}", fg=recommended_color)
        self.detected_class_label.config(
            text=f"Detected Class: {short_display_name(detected_class)}", fg=DEFAULT_TEXT_COLOR_TK)
        self.best_accuracy_label.config(
            text=f"Best Method Accuracy: {best_accuracy:.2f}%", fg=recommended_color)
        self.system_status_label.config(
            text=f"System Status: {system_status}", fg=status_color)

        self.method1_acc_label.config(
            text=f"Method 1 Accuracy: {self.recorded_method1_acc:.2f}%", fg=self.label_color_tk(self.recorded_method1_acc))
        self.method2_acc_label.config(
            text=f"Method 2 Accuracy: {self.recorded_method2_acc:.2f}%", fg=self.label_color_tk(self.recorded_method2_acc))
        self.method3_acc_label.config(
            text=f"Method 3 Accuracy: {self.recorded_method3_acc:.2f}%", fg=self.label_color_tk(self.recorded_method3_acc))

        return best_method, detected_class, best_accuracy, system_status

    def record_result(self):
        if not (self.last_info1 and self.last_info2 and self.last_info3 and self.last_canvas is not None):
            messagebox.showwarning(
                "Warning", "No prediction available yet. Please start the camera first.")
            return

        actual = self.actual_class.get().strip()
        if not actual:
            messagebox.showwarning(
                "Warning", "Please select an Actual Class first.")
            return

        snapshot_path = self.save_snapshot_for_record()
        best_method, detected_class, best_accuracy, system_status = self.update_summary_from_record(
            actual)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        row = [
            timestamp,
            snapshot_path if snapshot_path else "",
            self.lighting_mode,
            actual,

            self.last_info1["current_label"], f"{self.last_info1['current_conf'] * 100:.2f}%",
            "Yes" if self.last_info1["current_label"] == actual else "No",
            f"{self.recorded_method1_acc:.2f}%",
            self.last_info1["majority_label"], f"{self.last_info1['majority_conf'] * 100:.2f}%",
            f"{self.last_info1['stability']:.2f}%",
            self.last_info1["top2_label"], f"{self.last_info1['gap'] * 100:.2f}%",

            self.last_info2["current_label"], f"{self.last_info2['current_conf'] * 100:.2f}%",
            "Yes" if self.last_info2["current_label"] == actual else "No",
            f"{self.recorded_method2_acc:.2f}%",
            self.last_info2["majority_label"], f"{self.last_info2['majority_conf'] * 100:.2f}%",
            f"{self.last_info2['stability']:.2f}%",
            self.last_info2["top2_label"], f"{self.last_info2['gap'] * 100:.2f}%",

            self.last_info3["current_label"], f"{self.last_info3['current_conf'] * 100:.2f}%",
            "Yes" if self.last_info3["current_label"] == actual else "No",
            f"{self.recorded_method3_acc:.2f}%",
            self.last_info3["majority_label"], f"{self.last_info3['majority_conf'] * 100:.2f}%",
            f"{self.last_info3['stability']:.2f}%",
            self.last_info3["top2_label"], f"{self.last_info3['gap'] * 100:.2f}%",

            best_method,
            detected_class,
            f"{best_accuracy:.2f}%",
            system_status
        ]
        append_csv_log(row)

        messagebox.showinfo(
            "Recorded Successfully",
            f"Actual Class: {short_display_name(actual)}\n"
            f"Snapshot saved automatically.\n"
            f"CSV recorded automatically.\n\n"
            f"Recommended Method: {best_method}\n"
            f"Best Method Accuracy: {best_accuracy:.2f}%"
        )

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read frame from webcam.")
            self.stop_camera()
            return

        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (PROCESS_W, PROCESS_H))

        method1_img = preprocess_method1(frame)
        method2_img = preprocess_method2_live(frame)
        method3_img = preprocess_method3_live(frame)

        pred1 = predict_frame(model1, method1_img)
        pred2 = predict_frame(model2, method2_img)
        pred3 = predict_frame(model3, method3_img)

        update_history(self.m1_label_hist, self.m1_conf_hist,
                       pred1["top1_label"], pred1["top1_conf"])
        update_history(self.m2_label_hist, self.m2_conf_hist,
                       pred2["top1_label"], pred2["top1_conf"])
        update_history(self.m3_label_hist, self.m3_conf_hist,
                       pred3["top1_label"], pred3["top1_conf"])

        m1_majority, m1_majority_conf, m1_stability = compute_majority_and_stability(
            self.m1_label_hist, self.m1_conf_hist)
        m2_majority, m2_majority_conf, m2_stability = compute_majority_and_stability(
            self.m2_label_hist, self.m2_conf_hist)
        m3_majority, m3_majority_conf, m3_stability = compute_majority_and_stability(
            self.m3_label_hist, self.m3_conf_hist)

        info1 = {
            "current_label": pred1["top1_label"],
            "current_conf": pred1["top1_conf"],
            "majority_label": m1_majority,
            "majority_conf": m1_majority_conf,
            "stability": m1_stability,
            "top2_label": pred1["top2_label"],
            "gap": pred1["gap"],
        }
        info2 = {
            "current_label": pred2["top1_label"],
            "current_conf": pred2["top1_conf"],
            "majority_label": m2_majority,
            "majority_conf": m2_majority_conf,
            "stability": m2_stability,
            "top2_label": pred2["top2_label"],
            "gap": pred2["gap"],
        }
        info3 = {
            "current_label": pred3["top1_label"],
            "current_conf": pred3["top1_conf"],
            "majority_label": m3_majority,
            "majority_conf": m3_majority_conf,
            "stability": m3_stability,
            "top2_label": pred3["top2_label"],
            "gap": pred3["gap"],
        }

        self.last_info1 = info1
        self.last_info2 = info2
        self.last_info3 = info3

        selected_actual = self.actual_class.get()
        panel1 = make_panel(method1_img, "Method 1: Raw",
                            info1, selected_actual)
        panel2 = make_panel(
            method2_img, "Method 2: Masked + CLAHE", info2, selected_actual)
        panel3 = make_panel(
            method3_img, "Method 3: Isolated + CLAHE", info3, selected_actual)
        canvas = np.hstack([panel1, panel2, panel3])

        curr_time = time.time()
        fps = 1.0 / max(curr_time - self.prev_time, 1e-6)
        self.prev_time = curr_time

        canvas = add_bottom_bar(canvas, fps, self.lighting_mode)
        self.last_canvas = canvas.copy()

        if DISPLAY_SCALE != 1.0:
            new_w = int(canvas.shape[1] * DISPLAY_SCALE)
            new_h = int(canvas.shape[0] * DISPLAY_SCALE)
            canvas = cv.resize(canvas, (new_w, new_h))

        rgb = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_frame)

    def close_app(self):
        self.stop_camera()
        self.root.destroy()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    init_csv_log()
    root = tk.Tk()
    app = BananaApp(root)
    root.mainloop()
