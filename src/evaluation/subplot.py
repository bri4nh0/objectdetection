import matplotlib.pyplot as plt
import cv2
from matplotlib.gridspec import GridSpec
# Load your images (replace with your actual file paths)
img_n = cv2.cvtColor(cv2.imread("runs/detect/yolov8n_100epochs/confusion_matrix.png"), cv2.COLOR_BGR2RGB)
img_s = cv2.cvtColor(cv2.imread("runs/detect/yolov8s_100epochs/confusion_matrix.png"), cv2.COLOR_BGR2RGB)
img_m = cv2.cvtColor(cv2.imread("runs/detect/yolov8m_100epochs/confusion_matrix.png"), cv2.COLOR_BGR2RGB)

# Create a figure with GridSpec (2 rows, 2 columns)
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig)

# Top row: two subplots
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img_n)
ax1.set_title("(a) YOLOv8n")
ax1.axis("off")

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(img_s)
ax2.set_title("(b) YOLOv8s")
ax2.axis("off")

# Bottom row: span both columns (centered)
ax3 = fig.add_subplot(gs[1, :])
ax3.imshow(img_m)
ax3.set_title("(c) YOLOv8m")
ax3.axis("off")



plt.tight_layout()
plt.show()