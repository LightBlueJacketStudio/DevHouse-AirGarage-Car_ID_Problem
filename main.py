import cv2

#img = cv2.imread("sample.jpg")
#Load and Display an Image
# cv2.imshow("My Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Convert to Grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", gray)
# cv2.waitKey(0)

#Resize an Image
# resized = cv2.resize(img, (300, 300))
# cv2.imshow("Resized", resized)
# cv2.waitKey(0)

#draw shapes
# cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
# cv2.circle(img, (150, 150), 50, (255, 0, 0), 3)
#idk what this does

# from ultralytics import YOLO
# import cv2

# model = YOLO("yolov8n.pt")  # nano model (fast, auto-downloads)

# img = cv2.imread("downloads/0b7d4804-0a32-4466-a359-a9f6a67877f4_1739999185_4649_.jpeg")
# results = model(img)

# results[0].show()

# from ultralytics import YOLO
# import cv2
# import os



# # ---------- 1. DEFINE expand_box FIRST ----------
# def expand_box(x1, y1, x2, y2, img_w, img_h, margin=0.10):
#     bw = x2 - x1
#     bh = y2 - y1

#     dx = int(bw * margin)
#     dy = int(bh * margin)

#     nx1 = max(0, x1 - dx)
#     ny1 = max(0, y1 - dy)
#     nx2 = min(img_w - 1, x2 + dx)
#     ny2 = min(img_h - 1, y2 + dy)

#     return nx1, ny1, nx2, ny2

# # COCO class ids for vehicles
# VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# def crop_with_yolo(
#     image_path,
#     out_dir="crops",
#     model_name="yolov8s.pt",
#     conf_thres=0.3,
#     margin=0.10,
#     save_all=True
# ):
#     os.makedirs(out_dir, exist_ok=True)

#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"Could not read image: {image_path}")

#     h, w = img.shape[:2]

#     model = YOLO(model_name)  # downloads model automatically if missing
#     results = model(img, conf=conf_thres)

#     boxes = results[0].boxes
#     saved = 0

#     for i, box in enumerate(boxes):
#         cls = int(box.cls[0])
#         conf = float(box.conf[0])

#         if cls not in VEHICLE_CLASSES:
#             continue

#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, margin=margin)

#         crop = img[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         base = os.path.splitext(os.path.basename(image_path))[0]
#         out_path = os.path.join(out_dir, f"{base}_det{i}_c{cls}_conf{conf:.2f}_m{margin:.2f}.jpg")

#         cv2.imwrite(out_path, crop)
#         saved += 1

#         if not save_all:
#             break  # only save the first detection

#     return saved

# n = crop_with_yolo("downloads/0b7d4804-0a32-4466-a359-a9f6a67877f4_1739999185_4649_.jpeg", out_dir="vehicle_crops", margin=0.12, conf_thres=0.4)
# print("Saved crops:", n)



from ultralytics import YOLO
import cv2
import numpy as np

testImgLocation = "downloads/tripwire_03df51fe-6194-47ac-ba86-ae64db78cbaa.jpeg"
testImg = cv2.imread(testImgLocation)
if testImg is None:
    raise ValueError("Image not found or path is wrong")



model = YOLO("yolov8n.pt")  # nano model (fast, auto-downloads)


def normalize_lighting_keep_color(img_bgr, clip=2.0, grid=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    L2 = clahe.apply(L)

    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


import cv2, numpy as np

def match_mean_std(img_bgr, target_mean=128.0, target_std=50.0):
    x = img_bgr.astype(np.float32)
    mean = x.mean()
    std = x.std() + 1e-6
    y = (x - mean) * (target_std / std) + target_mean
    y = np.clip(y, 0, 255).astype(np.uint8)
    return y

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
print("testImg type:", type(testImg))
print("testImg shape:", getattr(testImg, "shape", None))
print("testImg dtype:", getattr(testImg, "dtype", None))
print("testImg min/max:", (testImg.min(), testImg.max()) if isinstance(testImg, np.ndarray) else None)

img_norm = match_mean_std(testImg)

print("img_norm shape:", img_norm.shape)
print("img_norm dtype:", img_norm.dtype)
print("img_norm min/max:", img_norm.min(), img_norm.max())


h, w = img_norm.shape[:2]
scale = min(1200 / w, 800 / h, 1.0)  # fit within ~1200x800
img_view = cv2.resize(img_norm, (int(w*scale), int(h*scale)))

#img_norm = normalize_lighting_keep_color(testImg)


cv2.imshow("sent to model", img_view)
cv2.waitKey(0)
cv2.destroyAllWindows()

results = model(img_view)

annotated_img = results[0].plot()   # draw boxes on image

cv2.imshow("My Image", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

