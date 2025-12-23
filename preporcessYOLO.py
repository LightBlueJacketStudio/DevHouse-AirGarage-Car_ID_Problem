# Standardize & Normalize
# Feed into YOLO
# Crop out the vihecle
# Feed into fast-reid
# ??
# 
# 

from ultralytics import YOLO
import cv2
import numpy as np
import os
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

def show_img_fit(img_bgr, win = "imageView"):
    #resizing to fit the display
    h, w = img_bgr.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)  # fit within ~1200x800
    img_view = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
    cv2.imshow(win, img_view)

def normalize_lighting_keep_color(img_bgr, clip=1.2, grid=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    L2 = clahe.apply(L)

    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

def match_mean_std(img_bgr, target_mean=128.0, target_std=50.0):
    x = img_bgr.astype(np.float32)
    mean = x.mean()
    std = x.std() + 1e-6
    y = (x - mean) * (target_std / std) + target_mean
    y = np.clip(y, 0, 255).astype(np.uint8)
    return y

def get_l_stats(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    return L.mean(), L.std()

def needs_mean_std(img_bgr,
                   mean_range=(110, 145),
                   std_min=28):   # tighter rule
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    mean = L.mean()
    std = L.std()

    if std < std_min:
        return True
    if mean < mean_range[0] or mean > mean_range[1]:
        return True
    return False

def preprocess(img_bgr):
    #L-clache the image first
    img_normd = normalize_lighting_keep_color(img_bgr)
    #test the global brightness, if abnormal, apply mean_std
    mean_need = needs_mean_std(img_normd)
    if(mean_need):
        img_normd = match_mean_std(img_normd)
    
    return img_normd #this image should be normalized and ready for cropping

def yolo_ID(img_bgr, yolo_model):
    results = yolo_model(img_bgr, conf=0.25, verbose=False)
    r = results[0]
    dets = []
    annotated = img_bgr.copy()
    
    if r.boxes is None:
        #show_img_fit(annotated, "yolo_ID")
        return dets

    for box in r.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in VEHICLE_CLASSES:
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        
        # # draw box
        # cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # label = f"{yolo_model.names[cls_id]} {conf:.2f}"
        # font_scale = 1.0 
        # thickness = 2

        # cv2.putText(
        #     annotated,
        #     label,
        #     (x1, max(y1 - 10, 20)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     font_scale,
        #     (0, 255, 0),
        #     thickness,
        #     cv2.LINE_AA
        # )
         # above is for displaying purpose

        dets.append((x1, y1, x2, y2, cls_id, conf))

    #show_img_fit(annotated, "yolo_ID")
    return dets

def pick_best_det(dets):
    if not dets:
        return None
    def score(d):
        x1,y1,x2,y2,cls_id,conf = d
        area = (x2-x1) * (y2-y1)
        return area * (conf * 3)
    return max(dets, key=score)

def pick_best_det_edge_penalty(dets, img_shape,
                               edge_weight=0.6,
                               edge_margin_frac=0.06):
    """
    Picks best detection by (area * conf) with a penalty if the box touches/approaches image edges.

    edge_margin_frac: how thick the 'edge zone' is (fraction of image size).
                      Boxes whose borders fall inside this zone get penalized.
    edge_weight: how strong the penalty is (0 = no penalty, 1 = strong penalty).
    """
    if not dets:
        return None

    H, W = img_shape[:2]
    mx = edge_margin_frac * W
    my = edge_margin_frac * H

    def score(d):
        x1, y1, x2, y2, cls_id, conf = d
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh

        # Base preference: big + confident
        base = area * conf

        # How deep into the edge-zone the box borders are (0 = not in edge zone, 1 = fully in it)
        left   = max(0.0, (mx - x1) / mx)
        top    = max(0.0, (my - y1) / my)
        right  = max(0.0, (x2 - (W - mx)) / mx)
        bottom = max(0.0, (y2 - (H - my)) / my)

        # Penalty factor in [1-edge_weight, 1]
        edge_pen = max(left, top, right, bottom)  # worst edge contact
        penalty_factor = 1.0 - edge_weight * edge_pen

        return base * penalty_factor

    return max(dets, key=score)


def filter_small_dets(dets, img_shape, min_area_frac=0.02):
    h, w = img_shape[:2]
    min_area = min_area_frac * (w * h)

    keep = []
    for d in dets:
        x1,y1,x2,y2,cls_id,conf = d
        area = (x2-x1) * (y2-y1)
        if area >= min_area:
            keep.append(d)
    return keep

def crop_from_dets(img_bgr, dets, margin=0.10):
    h, w = img_bgr.shape[:2]
    crops = []

    for (x1, y1, x2, y2, cls_id, conf) in dets:
        # expand (optional)
        bw, bh = (x2 - x1), (y2 - y1)
        dx, dy = int(bw * margin), int(bh * margin)
        x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx); y2 = min(h, y2 + dy)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)

    return crops

def write_crop_to_file(cropped_img_bgr, image_path, out_dir="crops"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base}_cropped.jpg")

    ok = cv2.imwrite(out_path, cropped_img_bgr)
    if not ok:
        raise IOError(f"Failed to write image: {out_path}")

    return out_path
    
def pipeline_single_file(image_path, out_path = "crops"):
    print(f"Processing: {image_path}")
    
    testImg = cv2.imread(image_path)
    if testImg is None:
        raise ValueError("Image not found or path is wrong")
    #show_img_fit(testImg, "original")

    out = preprocess(testImg)
    #show_img_fit(out, "pre-processed")
    
    dets = yolo_ID(out, model) # this returns a list of detected vehicles
    dets = filter_small_dets(dets, out.shape)
    best = pick_best_det_edge_penalty(dets, out.shape, edge_weight=0.6, edge_margin_frac=0.06)
    #show_img_fit(best.plot(), "picked")

    if best is None:
        print("No valid detections -> fallback")
        to_save = out
    else:
        cropped = crop_from_dets(out, [best])
        if cropped:
            # normal case: save first vehicle crop
            to_save = cropped[0]
            #show_img_fit(to_save, "cropped")
            print("Saved vehicle crop")
        else:
            # fallback: save original image
            to_save = out
            print("No vehicle detected → saved original")
    
    write_crop_to_file(to_save, image_path)
    
if __name__ == "__main__":
    VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    model = YOLO("yolov8s.pt") #global yolo modle settings

    # extremeDark = "downloads/tripwire_03df51fe-6194-47ac-ba86-ae64db78cbaa.jpeg"
    # extremeBright = "downloads/tripwire_6bbf88d0-c59f-484b-bc59-16f1f6f1062e.jpeg"
    # testImg = cv2.imread(extremeDark)
    # testImg = cv2.imread(extremeBright)


    img_dir = "downloads"
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for fname in sorted(os.listdir(img_dir)):

        if not fname.lower().endswith(exts):
            continue
        
        image_path = os.path.join(img_dir, fname)
        
        pipeline_single_file(image_path)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    """below is for individual redo"""
    #"C:\Users\grape\OneDrive - sjsu.edu\DevHouse-AirGarage-Car_ID_Problem\downloads\tripwire_0db2b19b-bae1-4468-be86-145132fb7ebf.jpeg"
    # specific_redo = "downloads/tripwire_03df51fe-6194-47ac-ba86-ae64db78cbaa.jpeg"
    # pipeline_single_file(specific_redo)
    
    # results = model.predict(cv2.imread(specific_redo), imgsz=1280, conf=0.01, iou=0.7, verbose=False)
    # boxes = results[0].boxes
    # print("num boxes:", 0 if boxes is None else len(boxes))
    # show_img_fit(results[0].plot(), "All boxes")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pass