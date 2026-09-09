import os
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO


class FaceRecognizer:
    def __init__(self, model_path="models/face_model.pt"):
        self.model_path = self._resolve_model_path(model_path)
        self.yolo_model = self.load_yolo_model()

    def _resolve_model_path(self, path):
        if os.path.isabs(path) and os.path.exists(path):
            return path
        if os.path.exists(path):
            return path
        file_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path1 = os.path.abspath(os.path.join(file_dir, "..", path))
        if os.path.exists(alt_path1):
            return alt_path1
        alt_path2 = os.path.abspath(os.path.join(file_dir, "..", "models", os.path.basename(path)))
        if os.path.exists(alt_path2):
            return alt_path2
        return path

    def load_yolo_model(self):
        try:
            return YOLO(self.model_path)
        except Exception as e:
            print(f"[YOLO Error] Failed to load {self.model_path}: {e}")
            try:
                print("[YOLO Info] Attempting fallback to yolov8n.pt")
                return YOLO("yolov8n.pt")
            except Exception as ex:
                print(f"[YOLO Error] Fallback failed: {ex}")
                return None

    def detect_and_recognize(self, image_bgr, known_users, conf_threshold=0.50, tolerance=0.45):
        """
        Detect faces using YOLO and match identities against known_users database safely.
        known_users: list of dicts [{"user_code": ..., "name": ..., "department": ..., "embedding": [float...]}, ...]
        """
        if image_bgr is None or self.yolo_model is None:
            return image_bgr, [], 0

        annotated_img = image_bgr.copy()
        h, w, _ = image_bgr.shape
        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run YOLO face detection
        try:
            results = self.yolo_model(image_bgr, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
        except Exception as e:
            print(f"[YOLO Inference Error] {e}")
            return image_bgr, [], 0

        recognized_faces = []
        face_count = 0

        if boxes is not None and len(boxes) > 0:
            face_count = len(boxes)
            for box in boxes:
                try:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    box_conf = float(box.conf[0].cpu().numpy())

                    # Strict boundary clamping
                    left = max(0, min(x1, w - 1))
                    top = max(0, min(y1, h - 1))
                    right = max(0, min(x2, w))
                    bottom = max(0, min(y2, h))

                    if (right - left) < 10 or (bottom - top) < 10:
                        continue

                    # face_recognition location format: (top, right, bottom, left)
                    face_location = [(top, right, bottom, left)]

                    # Extract 128-d face encoding
                    encodings = face_recognition.face_encodings(rgb_img, known_face_locations=face_location)

                    if len(encodings) == 0:
                        crop_roi = rgb_img[top:bottom, left:right]
                        if crop_roi.size > 0:
                            encodings = face_recognition.face_encodings(crop_roi)

                    match_found = False
                    best_match = None

                    # Filter valid known users with 128-d embeddings
                    valid_known = [u for u in known_users if isinstance(u.get("embedding"), (list, np.ndarray)) and len(u["embedding"]) == 128]

                    if len(encodings) > 0 and len(valid_known) > 0:
                        curr_emb = np.array(encodings[0], dtype=np.float64)
                        known_embs = [np.array(u["embedding"], dtype=np.float64) for u in valid_known]

                        distances = face_recognition.face_distance(known_embs, curr_emb)
                        if len(distances) > 0:
                            min_dist_idx = int(np.argmin(distances))
                            min_dist = float(distances[min_dist_idx])

                            if min_dist <= tolerance:
                                match_found = True
                                matched_user = valid_known[min_dist_idx]
                                match_conf = max(0.0, min(100.0, (1.0 - min_dist) * 100.0))
                                best_match = {
                                    "user_code": matched_user["user_code"],
                                    "name": matched_user["name"],
                                    "department": matched_user.get("department", ""),
                                    "distance": min_dist,
                                    "confidence": match_conf / 100.0,
                                    "yolo_conf": box_conf,
                                    "box": (left, top, right, bottom),
                                    "embedding": curr_emb.tolist()
                                }

                    if match_found and best_match:
                        recognized_faces.append(best_match)
                        # Green Box
                        cv2.rectangle(annotated_img, (left, top), (right, bottom), (0, 255, 0), 2)
                        label_text = f"{best_match['name']} ({best_match['user_code']})"
                        sub_text = f"{best_match['confidence']*100:.1f}% Match"

                        cv2.rectangle(annotated_img, (left, max(0, top - 40)), (right, top), (0, 255, 0), cv2.FILLED)
                        cv2.putText(annotated_img, label_text, (left + 5, top - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(annotated_img, sub_text, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                    else:
                        unrecognized_info = {
                            "user_code": "UNKNOWN",
                            "name": "Unknown",
                            "department": "N/A",
                            "distance": 1.0,
                            "confidence": box_conf,
                            "yolo_conf": box_conf,
                            "box": (left, top, right, bottom),
                            "embedding": encodings[0].tolist() if len(encodings) > 0 else []
                        }
                        recognized_faces.append(unrecognized_info)

                        # Red Box
                        cv2.rectangle(annotated_img, (left, top), (right, bottom), (0, 0, 255), 2)
                        label_text = "Unknown Face"
                        sub_text = f"Det: {box_conf*100:.1f}%"
                        cv2.rectangle(annotated_img, (left, max(0, top - 40)), (right, top), (0, 0, 255), cv2.FILLED)
                        cv2.putText(annotated_img, label_text, (left + 5, top - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(annotated_img, sub_text, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                except Exception as ex:
                    print(f"[Face Processing Error] {ex}")

        return annotated_img, recognized_faces, face_count

    def extract_single_face_embedding(self, image_bgr, conf_threshold=0.50):
        """
        Extract face embedding and cropped face ROI for registration.
        Returns: (embedding_numpy, face_crop_bgr, error_msg)
        """
        if image_bgr is None or self.yolo_model is None:
            return None, None, "Invalid image or YOLO model not loaded."

        h, w, _ = image_bgr.shape
        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        try:
            results = self.yolo_model(image_bgr, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
        except Exception as e:
            return None, None, f"YOLO detection error: {e}"

        if boxes is None or len(boxes) == 0:
            return None, None, "No face detected by YOLO. Please ensure your face is clearly visible."

        if len(boxes) > 1:
            return None, None, f"Multiple faces ({len(boxes)}) detected! Please provide an image with exactly one person."

        box = boxes[0]
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)

        left = max(0, min(x1, w - 1))
        top = max(0, min(y1, h - 1))
        right = max(0, min(x2, w))
        bottom = max(0, min(y2, h))

        pad = int(min(right - left, bottom - top) * 0.15)
        crop_left, crop_top = max(0, left - pad), max(0, top - pad)
        crop_right, crop_bottom = min(w, right + pad), min(h, bottom + pad)

        face_crop = image_bgr[crop_top:crop_bottom, crop_left:crop_right]

        face_location = [(top, right, bottom, left)]
        encodings = face_recognition.face_encodings(rgb_img, known_face_locations=face_location)

        if len(encodings) == 0:
            if face_crop.size > 0:
                encodings = face_recognition.face_encodings(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        if len(encodings) == 0:
            return None, None, "Could not extract facial features. Please try another photo with clear lighting."

        return encodings[0], face_crop, None
