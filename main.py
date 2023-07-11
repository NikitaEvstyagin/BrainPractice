
mediapipe = None
yolo = None

def setup_mediapipe():
    import MediaPipe
    global mediapipe
    mediapipe = MediaPipe
    return  mediapipe
def setup_yolo():
    import YOLOv5
    global yolo
    yolo = YOLOv5
    return yolo

def has_hands(frame, thresh, model):
    global mediapipe
    global yolo
    if model == "mediapipe":
        if mediapipe == None:
            mediapipe = setup_mediapipe()
            if mediapipe.mediapipe_for_images(frame):
                return True
            return False
        elif mediapipe.mediapipe_for_images(frame):
            return True
        return False
    elif model == "handobj":
        if yolo == None:
            yolo = setup_yolo()
            res = yolo.yolo_for_image(frame)
            for _, obj in res.pandas().xyxy[0].iterrows():
                if obj["name"] == "hand" and obj["confidence"] > thresh:
                    return True
            return False
        else:
            res = yolo.yolo_for_image(frame)
            for _, obj in res.pandas().xyxy[0].iterrows():
                if obj["name"] == "hand" and obj["confidence"] > thresh:
                    return True
            return False





IMAGE_FILES = ['images/1 (1).jpg', "images/1 (2).jpg"]

for frame in IMAGE_FILES:
    print(has_hands(frame, 0.8, "mediapipe"))
