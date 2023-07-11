import cv2
import torch

# Load the YOLOv5 model

model = torch.hub.load("ultralytics/yolov5", 'custom', path='weight/yolov5.pt')
# model.conf = 0.5
# image_files = ['images/1 (4).jpg']
# images
def yolo_for_image(file):
    # img = cv2.imread(file)
    # result = model(file)
    return model(file)
    # result.render()
    # print(dir(result))
    # print(result.pandas().xyxy[0])
    # result.show()
    # cv2.imshow("res", img)
    # cv2.waitKey(0)

# yolo_for_image(image_files)



# Open the video file
#video_path = 'fileobjects/videos/manwithhand.mp4'

def yolo_for_videos(video):
    cap = cv2.VideoCapture(video)
    cv2.namedWindow("YOLOv5 Inference", cv2.WINDOW_NORMAL)
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv5
            results = model(frame)

            # Visualize the results on the frame
            results.render()

            # Display the annotated frame
            cv2.imshow("YOLOv5 Inference", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            # Break the loop if the end ov the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()