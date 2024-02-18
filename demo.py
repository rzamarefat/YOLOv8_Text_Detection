from glob import glob
import cv2
from YOLOv8_Text_Detection import Yolov8TextDetection


if __name__ == "__main__":
    images = [cv2.imread(p) for p in sorted(glob(r"C:\Users\ASUS\Desktop\github_projects\YOLOv8_Text_Detection\images\*.jpg"))]

    detector = Yolov8TextDetection(device="cuda")

    polylines = detector.detect(images)

    print(polylines)

    drawn_images = detector.visualize_polylines(images, polylines)

    for d in drawn_images:
        cv2.imshow("Image", d)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

