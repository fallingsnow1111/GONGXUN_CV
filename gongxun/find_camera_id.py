import cv2 as cv

def test_cameras():
    for i in range(5):
        cap = cv.VideoCapture(i, cv.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv.imshow(f"Camera Index {i}", frame)
                print(f"Index {i} is working.")
            cap.release()
    cv.waitKey(0)
    cv.destroyAllWindows()

test_cameras()