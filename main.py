from gooey import Gooey
from gooey import GooeyParser

from classes.centroidtracker import CentroidTracker
from classes.peopletracker import PeopleTracker
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2


@Gooey
def main():
    ap = GooeyParser(description="People Counter By Danish Siddique!")
    ap.add_argument("-p", "--prototxt",type=str, default='mobilenet_ssd/MobileNetSSD_deploy.prototxt ')
    ap.add_argument("-m", "--model", default='mobilenet_ssd/MobileNetSSD_deploy.caffemodel ')
    ap.add_argument('input', widget="FileChooser")
    ap.add_argument("-o", "--output", type=str, default='output/output.avi')
    ap.add_argument("-c", "--confidence", type=float, default=0.4)
    ap.add_argument("-s", "--skip-frames", type=int, default=30)

    args = vars(ap.parse_args())
    # initialize the list of class labels MobileNet SSD was trained to detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    print("Loading pre trained model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # if no input video file is given then start Video camera if any else open the video file
    if not args.get("input", False):
        print("Starting Video Stream")
        video_stream = VideoStream(src=0).start()
        time.sleep(2.0)
    else:
        print("Opening Video File")
        video_stream = cv2.VideoCapture(args["input"])

    # Initialing variables for writing and tracking of input video
    writer = None
    W = None
    H = None
    centroid_tracker = CentroidTracker(maxDisappeared=50, maxDistance=50)
    trackers = []
    peopleTracker = {}

    totalFrames = 0

    peopleOut = 0
    peopleIn = 0

    fps = FPS().start()

    while True:

        frame = video_stream.read()
        frame = frame[1] if args.get("input", False) else frame

        if args["input"] is not None and frame is None:
            break

        # Resize the video file to 500 width
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Creating the output file
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        # initialize status as waiting
        status = "Waiting"
        rects = []
        # detect if the object in frame is a person
        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)
        # If not detecting then track the movement of that person in each frame
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        obj = centroid_tracker.update(rects)

        for (objectID, centroid) in obj.items():
            to = peopleTracker.get(objectID, None)

            if to is None:
                to = PeopleTracker(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        peopleOut += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > H // 2:
                        peopleIn += 1
                        to.counted = True

            peopleTracker[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        info = [
            ("People Out", peopleOut),
            ("People In", peopleIn),
            ("Status", status),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 255), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Frame", frame)
        # Press q to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()
    fps.stop()
    print("Time Elapsed: {:.2f}".format(fps.elapsed()))
    print("Approximate FPS: {:.2f}".format(fps.fps()))
    # Closing All Opened Windows
    if writer is not None:
        writer.release()

    if not args.get("input", False):
        video_stream.stop()
    else:
        video_stream.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
