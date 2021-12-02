"""Background Subtraction demostration application."""
import time
import edgeiq
import cv2
import numpy as np

def preprocess(frame, cuda = False):
    """Blur frame to reduce noise."""
    rows, columns = frame.shape[:2]
    gauss_frame = np.empty((rows, columns, 3), np.uint8)
    gauss_frame_2 = np.empty((rows, columns, 3), np.uint8)
    if not cuda:
        gauss_frame = cv2.GaussianBlur(frame, (5, 5), sigmaX=0, sigmaY=0)
        gauss_frame_2 = cv2.GaussianBlur(gauss_frame, (5, 5), sigmaX=0, sigmaY=0)
    else:
        stream1 = cv2.cuda_Stream()
        frame_device = cv2.cuda_GpuMat(rows, columns, cv2.CV_8UC3)
        frame_device_gauss = cv2.cuda_GpuMat(rows, columns, cv2.CV_8UC3)
        frame_device_gauss2 = cv2.cuda_GpuMat(rows, columns, cv2.CV_8UC3)
        gauss = cv2.cuda.createGaussianFilter(frame_device.type(),
                                              frame_device_gauss.type(),
                                              (5, 5), sigma1=0, sigma2=0)
        frame_device.upload(frame, stream1)
        gauss.apply(frame_device, frame_device_gauss, stream=stream1)
        gauss.apply(frame_device_gauss, frame_device_gauss2, stream=stream1)
        frame_device.download(stream1, gauss_frame_2)
        stream1.waitForCompletion()

    return gauss_frame_2


def postprocess(gray_frame, cuda = False):
    """Sharpen edges in image."""
    rows, columns = gray_frame.shape[:2]
    frame_morph2 = np.empty((rows, columns), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    if not cuda:
        frame_morph = np.empty((rows, columns), np.uint8)
        frame_morph = cv2.dilate(gray_frame, kernel, iterations=1)
        frame_morph2 = cv2.erode(frame_morph, kernel, iterations=2)
    else:
        stream2 = cv2.cuda_Stream()
        frame_gray_device = cv2.cuda_GpuMat(rows, columns, cv2.CV_8UC1)
        frame_device_morph = cv2.cuda_GpuMat(rows, columns, cv2.CV_8UC1)
        frame_device_morph2 = cv2.cuda_GpuMat(rows, columns, cv2.CV_8UC1)
        cuda_dilate = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE,
                                                      frame_device_morph.type(),
                                                      kernel=kernel, iterations=1)
        cuda_erode = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE,
                                                     frame_device_morph2.type(),
                                                     kernel=kernel, iterations=2)
        frame_gray_device.upload(gray_frame, stream2)
        cuda_dilate.apply(frame_gray_device, frame_device_morph, stream=stream2)
        # stream2.waitForCompletion()
        cuda_erode.apply(frame_device_morph, frame_device_morph2,
                         stream=stream2)
        frame_device_morph2.download(stream2, frame_morph2)
        stream2.waitForCompletion()

    return frame_morph2


def main():
    CUDA = False
    COLOR = False
    FILE = "vtest.avi"
    mog2_process = edgeiq.MOG2(history=120, var_threshold=250, detect_shadows=True, cuda=CUDA)
    fps = edgeiq.FPS()
    if CUDA:
        text = ["Using CUDA backend"]
    else:
        text = ["Using CPU"]

    try:
        with edgeiq.FileVideoStream(FILE, play_realtime=True) as video_stream,\
                edgeiq.Streamer() as streamer:
            # Input video stream speed in frame per second
            print(video_stream._thread._fps)
            fps.start()

            # loop background subtractor
            while video_stream.more():
                frame = video_stream.read()
                blurred_frame = preprocess(frame, cuda=CUDA)
                mog_frame = mog2_process.process_frame(blurred_frame,
                                                       learning_rate=-1)
                post_frame = postprocess(mog_frame, cuda=CUDA)
                if COLOR:
                    raw_contours = cv2.findContours(post_frame,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                    contours = edgeiq.get_contours(raw_contours=raw_contours)
                    boundingboxes = edgeiq.get_boundingboxes(contours=contours)
                    for contour in contours:
                        for bb in boundingboxes:
                            (x, y, w, h) = bb
                            aspect_ratio = w/h
                            if aspect_ratio < 1:
                                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2,)
                                M = cv2.moments(contour)
                                if M["m00"] > 0:
                                     cX = int(M["m10"] / M["m00"])
                                     cY = int(M["m01"] / M["m00"])
                                     cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)


                    streamer.send_data(frame, text)

                else:
                    streamer.send_data(post_frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
