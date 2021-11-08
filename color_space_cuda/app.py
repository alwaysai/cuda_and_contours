import cv2
import edgeiq
import time
import numpy as np

class get_color_spaces(object):
    def __init__(self, cuda=False):
        self.cuda = cuda
        self._initialized = False


    def _initialize_color_space(self, bgr_frame):
        print("Allocating Memory")
        self.rows, self.columns = bgr_frame.shape[:2]
        self.bgr_frame = np.empty((self.rows, self.columns, 3), np.uint8)
        self.hsv_frame = np.empty((self.rows, self.columns, 3), np.uint8)
        self.lab_frame = np.empty((self.rows, self.columns, 3), np.uint8)
        self.YCrCb_frame = np.empty((self.rows, self.columns, 3), np.uint8)
        if self.cuda:
            self.stream = cv2.cuda_Stream()
            self.cuda_bgr_frame = cv2.cuda_GpuMat(self.rows, self.columns,
                                                  cv2.CV_8UC3)
            self.cuda_hsv_frame = cv2.cuda_GpuMat(self.rows, self.columns,
                                                  cv2.CV_8UC3)
            self.cuda_lab_frame = cv2.cuda_GpuMat(self.rows, self.columns,
                                                  cv2.CV_8UC3)
            self.cuda_YCrCb_frame = cv2.cuda_GpuMat(self.rows, self.columns,
                                                    cv2.CV_8UC3)
        print("Finished Allocating Memory")


    def do_color_spaceing(self,frame):
        if not self._initialized:
            self._initialize_color_space(frame)
            self._initialized = True
        self.bgr_frame = frame
        if self.cuda:
            self.cuda_bgr_frame.upload(self.bgr_frame, self.stream)
            self.cuda_hsv_frame = cv2.cuda.cvtColor(self.cuda_bgr_frame,
                                                    cv2.COLOR_BGR2HSV,
                                                    stream=self.stream)
            self.cuda_lab_frame = cv2.cuda.cvtColor(self.cuda_bgr_frame,
                                                    cv2.COLOR_BGR2Lab,
                                                    stream=self.stream)
            self.cuda_YCrCb_frame = cv2.cuda.cvtColor(self.cuda_bgr_frame,
                                                      cv2.COLOR_BGR2YCrCb,
                                                      stream=self.stream)
            self.cuda_hsv_frame.download(self.stream, self.hsv_frame)
            self.cuda_lab_frame.download(self.stream, self.lab_frame)
            self.cuda_YCrCb_frame.download(self.stream, self.YCrCb_frame)
            self.stream.waitForCompletion()
        else:
            self.hsv_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_BGR2HSV)
            self.lab_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_BGR2Lab)
            self.YCrCb_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_BGR2YCrCb)

        return (self.hsv_frame, self.lab_frame, self.YCrCb_frame)

def main():
    """Run color space application"""
    text = "Color Space"
    FILE = "race.mp4"
    CUDA = False
    fps = edgeiq.FPS()
    try:
        with edgeiq.FileVideoStream(FILE, play_realtime=True) as video_stream,\
                edgeiq.Streamer() as streamer:
            # check video stream input fps
            print(video_stream._thread._fps)
            time.sleep(2.0)
            color_space = get_color_spaces(cuda=CUDA)
            fps.start()
            while video_stream.more():
                frame = video_stream.read()
                hsv, lab, YCrCb = color_space.do_color_spaceing(frame)
                combined = np.hstack((frame, hsv))
                combined_2 = np.hstack((lab, YCrCb))
                combined_3 = np.vstack((combined, combined_2))
                streamer.send_data(combined_3)
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
