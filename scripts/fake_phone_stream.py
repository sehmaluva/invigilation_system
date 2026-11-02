"""
Pretend to be a phone camera: serves a video file (looped) as an MJPEG stream,
like the IP Webcam app does. Handy for testing without a phone.

  python scripts/fake_phone_stream.py data/synthetic.mp4 --port 8081
  -> then start a live session with URL  http://127.0.0.1:8081/video
"""
import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--fps", type=float, default=10)
a = ap.parse_args()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith("/video"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        cap = cv2.VideoCapture(a.video)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:                                   # loop the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                jpg = cv2.imencode(".jpg", frame)[1].tobytes()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1 / a.fps)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            cap.release()

    def log_message(self, *args):
        pass


print(f"Fake phone camera on http://127.0.0.1:{a.port}/video")
ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
