#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import cv2 as cv
import time
import os

if __name__ == '__main__':
    os.mkdir('calibration_images', exist_ok=True)

    cv.namedWindow('video', cv.WINDOW_NORMAL)
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv.imshow('video', frame)

        key_event = 0xff & cv.waitKey(1)
        if key_event == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            exit()
        elif key_event == ord('s'):
            now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            path_output = f'./calibration_images/{now}.jpg'
            cv.imwrite(
                path_output,
                frame,
                [cv.IMWRITE_JPEG_QUALITY, 60],
            )

        time.sleep(.01)
