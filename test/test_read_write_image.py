#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import filedialog
import cv2 as cv

if __name__ == '__main__':
    path_image = filedialog.askopenfilename(
        title='select an image file to open',
        initialdir='./calibration_images',
        filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp;*.tiff'), ('All files', '*.*')],
    )
    print(f'Opening file {path_image!r} ...')
    img = cv.imread(path_image)

    cv.imshow('img', img)
    while True:
        key_event = 0xff & cv.waitKey(0)
        if key_event == ord('q'):
            cv.destroyAllWindows()
            exit()
        elif key_event == ord('s'):
            cv.imwrite('./output.png', img)
        else:
            print(f'Unexpected key: {key_event}')
