import cv2
import cv2.aruco
import numpy as np

import calibration.endoscope as endoscope
import calibration.aruco_board as aruco_board

video_file = 'data/calibration/rid_5_cm_0.mpeg'

null_dist_coeffs = np.zeros(4)


def main():
    video = cv2.VideoCapture(video_file)
    video.set(cv2.CAP_PROP_POS_MSEC, 10000)

    dictionary = aruco_board.get_dictionary()
    board = aruco_board.get_board()

    while True:
        ret, image = video.read()
        if not ret:
            break

        image = endoscope.preprocess_endoscope_image(image)

        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, endoscope.camera_matrix(),
                                                         null_dist_coeffs)

        if retval > 0:
            # print(tvec)
            cv2.aruco.drawAxis(image, endoscope.camera_matrix(), null_dist_coeffs, rvec, tvec, 1)

        cv2.imshow("Undistort", image)
        if cv2.waitKey(20) == 27:
            break


if __name__ == '__main__':
    main()
