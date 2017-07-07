import cv2.aruco
import cv2
import numpy as np

# Dots per cm (600 DPI)
dpcm = 236.2205

board_width = 9
board_height = 7

board_width_cm = 5
board_height_cm = board_width_cm * (board_height / board_width)

_board_image_size = (round(board_width_cm * dpcm), round(board_height_cm * dpcm))

padding = 0.15
full_size = board_width_cm / board_width
marker_size = full_size - padding

_dictionary = None
_board = None


def get_dictionary():
    global _dictionary

    if _dictionary is None:
        _dictionary = cv2.aruco.custom_dictionary(28, 4)

    return _dictionary


def get_board():
    global _board
    dictionary = get_dictionary()

    if _board is None:
        board_points = list(range(28))

        i = 0
        for y in (0, 6):
            y *= full_size
            for x in range(1, board_width - 1):
                x *= full_size
                board_points[i] = np.array([(x, y, 0), (x, y + marker_size, 0), (x + marker_size, y + marker_size, 0),
                                            (x + marker_size, y, 0)], np.float32)
                i += 1

        for x in (0, 8):
            x *= full_size
            for y in range(board_height):
                y *= full_size
                board_points[i] = np.array([(x, y, 0), (x, y + marker_size, 0), (x + marker_size, y + marker_size, 0),
                                            (x + marker_size, y, 0)], np.float32)
                i += 1

        _board = cv2.aruco.Board_create(board_points, dictionary, tuple(range(28)))
    return _board


def main():
    print("Marker Size: {0:.4f} cm".format(marker_size))

    board = get_board()
    board_image = cv2.aruco.drawPlanarBoard(board=board, outSize=_board_image_size, marginSize=0, borderBits=1)

    cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Board", 600, int(600 * (board_height_cm / board_width_cm)))
    cv2.imshow("Board", board_image)
    cv2.waitKey()
    cv2.imwrite('aruco_board.png', board_image)


if __name__ == '__main__':
    main()
