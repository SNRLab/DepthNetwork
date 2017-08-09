#!/usr/bin/env python3

import OpenEXR
import numpy as np
import argparse
import Imath


def depth_map_to_point_cloud(depth, focal_length, principal_point=None):
    height, width = depth.shape[:2]

    if type(focal_length) is int:
        focal_length = (focal_length, focal_length)

    fx_d, fy_d = focal_length

    if principal_point is None:
        principal_point = (width / 2, height / 2)

    px_d, py_d = principal_point

    # Create point cloud
    pcloud = np.zeros((height * width, 3))

    u = round(height / 2)
    v = round(width / 2)
    vflipped = width - (v + 1)
    z = depth[u, vflipped]
    center_x = z * (height / 2 - px_d) / fx_d
    center_y = z * (width / 2 - py_d) / fy_d

    for u in range(height):
        for v in range(width):
            # z = (1.0/depth(u,v) * dc1) + dc2;
            vflipped = width - (v + 1)
            z = depth[u, vflipped]

            # z = depth[u][v];

            # if 1:
            # pcloudX = z * (u - px_d) / fx_d; # IS U FOR X OR IS U FOR Y??!?!
            pcloudX = z * (v - px_d) / fx_d
            pcloudY = z * (u - py_d) / fy_d
            # pcloudY = z * (v - py_d) / fy_d;

            # pcloudPoints = [pcloudX, pcloudY, pcloudZ, 1]
            # pcloudPoints = [pcloudX - center_x, pcloudY - center_y, pcloudZ]
            # pcloudPoints = [pcloudX + trackingTransform.GetElement(0,3),pcloudY + trackingTransform.GetElement(1,3),pcloudZ + trackingTransform.GetElement(2,3)]

            pcloud[u * vflipped][0] = pcloudX
            pcloud[u * vflipped][1] = pcloudY
            pcloud[u * vflipped][2] = z

    # Load original image to color point cloud points
    # correctedImage = cv2.imread(correctedFilename)

    return pcloud


def write_ply(file, point_cloud):
    vertices = point_cloud.shape[0]

    lines = ['ply\n',
             'format ascii 1.0\n',
             'element vertex {} \n'.format(vertices),
             'property float x\n',
             'property float y\n',
             'property float z\n',
             'end_header\n']

    # Body
    for p in point_cloud:
        lines.append(str(p[0]) + ' ')
        lines.append(str(p[1]) + ' ')
        lines.append(str(p[2]) + ' ')

    file.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert an EXR depth map to a PLY point cloud.")
    parser.add_argument('input_file', help='EXR input file (depth in \'Z\' channel)')
    parser.add_argument('output_file', help='PLY output file', type=argparse.FileType('w'))
    args = parser.parse_args()

    input_file = OpenEXR.InputFile(args.input_file)
    data_window = input_file.header()['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)

    (depth,) = input_file.channels("Z", Imath.PixelType(Imath.PixelType.HALF))
    depth = np.fromstring(depth, dtype=np.float16)
    depth.shape = size[::-1]

    point_cloud = depth_map_to_point_cloud(depth, 100)
    write_ply(args.output_file, point_cloud)


if __name__ == '__main__':
    main()
