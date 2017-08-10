#!/usr/bin/env python3

"""
Load pretrained models and view their output. The image index is selected
interactively on the terminal, and matplotlib is used to display the output.
"""

import argparse
import logging
import sys

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import cmd
import depth_network.common as common
import random
import OpenEXR
import Imath

logging.basicConfig(level=logging.INFO)

render_model = None
depth_model = None


class TestCmd(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.prompt = "> "

    @staticmethod
    def _parse_index(args):
        if len(args) > 1:
            print("*** Too many parameters")
            return None
        elif len(args) == 0:
            index = random.randint(0, len(brdf_images) - 1)
            print("Randomly chose image: {}".format(index))
        else:
            try:
                index = int(args[0])
            except ValueError:
                print("*** Sample index is not a number: {}".format(args[0]))
                return None
            if index >= len(brdf_images):
                print("*** Sample index out of range: {}".format(index))
                return None

        return index

    def do_display(self, args):
        """
        display [INDEX]

        Display the specified sample.
        """
        args = [a.strip() for a in args.split(' ') if a]

        index = TestCmd._parse_index(args)
        if index is not None:
            images = process_sample(index)
            common.show_images(images.values())
            plt.show()

    def do_save(self, args):
        """
        save TYPE FILE [INDEX]

        Save output to file.
        """
        args = [a.strip() for a in args.split(' ') if a]

        if len(args) < 2:
            print("*** Too few arguments")
            return

        image_type = args[0]
        file_name = args[1]

        index = TestCmd._parse_index(args[2:])
        if index is not None:
            images = process_sample(index)
            if image_type not in images:
                print("*** Invalid image type: {}".format(image_type))
                return
            image = images[image_type]
            if image_type.startswith('brdf_') or image_type.startswith('rgb_'):
                cv2.imwrite(file_name, (image * 255)[..., ::-1])
            elif image_type.startswith('depth_'):
                header = OpenEXR.Header(*image.shape)
                depth_channel = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
                header['channels'] = {'Z': depth_channel}
                file = OpenEXR.OutputFile(file_name, header)
                file.writePixels({'Z': image.astype(np.float16).tostring()})
                file.close()

    def do_evaluate(self, args):
        """
        evaluate [START [END]]

        Print metrics for the specified range.
        :param args:
        :return:
        """
        args = [a.strip() for a in args.split(' ') if a]

        if len(args) < 2:
            end = brdf_images.shape[0]
        else:
            end = TestCmd._parse_index(args[1:])
            if end is None:
                return

        if len(args) == 0:
            start = 0
        else:
            start = TestCmd._parse_index(args[0:1])
            if start is None:
                return

        print(evaluate_batch(start, end))

    def do_quit(self, args):
        """
        Quit the program.
        """
        raise SystemExit()


def evaluate_batch(start=0, end=None):
    if end is None:
        end = brdf_images.shape[0]

    batches = process_batch(start, end)

    metrics = {}

    brdf_true = batches['brdf_true']

    if render_model is not None:
        render_metrics = metrics['render'] = {}

        rgb_true = batches['rgb_true']
        brdf_pred = batches['brdf_pred']

        brdf_diff = np.abs(brdf_true - brdf_pred)
        render_metrics['mse'] = (brdf_diff ** 2).mean()
        render_metrics['mae'] = brdf_diff.mean()

    if depth_model is not None:
        depth_metrics = metrics['depth'] = {}

        depth_true = batches['depth_true']
        depth_pred = batches['depth_pred']
        depth_pred_from_true = batches['depth_pred_from_true']

        depth_diff = np.abs(depth_true - depth_pred)
        depth_metrics['mse'] = (depth_diff ** 2).mean()
        depth_metrics['mae'] = depth_diff.mean()

        depth_diff_from_true = np.abs(depth_true - depth_pred_from_true)
        depth_metrics['mse_true'] = (depth_diff_from_true ** 2).mean()
        depth_metrics['mae_true'] = depth_diff_from_true.mean()

    return metrics


def process_batch(start, end):
    import depth_network.model

    batches = {}

    if rgb_images is not None:
        # Preprocess RGB ground truth image
        rgb_batch_true_pre = depth_network.model.preprocess_rgb_batch(rgb_images[start:end])
        rgb_batch_true = np.transpose(depth_network.model.postprocess_rgb_batch(rgb_batch_true_pre), (0, 2, 3, 1))
        batches['rgb_true'] = rgb_batch_true

    # Preprocess BRDF ground truth image
    brdf_batch_true_pre = depth_network.model.preprocess_brdf_batch(brdf_images[start:end])
    brdf_batch_true = np.transpose(depth_network.model.postprocess_rgb_batch(brdf_batch_true_pre), (0, 2, 3, 1))
    batches['brdf_true'] = brdf_batch_true

    if depth_images is not None:
        # Preprocess depth ground truth image
        depth_batch_true_pre = depth_network.model.preprocess_depth_batch(depth_images[start:end],
                                                                          swap_axes=swap_depth_axes)
        depth_batch_true = depth_network.model.postprocess_depth_batch(depth_batch_true_pre)
        batches['depth_true'] = depth_batch_true

    if render_model is not None:
        # Predict BRDF from ground truth RGB image
        brdf_batch_pred = render_model.predict(rgb_batch_true_pre, batch_size=128)
        brdf_batch_pred_post = depth_network.model.postprocess_rgb_batch(brdf_batch_pred)

        brdf_batch_pred_post = np.transpose(brdf_batch_pred_post, (0, 2, 3, 1))
        batches['brdf_pred'] = brdf_batch_pred_post

    if depth_model is not None:
        if render_model is not None:
            # Predict depth from predicted BRDF
            depth_batch_pred_post = depth_network.model.postprocess_depth_batch(
                depth_model.predict(brdf_batch_pred, batch_size=128))
            batches['depth_pred'] = depth_batch_pred_post

        # Predict depth from ground truth BRDF
        depth_batch_pred2_post = depth_network.model.postprocess_depth_batch(depth_model.predict(brdf_batch_true_pre))
        batches['depth_pred_from_true'] = depth_batch_pred2_post

    return batches


def process_sample(i):
    batches = process_batch(i, i + 1)

    # Remove batch axis and depth channel axis
    return {n: np.squeeze(b) for n, b in batches.items()}


def main():
    global render_model
    global depth_model
    global rgb_images
    global brdf_images
    global depth_images
    global logger
    global swap_depth_axes

    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="View outputs from trained models.")
    parser.add_argument('-r', '--render-model', help="render model weights file")
    parser.add_argument('-d', '--depth-model', help="depth model weights file")
    parser.add_argument('--rgb', help='RGB data file')
    parser.add_argument('--brdf', help='BRDF data file', required=True)
    parser.add_argument('--depth', help='depth data file')
    parser.add_argument('--swap-depth-axes', help='swap axes in depth images (required for sample data from paper)',
                        action='store_true')
    args = parser.parse_args()
    swap_depth_axes = args.swap_depth_axes

    if args.render_model is None and args.depth_model is None:
        parser.error("at least one of --depth-model and --render-model must be specified")
    if args.render_model is not None and args.rgb is None:
        parser.error("--render-model requires --rgb")
    if args.depth_model is not None and args.depth is None:
        parser.error("--depth-model requires --depth")

    # Keras takes a long time to load, so defer it until after argument parsing
    import depth_network.common as common
    import depth_network.model

    if args.render_model is not None:
        render_model = common.load_render_model(args.render_model, create=False)
        if render_model is None:
            logger.critical("Could not load render model.")
            sys.exit(1)

    if args.depth_model is not None:
        depth_model = common.load_depth_model(args.depth_model, create=False)
        if depth_model is None:
            logger.critical("Could not load depth model.")
            sys.exit(1)

    if args.rgb:
        rgb_data = h5py.File(args.rgb, 'r')
        rgb_images = rgb_data['RGB']
    else:
        rgb_images = None

    brdf_data = h5py.File(args.brdf, 'r')
    brdf_images = brdf_data['BRDF']

    if args.depth:
        depth_data = h5py.File(args.depth, 'r')
        depth_images = depth_data['Z']
    else:
        depth_images = None

    if rgb_images is not None and len(brdf_images) != len(rgb_images):
        logger.critical("Length of RGB data (%d) and BRDF data (%d) does not match", len(rgb_images), len(brdf_images))
        sys.exit(3)

    if depth_images is not None and len(brdf_images) != len(depth_images):
        logger.critical("Length of depth data (%d) and BRDF data (%d) does not match", len(depth_images),
                        len(brdf_images))
        sys.exit(3)

    TestCmd().cmdloop()


if __name__ == "__main__":
    main()
