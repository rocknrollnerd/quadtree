from PIL import Image

import argparse
import matplotlib.pyplot as plt
import numpy as np


# split bbox side as close to the middle as possible
# note: side is at least 1
def split(side):
    if side % 2 == 0:
        return side // 2
    else:
        return (side - 1) // 2


def recurse(args, inp, out, bbox):
    top, left, bottom, right = bbox
    height = bottom - top
    width = right - left
    if height < args.min_side or width < args.min_side:
        return

    # paint the output by averaging each channel.
    max_std = 0
    max_iqr = 0
    for i in range(inp.shape[2]):
        chunk = inp[top: bottom, left: right, i].astype(np.float32)
        out[top: bottom, left: right, i] = np.uint8(chunk.mean())
        # should we add borders?
        if args.border_width > 0:
            bw = args.border_width
            out[top:top + bw, left:right, :] = 0
            out[bottom - bw:bottom, left:right, :] = 0
            out[top:bottom, left:left + bw, :] = 0
            out[top:bottom, right - bw:right, :] = 0

        # compute chunk statistics for stopping
        max_std = max(max_std, chunk.std())
        max_iqr = max(max_iqr, np.percentile(chunk.ravel(), 75) - np.percentile(chunk.ravel(), 25))

    # stop if necessary
    if args.stopping_criterion == 'std' and max_std < args.stopping_threshold:
        return
    if args.stopping_criterion == 'iqr' and max_iqr < args.stopping_threshold:
        return

    # divide bbox in 4 parts, possibly unevenly
    mid_height = split(height)
    mid_width = split(width)

    # recurse
    recurse(args, inp, out, (top, left, top + mid_height, left + mid_width))
    recurse(args, inp, out, (top, left + mid_width, top + mid_height, right))
    recurse(args, inp, out, (top + mid_height, left, bottom, left + mid_width))
    recurse(args, inp, out, (top + mid_height, left + mid_width, bottom, right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--min_side', type=int, default=3)
    parser.add_argument('--stopping_criterion', type=str, choices=('std', 'iqr'), default='std')
    parser.add_argument('--stopping_threshold', type=float, default=10.0)
    parser.add_argument('--border_width', type=int, default=0)
    args = parser.parse_args()
    assert args.min_side > 0
    assert args.border_width >= 0

    inp = np.asarray(Image.open(args.input))
    # print(inp.dtype, inp, inp.max(), inp.min())
    height, width, _ = inp.shape
    out = np.zeros_like(inp)
    bbox = (0, 0, height, width)  # top, left, bottom, right

    recurse(args, inp, out, bbox)

    Image.fromarray(out).save(args.output)


if __name__ == '__main__':
    main()