import struct
import numpy as np
import matplotlib.pyplot as plt
import sys


def read_bim(filename):
    with open(filename, "rb") as f:
        # --- Header ---
        header = f.read(4)
        if len(header) != 4:
            raise ValueError("Invalid or corrupted header")

        width, height = struct.unpack("<HH", header)
        print(f"Image size: {width} x {height}")

        bytes_per_row = (width + 7) // 8
        total_bytes = bytes_per_row * height
        data = f.read(total_bytes)

        if len(data) != total_bytes:
            raise ValueError("File truncated or corrupted")

        # --- Unpack bits row by row (MSB first, padded per row) ---
        bitmap = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            row_start = y * bytes_per_row
            row_bytes = data[row_start:row_start + bytes_per_row]

            # unpack MSB→LSB for each byte
            bits = np.unpackbits(np.frombuffer(row_bytes, dtype=np.uint8))
            bitmap[y, :] = bits[:width]  # discard padding bits

        return bitmap


def show_bitmap(bitmap, invert=False, save=True, show=False):
    if invert:
        bitmap = 1 - bitmap  # optional inversion

    plt.imshow(bitmap, cmap="gray", interpolation="nearest")
    plt.axis("off")

    if save:
        plt.savefig("output.png", bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file.bim>")
        sys.exit(1)

    filename = sys.argv[1]
    bmp = read_bim(filename)
    show_bitmap(bmp, invert=True)
