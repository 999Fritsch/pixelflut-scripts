import socket
from typing import Tuple, Union
import numpy as np
from PIL import Image
from random import randint, shuffle
from itertools import product

def voronoi_noise(size=128, n_seeds=40, metric="euclid", invert=False,
                  rng=None):
    """
    Generate an RGB Voronoi-noise image.

    Parameters
    ----------
    size     : int   – width & height of the square image in pixels
    n_seeds  : int   – feature (seed) points per colour channel
    metric   : str   – 'euclid' (d) or 'manhattan' (d₁) distance
    invert   : bool  – True makes cell centres bright, borders dark
    rng      : np.random.Generator or None

    Returns
    -------
    np.ndarray uint8 with shape (size, size, 3)
    """
    rng = np.random.default_rng(rng)

    # Pre-compute the x,y coordinates of every pixel
    yx = np.stack(np.mgrid[:size, :size], axis=-1)          # (h, w, 2)

    # Function that builds one greyscale slice
    def channel_slice():
        seeds = rng.random((n_seeds, 2)) * size
        diff  = yx[None, ...] - seeds[:, None, None, :]
        dist  = np.sqrt((diff ** 2).sum(-1))          # Euclidean
        f1    = dist.min(0)

        # normalise 0‒1, robust to NumPy ≥2.0
        f1 = (f1 - f1.min()) / (np.ptp(f1) + 1e-9)
        if invert:
            f1 = 1.0 - f1
        return (f1 * 255).astype(np.uint8)

    # Build R, G, B slices independently
    rgb = np.stack([channel_slice() for _ in range(3)], axis=-1)
    return rgb


class PixelClient:
    """
    Client for the TCP “pixel server” protocol.

    Connection is opened once in __init__ and kept for the lifetime
    of the object (or the surrounding with-block).

        SIZE            → “SIZE <width> <height>”
        PX x y          → read pixel
        PX x y RRGGBB   → write pixel (opaque)
        PX x y RRGGBBAA → write pixel (with alpha)

    Example
    -------
    >>> with PixelClient() as pc:
    ...     w, h = pc.get_size()
    ...     pc.set_pixel(w // 2, h // 2, 255, 0, 0)   # red dot
    """

    RECV_BYTES = 1024

    # ------------------------------------------------------------------ #
    #  Construction & teardown
    # ------------------------------------------------------------------ #

    def __init__(self, host: str = "localhost", port: int = 1337) -> None:
        self.host = host
        self.port = port
        self.sock = socket.create_connection((host, port))

    def close(self) -> None:
        """Close the underlying TCP connection."""
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                # Already closed by peer or never fully opened – ignore.
                pass
            self.sock.close()
            self.sock = None

    # Enable `with PixelClient() as pc: ...`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _send(self, data: str) -> None:
        self.sock.send(data.encode())

    def _recv(self) -> str:
        return self.sock.recv(self.RECV_BYTES).decode()

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert an RRGGBB hex string to (r, g, b)."""
        r_hex, g_hex, b_hex = hex_color[0:2], hex_color[2:4], hex_color[4:6]
        return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)

    # ------------------------------------------------------------------ #
    #  Protocol operations
    # ------------------------------------------------------------------ #

    def get_size(self) -> Tuple[int, int]:
        """Return (width, height) of the remote canvas."""
        self._send("SIZE\n")
        _, x, y = self._recv().split()
        return int(x), int(y)

    def get_pixel(self, x: int, y: int) -> Union[Tuple[int, int, int], bool]:
        """
        Return RGB tuple for pixel (x, y).

        If the server responds with “ERROR: Coordinates out of bounds”
        (or any line starting with “ERROR”), return False.
        """
        self._send(f"PX {x} {y}\n")
        response = self._recv()

        if response.startswith("ERROR"):
            return False

        _, _, _, color = response.split()
        return self.hex_to_rgb(color)

    def set_pixel(
        self,
        x: int,
        y: int,
        r: int,
        g: int,
        b: int,
        a: int = 255,
    ) -> None:
        """
        Write pixel (x, y) with the given colour.

        If alpha is 255 (default) we omit it, matching the original behaviour.
        """
        if a == 255:
            cmd = f"PX {x} {y} {r:02x}{g:02x}{b:02x}\n"
        else:
            cmd = f"PX {x} {y} {r:02x}{g:02x}{b:02x}{a:02x}\n"
        self._send(cmd)

import time
import random
from typing import Tuple, Optional


class PixelWorm:
    """
    A little ‘worm’ that crawls across a PixelServer canvas, leaving a trail.
    Direction and speed are recalculated from the underlying pixel’s colour.

    Parameters
    ----------
    client : PixelClient
        An *already-connected* PixelClient instance.
    start_x, start_y : int, optional
        Initial coordinates.  If omitted, choose a random point on the canvas.
    trail_colour : Tuple[int, int, int]
        RGB colour the worm draws with (default bright red).
    step_delay : float
        Seconds to sleep between drawing steps (default 0.1 s).
    """

    def __init__(
        self,
        client: "PixelClient",
        start_x: Optional[int] = None,
        start_y: Optional[int] = None,
        trail_colour: Tuple[int, int, int] = (255, 0, 0),
        step_delay: float = 0.1,
    ) -> None:
        self.client = client
        self.w, self.h = client.get_size()

        self.x = (
            start_x if start_x is not None else random.randint(0, self.w - 1)
        )
        self.y = (
            start_y if start_y is not None else random.randint(0, self.h - 1)
        )
        self.velocity: Tuple[int, int] = (0, 0)
        self.trail_colour = trail_colour
        self.step_delay = step_delay

        # Initial colour sample where the worm starts
        self.r, self.g, self.b = self.client.get_pixel(self.x, self.y)

    # ------------------------------------------------------------------ #
    #  Behaviour helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calc_new_velocity(red: int, green: int, blue: int) -> Tuple[int, int]:
        """
        Map colour (r, g, b) to a small velocity vector.

        The maths reproduces the original script’s logic:  
            vx = (r – 125)/12 – (b – 125)/12  
            vy = (g – 125)/12 – (b – 125)/12

        If both components round to 0, pick either x or y randomly
        and give it a random speed between 5 and 15 pixels per step.
        """
        vx = int((red - 125) / 12 - (blue - 125) / 12)
        vy = int((green - 125) / 12 - (blue - 125) / 12)

        if vx == 0 and vy == 0:
            # Ensure we keep moving
            if random.randint(0, 1) == 0:
                vx = random.randint(5, 15)
            else:
                vy = random.randint(5, 15)
        return vx, vy

    @staticmethod
    def _step_coord(pos: int, vel: int) -> int:
        """
        Move `pos` one pixel towards the direction of `vel`.

        Only ±1 or 0 movement per drawn pixel so our trail is continuous.
        """
        if vel > 0:
            return pos + 1
        elif vel < 0:
            return pos - 1
        return pos

    def _out_of_bounds(self, x: int, y: int) -> bool:
        """True if (x, y) lies outside the canvas."""
        return x < 0 or y < 0 or x >= self.w or y >= self.h

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        """
        Advance the worm by **one full velocity update**:

        *   Sample current pixel → compute new velocity.
        *   If next target is off the canvas, teleport to a new random spot.
        *   Otherwise walk one-pixel-at-a-time toward the target,
            drawing the trail colour at every sub-step.
        """
        # 1. Update velocity based on current colour
        self.velocity = self._calc_new_velocity(self.r, self.g, self.b)
        target_x = self.x + self.velocity[0]
        target_y = self.y + self.velocity[1]

        # 2. If we’d leave the canvas, jump elsewhere and resample colour
        if self._out_of_bounds(target_x, target_y):
            self.x = random.randint(0, self.w - 1)
            self.y = random.randint(0, self.h - 1)
            self.r, self.g, self.b = self.client.get_pixel(self.x, self.y)
            return  # nothing drawn during teleport
        
        # 3. Sample colour under new head for the next iteration
        self.r, self.g, self.b = self.client.get_pixel(target_x, target_y)

        # 4. Walk pixel-by-pixel towards (target_x, target_y)
        while self.x != target_x or self.y != target_y:
            if self.x != target_x:
                self.x = self._step_coord(self.x, self.velocity[0])
            if self.y != target_y:
                self.y = self._step_coord(self.y, self.velocity[1])

            # Draw the trail
            self.client.set_pixel(
                self.x, self.y, *self.trail_colour
            )
            time.sleep(self.step_delay)


    def run(self, steps: Optional[int] = None) -> None:
        """
        Keep the worm moving.

        Parameters
        ----------
        steps : int or None
            * If `None`, run indefinitely (until Ctrl-C).
            * Otherwise perform exactly `steps` calls to `step()`.
        """
        try:
            if steps is None:
                while True:
                    self.step()
            else:
                for _ in range(steps):
                    self.step()
        except KeyboardInterrupt:
            # Graceful stop when user presses Ctrl-C
            pass

from pathlib import Path
from random import shuffle
from typing import Tuple

import numpy as np
from PIL import Image


def print_image_by_path(
    client: "PixelClient",
    image_path: str | Path,
    *,
    top_left: Tuple[int, int] = (0, 0),
    dither_alpha: bool = False,
) -> None:
    """
    Draw a local image file onto the PixelServer canvas.

    Parameters
    ----------
    client : PixelClient
        An **open** PixelClient instance.
    image_path : str or Path
        Path to a bitmap that Pillow can read (PNG, JPEG, …).
    top_left : (x, y)
        Where the image’s (0, 0) pixel should be placed on the canvas.
    dither_alpha : bool
        If True, fully transparent pixels are skipped; otherwise they are
        written as fully transparent (requires server alpha support).
    """
    im = Image.open(image_path).convert("RGBA")
    width, height = im.size
    off_x, off_y = top_left

    for y in range(height):
        for x in range(width):
            r, g, b, a = im.getpixel((x, y))

            # Optionally skip fully-transparent pixels
            if dither_alpha and a == 0:
                continue

            client.set_pixel(off_x + x, off_y + y, r, g, b, a)


def print_image_by_array(
    client: "PixelClient",
    arr: np.ndarray,
    *,
    top_left: Tuple[int, int] = (0, 0),
    scramble: bool = True,
) -> None:
    """
    Render a NumPy array as an image on the PixelServer canvas.

    Parameters
    ----------
    client : PixelClient
        An active PixelClient connection.
    arr : np.ndarray
        A uint8 array of shape (height, width, 3) for RGB or (height, width, 4) for RGBA.
    top_left : tuple of int, optional
        Coordinates (x, y) for the top-left corner where the image will be drawn.
    scramble : bool, optional
        If True, pixels are drawn in random order for a loading effect; if False, pixels are drawn row by row.
    """
    # Validate input array shape
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("array must have shape (H, W, 3) or (H, W, 4)")

    h, w, channels = arr.shape
    xs = list(range(w))
    ys = list(range(h))
    off_x, off_y = top_left

    if not scramble:
        # Draw pixels in row-major order
        for y in ys:
            for x in xs:
                pixel = arr[y, x]
                r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
                a = int(pixel[3]) if channels == 4 else 255
                client.set_pixel(off_x + x, off_y + y, r, g, b, a)
    else:
        # Draw pixels in random order for a visual effect
        scrambled_list = list(product(xs, ys))
        shuffle(scrambled_list)
        for x, y in scrambled_list:
            pixel = arr[y, x]
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            a = int(pixel[3]) if channels == 4 else 255
            client.set_pixel(off_x + x, off_y + y, r, g, b, a)

    



