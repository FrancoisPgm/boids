import argparse
import pygame as pg
import numpy as np
from boids import update_boids

PERCEPTION = 75
MARGIN = 50
SAFE_SPACE = 20
AVOID_FACTOR = 1  # to avoid edges
COHESION_FACTOR = 0.0005
SEPARATION_FACTOR = 0.05
ALIGNMENT_FACTOR = 0.05


class BoidSprite(pg.sprite.Sprite):
    def __init__(self, id, x, y, dx, dy, color=None):
        super().__init__()
        self.id = id
        self.image = pg.Surface((15, 15)).convert()  # TODO: use non hardcoded size
        self.image.set_colorkey(0)
        if color is None:
            self.color = pg.Color(0)
            self.color.hsva = (np.random.randint(0, 360), 90, 90)
        else:
            self.color = pg.Color(color)
        pg.draw.polygon(self.image, self.color, ((7, 0), (13, 14), (7, 11), (1, 14), (7, 0)))
        self.orig_image = self.image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.ang = np.arctan2(dx, -dy) * 180 / np.pi
        self.image = pg.transform.rotate(self.orig_image, -self.ang)

    def update(self, boids):
        self.rect.center = boids[self.id, :2]
        self.ang = np.arctan2(boids[self.id, 2], -boids[self.id, 3]) * 180 / np.pi
        self.image = pg.transform.rotate(self.orig_image, -self.ang)


def main(args):
    pg.init()
    pg.display.set_caption("Boids")

    # setup fullscreen or window mode
    if args.fullscreen:
        currentRez = (pg.display.Info().current_w, pg.display.Info().current_h)
        screen = pg.display.set_mode(currentRez, pg.SCALED | pg.NOFRAME | pg.FULLSCREEN, vsync=1)
        pg.mouse.set_visible(False)
    else:
        screen = pg.display.set_mode((args.width, args.height), pg.RESIZABLE | pg.SCALED, vsync=1)

    screen_width, screen_height = screen.get_size()

    boids = np.stack(
        (
            np.random.randint(MARGIN, screen_width - MARGIN, args.n_boids),
            np.random.randint(MARGIN, screen_height - MARGIN, args.n_boids),
            np.random.randn(args.n_boids) * args.speed,
            np.random.randn(args.n_boids) * args.speed,
        ),
        axis=1,
    )

    boid_sprite_group = pg.sprite.Group()
    for i, boid in enumerate(boids):
        boid_sprite_group.add(BoidSprite(i, boid[0], boid[1], boid[2], boid[3], color=args.boid_col))

    clock = pg.time.Clock()
    if args.show_fps:
        font = pg.font.Font(None, 30)

    # main loop
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT or e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                return

        clock.tick(args.fps)
        boids = update_boids(
            boids,
            PERCEPTION,
            SAFE_SPACE,
            args.speed,
            COHESION_FACTOR,
            SEPARATION_FACTOR,
            ALIGNMENT_FACTOR,
            AVOID_FACTOR,
            MARGIN,
            screen_width,
            screen_height,
        )
        screen.fill(args.bg_col)
        boid_sprite_group.update(boids)
        boid_sprite_group.draw(screen)

        if args.show_fps:
            screen.blit(font.render(str(int(clock.get_fps())), True, [0, 200, 0]), (8, 8))

        pg.display.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_boids", "-n", type=int, help="Number of boids, default=150.", default=150
    )
    parser.add_argument("--fps", type=int, default=60, help="FPS, default=60.")
    parser.add_argument("--show_fps", action="store_true", help="Show FPS.")
    parser.add_argument("--fullscreen", action="store_true", help="Show in full screen.")
    parser.add_argument(
        "--wrap", action="store_true", help="Wrap edges, otherwise edges are walls to be avoided."
    )
    parser.add_argument("--fish", action="store_true", help="Turn boids into fish.")
    parser.add_argument(
        "--width", type=int, default=1200, help="Window width, ignored if fullscreen, default=1200."
    )
    parser.add_argument(
        "--height", type=int, default=800, help="Window height, ignored if fullscreen, default=800."
    )
    parser.add_argument(
        "--bg_col",
        type=int,
        default=(0, 0, 0),
        nargs=3,
        help="Background color in RGB, default is balck.",
    )
    parser.add_argument(
        "--boid_col",
        type=int,
        default=None,
        nargs=3,
        help="Color of boids in RGB, default is random.",
    )
    parser.add_argument("--speed", "-s", type=int, default=15, help="Maximum speed, defult=15.")
    args = parser.parse_args()
    main(args)
    pg.quit()
