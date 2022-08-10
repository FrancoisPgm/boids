import os

# don't show pygame hello message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import argparse
import pygame as pg
import numpy as np
from boids import update_boids
from pygame_widgets.slider import Slider
import pygame_widgets

PERCEPTION = 75
PRED_PERCEPTION = 100
MARGIN = 50
SAFE_SPACE = 20
AVOID_FACTOR = 1
AVOID_PRED_FACTOR = 0.05
COHESION_FACTOR = 0.005
SEPARATION_FACTOR = 0.05
ALIGNMENT_FACTOR = 0.05


def scaling_func(x):
    return (np.exp(x * 4) - 1) / (np.exp(2) - 1)


def make_sliders(screen, screen_width, screen_height):
    slider_width = int(0.2 * screen_width)
    slider_height = min(15, 0.2 * screen_height)
    slider_x_margin = int(0.2 * 0.2 * screen_width)
    slider_y = int(0.9 * screen_height)
    args = (slider_width, slider_height)
    kwargs = {"min": 0, "max": 1, "step": 0.01, "handleColour": (100, 100, 100)}
    cohesion_slider = Slider(screen, slider_x_margin, slider_y, *args, **kwargs)
    alignmnet_slider = Slider(screen, 2 * slider_x_margin + slider_width, slider_y, *args, **kwargs)
    separation_slider = Slider(
        screen, 3 * slider_x_margin + 2 * slider_width, slider_y, *args, **kwargs
    )
    perception_slider = Slider(
        screen, 4 * slider_x_margin + 3 * slider_width, slider_y, *args, **kwargs
    )
    return cohesion_slider, alignmnet_slider, separation_slider, perception_slider


class BoidSprite(pg.sprite.Sprite):
    def __init__(self, id, x, y, dx, dy, color=None):
        super().__init__()
        self.id = id
        self.ang = np.arctan2(dx, -dy) * 180 / np.pi
        self.image = pg.Surface((15, 15)).convert()  # TODO: use non hardcoded size
        self.image.set_colorkey(0)
        self.angular_color = color == "angular"
        if color is None:
            color = pg.Color(0)
            color.hsva = (np.random.randint(0, 360), 90, 90)
            color = (color.r, color.g, color.b)
        elif self.angular_color:
            color = pg.Color(0)
            color.hsva = (self.ang % 360, 90, 90)
            color = (color.r, color.g, color.b)
        self.color = pg.Color(color)

        pg.draw.polygon(self.image, self.color, ((7, 0), (13, 14), (7, 11), (1, 14), (7, 0)))
        self.orig_image = self.image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.image = pg.transform.rotate(self.orig_image, -self.ang)

    def update(self, boids):
        self.rect.center = boids[self.id, :2]
        self.ang = np.arctan2(boids[self.id, 2], -boids[self.id, 3]) * 180 / np.pi
        if self.angular_color:
            color = pg.Color(0)
            color.hsva = (self.ang % 360, 90, 90)
            color = (color.r, color.g, color.b)
            self.color = pg.Color(color)
            pg.draw.polygon(
                self.orig_image, self.color, ((7, 0), (13, 14), (7, 11), (1, 14), (7, 0))
            )
        self.image = pg.transform.rotate(self.orig_image, -self.ang)


def reset_boids(n_boids, width, height, margin, max_speed):
    boids = np.stack(
        (
            np.random.randint(margin, width - margin, n_boids),
            np.random.randint(margin, height - margin, n_boids),
            np.random.randn(n_boids) * max_speed,
            np.random.randn(n_boids) * max_speed,
        ),
        axis=1,
    )
    return boids


def main(args):
    global COHESION_FACTOR
    pg.init()
    pg.display.set_caption("Boids")

    # setup fullscreen or window mode
    if args.fullscreen:
        currentRez = (pg.display.Info().current_w, pg.display.Info().current_h)
        screen = pg.display.set_mode(currentRez, pg.SCALED | pg.NOFRAME | pg.FULLSCREEN, vsync=1)
        pg.mouse.set_visible(False)
    else:
        screen = pg.display.set_mode((args.width, args.height), pg.RESIZABLE)  # , vsync=1)

    screen_width, screen_height = screen.get_size()
    cohesion_slider, alignment_slider, separation_slider, perception_slider = make_sliders(
        screen, screen_width, screen_height
    )

    boids = reset_boids(args.n_boids, screen_width, screen_height, MARGIN, args.speed)

    boid_sprite_group = pg.sprite.Group()
    color = "angular" if args.ang_col else args.boid_col
    for i, boid in enumerate(boids):
        boid_sprite_group.add(BoidSprite(i, boid[0], boid[1], boid[2], boid[3], color=color))

    clock = pg.time.Clock()
    font = pg.font.Font(None, 25)
    show_sliders = False

    # main loop
    while True:
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return
            elif event.type == pg.KEYDOWN and event.key == pg.K_r:
                boids = reset_boids(args.n_boids, screen_width, screen_height, MARGIN, args.speed)
            elif event.type == pg.KEYDOWN and event.key == pg.K_s:
                show_sliders = not show_sliders
            elif event.type == pg.VIDEORESIZE:
                screen_width, screen_height = screen.get_size()
                # TODO: reposition and rescale sliders

        cohesion_fact = scaling_func(cohesion_slider.getValue()) * COHESION_FACTOR
        alignment_fact = scaling_func(alignment_slider.getValue()) * ALIGNMENT_FACTOR
        separation_fact = scaling_func(separation_slider.getValue()) * SEPARATION_FACTOR
        perception = perception_slider.getValue() * 2 * PERCEPTION

        predators = []
        mouse_speed = 0
        mouse_x, mouse_y = pg.mouse.get_pos()
        if mouse_x and mouse_y and mouse_x < screen_width and mouse_y < screen_height:
            mouse_dx, mouse_dy = pg.mouse.get_rel()
            mouse_speed = PRED_PERCEPTION + np.sqrt(mouse_dx**2 + mouse_dy**2)
            predators.append([mouse_x, mouse_y, mouse_dx, mouse_dy])
            predators = np.array(predators)

        clock.tick(args.fps)
        boids = update_boids(
            boids=boids,
            perception=perception,
            predators=predators,
            pred_perception=mouse_speed,
            safe_space=SAFE_SPACE,
            max_speed=args.speed,
            cohesion_factor=cohesion_fact,
            separation_factor=separation_fact,
            alignment_factor=alignment_fact,
            avoid_factor=AVOID_FACTOR,
            avoid_pred_factor=AVOID_PRED_FACTOR,
            margin=MARGIN,
            width=screen_width,
            height=screen_height,
        )
        screen.fill(args.bg_col)
        boid_sprite_group.update(boids)
        boid_sprite_group.draw(screen)

        if args.show_fps:
            screen.blit(font.render(str(int(clock.get_fps())), True, [0, 200, 0]), (8, 8))

        if show_sliders:
            pygame_widgets.update(events)
            text_y = cohesion_slider.getY() + int(1.5 * cohesion_slider.getHeight())
            screen.blit(
                font.render(f"cohesion {cohesion_fact:.2e}", True, [100, 100, 100]),
                (cohesion_slider.getX() + int(0.2 * cohesion_slider.getWidth()), text_y),
            )
            screen.blit(
                font.render(f"alignment {alignment_fact:.2e}", True, [100, 100, 100]),
                (alignment_slider.getX() + int(0.2 * alignment_slider.getWidth()), text_y),
            )
            screen.blit(
                font.render(f"separation {separation_fact:.2e}", True, [100, 100, 100]),
                (separation_slider.getX() + int(0.2 * separation_slider.getWidth()), text_y),
            )
            screen.blit(
                font.render(f"perception {perception}", True, [100, 100, 100]),
                (perception_slider.getX() + int(0.2 * perception_slider.getWidth()), text_y),
            )
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
    parser.add_argument(
        "--ang_col",
        action="store_true",
        help="Change color of boids according to their angle, if set it overrides boid_col.",
    )
    parser.add_argument("--speed", "-s", type=int, default=15, help="Maximum speed, defult=15.")
    args = parser.parse_args()
    main(args)
    pg.quit()
