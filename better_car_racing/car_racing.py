__credits__ = ["Andrea PIERRÉ"]

import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from .car_dynamics import Car
from . import car_dynamics
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle


try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[box2d]`"
    ) from e


# CUSTOM
from .ray_tracing import Ray 

rays: list[Ray] = []

# ENCUSTOOM


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 60  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)

CAR_COLORS = [(0.8, 0.0, 0.0), (0.0, 0.0, 0.8),
              (0.0, 0.8, 0.0), (0.0, 0.8, 0.8),
              (0.8, 0.8, 0.8), (0.0, 0.0, 0.0),
              (0.8, 0.0, 0.8), (0.8, 0.8, 0.0)]


def isLeft(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) >0

def isInRoad(pos, walls):
    for i in range(0, len(walls), 2):
        left = walls[i]
        right = walls[i+1]

        inleft = not isLeft(left[0], left[1], pos)
        inright = isLeft(right[0], right[1], pos)
        inup = not isLeft(left[1], right[1], pos)
        indown = isLeft(left[0], right[0], pos)

        if inleft and inright and inup and indown:
            return True
        
    return False

class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        if not obj or "tiles" not in obj.__dict__:
            return


        if begin:
            # if self.env.pos_flag[obj.car_id] == 1:
            obj.tiles.add(tile)
            # print(tile.road_visited)
            if not tile.road_visited[obj.car_id] and self.env.pos_flag[obj.car_id] == 1:
                tile.color[obj.car_id][:] = self.env.road_color
                tile.road_visited[obj.car_id] = True
                self.env.rewards[obj.car_id] += 1000.0 / len(self.env.track[obj.car_id])
                self.env.tile_visited_counts[obj.car_id] += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_counts[obj.car_id] / len(self.env.track[obj.car_id])
                    >= self.lap_complete_percent
                ):
                    self.env.new_lap[obj.car_id] = True
        else:
            obj.tiles.remove(tile)


class BetterCarRacing(gym.Env, EzPickle):
    """
    ## Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gymnasium/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: breaking

    If discrete there are 5 actions:
    - 0: do nothing
    - 1: steer left
    - 2: steer right
    - 3: gas
    - 4: brake

    ## Observation Space

    A top-down 96x96 RGB image of the car and race track.

    ## Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited,
    where N is the total number of tiles visited in the track. For example,
    if you have finished in 732 frames, your reward is
    1000 - 0.1*732 = 926.8 points.

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go
    outside the playfield - that is, far off the track, in which case it will
    receive -100 reward and die.

    ## Arguments
    `lap_complete_percent` dictates the percentage of tiles that must be visited by
    the agent before a lap is considered complete.

    Passing `domain_randomize=True` enables the domain randomized variant of the environment.
    In this scenario, the background and track colours are different on every reset.

    Passing `continuous=False` converts the environment to use discrete action space.
    The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    ## Reset Arguments
    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.
    Example usage:
    ```python
    import gymnasium as gym
    env = gym.make("CarRacing-v1", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    env.reset()

    # reset with colour scheme change
    env.reset(options={"randomize": True})

    # reset with no colour scheme change
    env.reset(options={"randomize": False})
    ```

    ## Version History
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ## References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": [
            "human",            # render to pygame
            "human_state",      # render to pygame but with black and white state
            "rgb_array",        # render to numpy array
            "state_pixels",     # render to numpy array but with black and white state  (width: height = 96:96)
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        num_agents = 1,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        direction: str = "CCW",
        random_direction: bool = True,
        lap_complete_percent: float = 1,
        domain_randomize: bool = False,
        continuous: bool = True,
        ray_length: float = 300,
        ray_angles: list[int] = [0, 30, 60, 90, 120, 150, 180],
        render_ray = False,
        die_if_grass = False
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.num_agents = num_agents

        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surfs: list[Optional[pygame.Surface]] = [None] * num_agents
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.cars: list[Optional[Car]] = [None] * num_agents
        self.rewards = np.zeros(num_agents)
        self.prev_rewards = np.zeros(num_agents)
        self.true_rewards = np.zeros(num_agents)
        self.tile_visited_counts = np.zeros(num_agents, dtype=np.uint32)
        self.verbose = verbose

        self.random_direction = random_direction

        if self.random_direction:
            self.direction = np.random.choice(["CW", "CCW"])
        else:
            self.direction = direction

        self.new_laps = np.zeros(num_agents, dtype=bool)
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.num_agents == 1:
            n_agents = []
        else:
            n_agents = [self.num_agents]

        if self.continuous:
            self.action_space = spaces.Box(
                np.tile([-1, 0, 0], (*n_agents, 1)),
                np.tile([+1, +1, +1], (*n_agents, 1)),
                shape = (*n_agents, 3),
                dtype=np.float32
            )  # steer, gas, brake
        else:
            if self.num_agents == 1:
                self.action_space = spaces.Discrete(5)
            else:
                self.action_space = spaces.Box(0, 4, shape=(self.num_agents,), dtype=np.uint8)
            
            # do nothing, left, right, gas, brake
        


        self.ray_length = ray_length
        self.ray_angles = ray_angles
        self.render_ray = render_ray
        self.die_if_grass = die_if_grass

        self.rays_value = np.zeros((self.num_agents, len(self.ray_angles)), dtype=np.float32)

        self.vels = np.zeros((self.num_agents, 7), dtype=np.float32)

        observation_image_space = spaces.Box(low=0, high=255, shape=(*n_agents, STATE_H, STATE_W, 3), dtype=np.uint8)
        observation_rays_space = spaces.Box(low=0, high=self.ray_length, shape=(*n_agents, len(self.ray_angles)), dtype=np.float32)
        observation_vels_space = spaces.Box(low=-np.inf, high=np.inf, shape=(*n_agents, 7), dtype=np.float32)
        observation_pos_flag = spaces.Box(-1, 1, (*n_agents, 1), dtype=np.int8)
        
        self.observation_space = spaces.Dict({
            'image': observation_image_space,
            'rays': observation_rays_space,
            'vels': observation_vels_space,
            'pos_flag': observation_pos_flag
        })


        self.observation_space._shape = {}

        for key, item in self.observation_space.spaces.items():
            self.observation_space._shape[key] = item.shape

        self.render_mode = render_mode

        self.grassed = False

        self.pos_flag = np.ones((self.num_agents, 1), dtype=np.int8)

        self.last_pos = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.last_vel = np.zeros((self.num_agents, 2), dtype=np.float32)

        self.n_close = np.zeros(self.num_agents, dtype=np.uint8)

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []

        for car in self.cars:
            assert car is not None
            car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

        self.state_road_color = np.array([255, 255, 255])
        self.state_bg_color = np.array([0, 0, 0])
        self.state_grass_color = np.array([0, 0, 0])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = [self.road_color + c for _ in range(self.num_agents)]
            t.road_visited = [False] * self.num_agents
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )

                
                self.road_border.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        


        track = np.array(track)

        self.track = np.tile(track, (self.num_agents, *[1 for _ in range(len(track.shape))]))

        return True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        global TRACK_WIDTH

        # TRACK_WIDTH = self.np_random.uniform(40, 60) / SCALE
        TRACK_WIDTH = 50 / SCALE

        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.rewards = np.zeros(self.num_agents)
        self.prev_rewards = np.zeros(self.num_agents)
        self.true_rewards = np.zeros(self.num_agents)
        self.tile_visited_counts = np.zeros(self.num_agents, dtype=np.uint32)
        self.t = 0.0
        self.new_laps = np.zeros(self.num_agents, dtype=bool)
        self.road_poly = []
        self.road_border = []

        self.last_pos = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.last_vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.n_close = np.zeros(self.num_agents, dtype=np.uint8)

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        if self.random_direction:
            self.direction = self.np_random.choice(["CW", "CCW"])

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )

        # CUSTOM
        x, y, w, h = (477, 160, 45, 83)

        self.car_box = (x, y, w, h)
        self.car_head = (x+w//2, y+h)

        self.rays = [
            Ray((x+w//2, y+h), angle, self.ray_length) for angle in self.ray_angles
        ]

        self.vels = np.zeros((self.num_agents, 7), dtype=np.float32)

        self.rays_value.fill(0)

        self.grassed = np.zeros(self.num_agents, dtype=bool)

        self.pos_flag = np.ones((self.num_agents, 1), dtype=np.int8)


        # END CUSTOM

        idxes = [i for i in range(self.num_agents)]


        angle, pos_x, pos_y = self.track[0][0][1:4]

        car_orders = np.random.choice(idxes, size=self.num_agents, replace=False)
        # car_orders = np.arange(self.num_agents)
        # car_orders = car_orders[::-1]


        for car_id in range(self.num_agents):
            line_spacing = 5
            lateral_spacing = 3

            car_order = car_orders[car_id]
            line_number = math.floor(car_order / 2)
            side = (2 * (car_order % 2)) - 1

            dx = self.track[0][-line_number * line_spacing][2] - pos_x
            dy = self.track[0][-line_number * line_spacing][3] - pos_y

            angle = self.track[0][-line_number * line_spacing][1]
        
            if self.direction == "CW":
                angle += np.pi

            norm_theta = angle - np.pi/2

            new_x = pos_x + dx + (lateral_spacing * np.sin(norm_theta) * side)
            new_y = pos_y + dy + (lateral_spacing * np.cos(norm_theta) * side)

            self.cars[car_id] = Car(self.world, angle, new_x, new_y)
            self.cars[car_id].hull.color = CAR_COLORS[car_id % len(CAR_COLORS)]

            for wheel in self.cars[car_id].wheels:
                wheel.car_id = car_id
        
        # self.car = Car(self.world, *self.track[car_id][0][1:4])

        # if self.render_mode == "human" or self.render_mode == "human_state":
        #     self.render()
        
        return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        for car in self.cars:
            assert car is not None

        if action is not None:
            # print(action)
            # print(self.action_space)
            if self.num_agents == 1 and self.action_space.contains(action):
                action = np.stack([action], axis=0)

            for car_id in range(self.num_agents):
                if self.continuous:
                    self.cars[car_id].steer(-action[car_id][0])
                    self.cars[car_id].gas(action[car_id][1])
                    self.cars[car_id].brake(action[car_id][2])

                else:
                    # if not self.action_space.contains(action[car_id]):
                    #     raise InvalidAction(
                    #         f"you passed the invalid action `{action[car_id]}`. "
                    #         f"The supported action_space is `{self.action_space}`"
                    #     )

                    steering_coef = 1
                    gas_coef = 0.8
                    brake_coef = 0.8
                    
                    self.cars[car_id].steer(-steering_coef * (action[car_id] == 1) + steering_coef * (action[car_id] == 2))
                    self.cars[car_id].gas(gas_coef * (action[car_id] == 3))
                    self.cars[car_id].brake(brake_coef * (action[car_id] == 4))
        
        for car in self.cars:
            car.step(1.0 / FPS)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        if self.num_agents == 1:
            self.state = {'image': self._render("state_pixels"), 'rays': self.rays_value[0], 'vels': self.vels[0], 'pos_flag': self.pos_flag[0]}
        else:
            self.state = {'image': self._render("state_pixels"), 'rays': self.rays_value, 'vels': self.vels, 'pos_flag': self.pos_flag}

        _eps = 1e-2
        for car_id in range(self.num_agents):
            if abs(self.last_pos[car_id][0] - self.cars[car_id].hull.position[0]) < _eps\
                and abs(self.last_pos[car_id][1] - self.cars[car_id].hull.position[1]) < _eps\
                and abs(self.last_vel[car_id][0] - self.cars[car_id].hull.linearVelocity[0]) < _eps\
                and abs(self.last_vel[car_id][1] - self.cars[car_id].hull.linearVelocity[1]) < _eps:
                self.n_close[car_id] += 1
            else:
                self.n_close[car_id] = 0
            
            self.last_pos[car_id] = np.array(self.cars[car_id].hull.position)
            self.last_vel[car_id] = np.array(self.cars[car_id].hull.linearVelocity)
        
            
        step_rewards = np.zeros(self.num_agents)
        
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.rewards -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            step_rewards = self.rewards - self.prev_rewards
            self.prev_rewards = self.rewards.copy()

            for car_id in range(self.num_agents):
                self.cars[car_id].fuel_spent = 0.0

                vel = self.cars[car_id].hull.linearVelocity

                # if car is moving fast, compute car_angle using v_x, v_y
                if np.linalg.norm(vel) > 0.5:
                    car_angle = -math.atan2(vel[0], vel[1])
                else:
                    car_angle = self.cars[car_id].hull.angle
                
                # to [0, 2*pi]
                car_angle = (car_angle + (2*np.pi)) % (2 * np.pi)

                car_pos = np.array(self.cars[car_id].hull.position).reshape((1, 2))
                
                distance_to_tiles = np.linalg.norm(car_pos - np.array(self.track[car_id][:, 2:]), ord=2, axis=1)
                track_index = np.argmin(distance_to_tiles)

                desired_angle = self.track[car_id][track_index][1]

                if self.direction == 'CW':
                    desired_angle += np.pi

                # to [0, 2*pi]
                desired_angle = (desired_angle + (2*np.pi)) % (2 * np.pi)

                angle_diff = np.abs(car_angle - desired_angle)
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                
                if angle_diff >= np.pi/2:
                    self.pos_flag[car_id][0] = -1
                else:
                    self.pos_flag[car_id][0] = 1
                

                if self.grassed[car_id]:
                    step_rewards -= 0.5
                    self.pos_flag[car_id][0] = 0
                else:
                    if self.pos_flag[car_id][0] == -1:
                        step_rewards[car_id] -= 0.5*angle_diff

                if self.tile_visited_counts[car_id] == len(self.track[car_id]) or self.new_laps[car_id]:
                    # Truncation due to finishing lap
                    # This should not be treated as a failure
                    # but like a timeout
                    truncated = True

                x, y = self.cars[car_id].hull.position

                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD\
                    or (self.die_if_grass and self.grassed[car_id]):
                    terminated = True
                    step_rewards[car_id] = -500

                if self.n_close[car_id] > 100:
                    terminated = True
                    step_rewards[car_id] = -1000
        
        # if self.render_mode == "human" or self.render_mode == "human_state":
            # self.render()

        self.true_rewards += step_rewards
        
        if self.num_agents == 1:
            step_rewards = step_rewards[0]
        
        if self.num_agents == 1:
            self.state['pos_flag'] = self.pos_flag[0]
        else:
            self.state['pos_flag'] = self.pos_flag


        return self.state, step_rewards, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        results = [None] * self.num_agents

        if mode == 'human' or mode == 'human_state':
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((WINDOW_W * self.num_agents, WINDOW_H))
            self.screen.fill(0)
        
        for car_id in range(self.num_agents):
            results[car_id] = self._render_window(car_id, mode)
        
        if mode == 'human' or mode == 'human_state':
            pygame.display.flip()
        
        if self.num_agents == 1:
            return results[0]

        return np.stack(results, axis=0)

    def _render_window(self, car_id: int, mode: str):

        pygame.font.init()
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surfs[car_id] = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.cars[car_id] is not None
        # computing transformations
        angle = -self.cars[car_id].hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)

        zoom = ZOOM * SCALE

        # print(zoom)
        scroll_x = -(self.cars[car_id].hull.position[0]) * zoom
        scroll_y = -(self.cars[car_id].hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        roads = self._render_road(car_id, zoom, trans, angle, mode)

        current_car_color = [127, 127, 127]

        other_cars = []
        for _car_id in range(self.num_agents):
            if _car_id == car_id:
                self.cars[_car_id].draw(
                    self.surfs[car_id],
                    zoom,
                    trans,
                    angle,
                    mode not in ["state_pixels_list", "state_pixels", "human_state"],
                    current_car_color
                )
            else:
                poly = self.cars[_car_id].draw(
                    self.surfs[car_id],
                    zoom,
                    trans,
                    angle,
                    mode not in ["state_pixels_list", "state_pixels", "human_state"],
                    [0, 0, 0]
                )

                other_cars.append(poly)
        
        if isInRoad(self.car_head, roads):
            self.grassed[car_id] = False
           
        else:
            self.grassed[car_id] = True
            # self.rays_value[car_id].fill(0)
        
        walls = roads.copy()
        for poly in other_cars:
            wall = []
            for i in range(4):
                wall.append([poly[i], poly[(i+1)%4]])
            
            walls.extend(wall)
        
        for i in range(len(self.rays)):
            self.rays[i].cast(walls)
            if mode not in ["state_pixels_list", "state_pixels", "human_state"] and self.render_ray:
                self.rays[i].draw(self.surfs[car_id])
            self.rays_value[car_id][i] = self.rays[i].length

        self.surfs[car_id] = pygame.transform.flip(self.surfs[car_id], False, True)

        # showing stats
        self._render_indicators(car_id, WINDOW_W, WINDOW_H, mode)

        if mode == 'human' or mode == 'rgb_array':
            font = pygame.font.Font(pygame.font.get_default_font(), 42)
            text = font.render("%04i" % self.true_rewards[car_id], True, (255, 255, 255), (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
            self.surfs[car_id].blit(text, text_rect)


        if mode == "human" or mode == 'human_state':
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.blit(self.surfs[car_id], (WINDOW_W * car_id, 0))
            
        elif mode == "rgb_array":
            return self._create_image_array(self.surfs[car_id], (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surfs[car_id], (STATE_W, STATE_H))
        else:
            return self.isopen


    def _render_road(self, car_id, zoom, translation, angle, mode):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        bg_color = self.bg_color
        if mode in ["state_pixels_list", "state_pixels", "human_state"]:
            bg_color = self.state_bg_color
        self._draw_colored_polygon(
            self.surfs[car_id], field, bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )

        for poly in grass:
            grass_color = self.grass_color
            if mode in ["state_pixels_list", "state_pixels", "human_state"]:
                grass_color = self.state_grass_color
            
            self._draw_colored_polygon(
                self.surfs[car_id], poly, grass_color, zoom, translation, angle
            )
        

        walls = []


        # # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            if mode in ["state_pixels_list", "state_pixels", "human_state"]:
                color = [255, 255, 255]
            else:
                color = [int(c) for c in color[car_id]]

            tfm_poly = self._draw_colored_polygon(self.surfs[car_id], poly, color, zoom, translation, angle)
            if tfm_poly is not None:

                pos1, pos2, pos3, pos4 = tfm_poly

                walls.append((pos4, pos1))
                walls.append((pos3, pos2))
        
        if mode not in ["state_pixels_list", "state_pixels", "human_state"]:
            for poly, color in self.road_border:
                # converting to pixel coordinates
                poly = [(p[0], p[1]) for p in poly]
                color = [int(c) for c in color]
                self._draw_colored_polygon(self.surfs[car_id], poly, color, zoom, translation, angle)

        # for i, wall in enumerate(walls):
        #     if i%2==0:
        #         color = [0, 255, 0]
        #     else:
        #         color = [255, 0, 0]
            # pygame.draw.line(self.surfs[car_id], color, wall[0], wall[1], 3)
        

        return walls


    def _render_indicators(self, car_id, W, H, mode):
        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        # assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.cars[car_id].hull.linearVelocity[0])
            + np.square(self.cars[car_id].hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surfs[car_id], points=points, color=color)
        
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]

        self.vels[car_id] = np.array([
            true_speed, 
            self.cars[car_id].wheels[0].omega, 
            self.cars[car_id].wheels[1].omega, 
            self.cars[car_id].wheels[2].omega, 
            self.cars[car_id].wheels[3].omega,
            self.cars[car_id].wheels[0].joint.angle,
            self.cars[car_id].hull.angularVelocity
        ], dtype=np.float32)

        if mode in ['state_pixels_list', 'state_pixels', 'human_state']:
            return
        
        pygame.draw.polygon(self.surfs[car_id], color=color, points=polygon)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.cars[car_id].wheels[0].omega,
            vertical_ind(7, 0.01 * self.cars[car_id].wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.cars[car_id].wheels[1].omega,
            vertical_ind(8, 0.01 * self.cars[car_id].wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.cars[car_id].wheels[2].omega,
            vertical_ind(9, 0.01 * self.cars[car_id].wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.cars[car_id].wheels[3].omega,
            vertical_ind(10, 0.01 * self.cars[car_id].wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.cars[car_id].wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.cars[car_id].wheels[0].joint.angle),
            (0, 255, 0),
        )

        render_if_min(
            self.cars[car_id].hull.angularVelocity,
            horiz_ind(30, -0.8 * self.cars[car_id].hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(surface, poly, color)
            gfxdraw.filled_polygon(surface, poly, color)
        
            return poly

    @staticmethod
    def _create_image_array(screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

def main():
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = BetterCarRacing(render_mode="human")

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            print(s['rays'])
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                print(total_reward)
                break
    env.close()

if __name__ == "__main__":
    main()