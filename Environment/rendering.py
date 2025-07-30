import pygame
import sys

# Grid Constants
GRID_SIZE = 5
CELL_SIZE = 100
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 3  # Slow enough to see what’s happening

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GREY = (211, 211, 211)

# --- Initialization Code ---
class FishFeedingRenderer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🐟 Fish Feeding Environment")
        self.clock = pygame.time.Clock()

    def draw_grid(self):
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GREY, (0, y), (WIDTH, y))

    def draw_agent(self, agent_pos):
        x, y = agent_pos
        pygame.draw.circle(self.screen, GREEN, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 30)

    def draw_fish(self, fish_status):
        for (fx, fy), is_hungry in fish_status:
            color = RED if is_hungry else BLUE
            pygame.draw.rect(self.screen, color, pygame.Rect(fx * CELL_SIZE + 20, fy * CELL_SIZE + 20, 60, 60))

    def render(self):
        # Event loop (clicking X exits)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(WHITE)
        self.draw_grid()

        agent_pos = self.env.agent_pos
        self.draw_agent(agent_pos)

        # NEW CODE that works with fish_hunger matrix
        fish_info = []
        for y in range(self.env.grid_size):
            for x in range(self.env.grid_size):
                is_hungry = self.env.fish_hunger[y][x] == 1
                fish_info.append(((x, y), is_hungry))
        self.draw_fish(fish_info)

        #self.render_mode = render_mode

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()
