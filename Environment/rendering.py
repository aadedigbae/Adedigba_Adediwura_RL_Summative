# -*- coding: utf-8 -*-
# rendering.py (with Sprites)
import pygame
import sys
import os

GRID_SIZE = 5
CELL_SIZE = 100
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 3

ASSET_DIR = "assets"

class FishFeedingRenderer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🐟 Precision Aquaculture Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

        # Load sprites
        self.agent_sprite = pygame.image.load(os.path.join(ASSET_DIR, "agent.png"))
        self.agent_sprite = pygame.transform.scale(self.agent_sprite, (60, 60))

        self.fish_fed_sprite = pygame.image.load(os.path.join(ASSET_DIR, "fish_fed.png"))
        self.fish_fed_sprite = pygame.transform.scale(self.fish_fed_sprite, (60, 60))

        self.fish_hungry_sprite = pygame.image.load(os.path.join(ASSET_DIR, "fish_hungry.png"))
        self.fish_hungry_sprite = pygame.transform.scale(self.fish_hungry_sprite, (60, 60))

        # Optional: background tile
        self.bg_tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
        self.bg_tile.fill((220, 240, 255))  # Light blue water tile

    def draw_grid(self):
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                self.screen.blit(self.bg_tile, rect)
                pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)  # grid lines

    def draw_agent(self, agent_pos):
        x, y = agent_pos
        self.screen.blit(self.agent_sprite, (x * CELL_SIZE + 20, y * CELL_SIZE + 20))

    def draw_fish(self, fish_status):
        for (fx, fy), is_hungry in fish_status:
            sprite = self.fish_hungry_sprite if is_hungry else self.fish_fed_sprite
            self.screen.blit(sprite, (fx * CELL_SIZE + 20, fy * CELL_SIZE + 20))

    def draw_legend(self):
        self.screen.blit(self.agent_sprite, (10, HEIGHT - 30))
        self.screen.blit(self.fish_fed_sprite, (90, HEIGHT - 30))
        self.screen.blit(self.fish_hungry_sprite, (170, HEIGHT - 30))

        labels = [("Drone", 10), ("Fed", 90), ("Hungry", 170)]
        for label, x in labels:
            text = self.font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (x + 25, HEIGHT - 10))

    def render(self, step=0, reward=0.0):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))  # white background
        self.draw_grid()
        self.draw_agent(self.env.agent_pos)

        fish_info = []
        for y in range(self.env.grid_size):
            for x in range(self.env.grid_size):
                is_hungry = self.env.fish_hunger[y][x] == 1
                fish_info.append(((x, y), is_hungry))
        self.draw_fish(fish_info)

        self.draw_legend()

        # HUD
        info = f"Step: {step}  |  Reward: {reward:.2f}  |  Fish Fed: {self.env.fish_fed_count}"
        hud = self.font.render(info, True, (0, 0, 0))
        self.screen.blit(hud, (10, 5))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()
