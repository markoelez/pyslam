#!/usr/bin/env python3

import sys
import pygame
from pygame.locals import DOUBLEBUF
from multiprocessing import Process, Queue

class Display2D:
  def __init__(self, W, H):
    pygame.init()
    self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
    self.surface = pygame.Surface(self.screen.get_size())

  def paint(self, img):
    for _  in pygame.event.get():
      pass

    pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1)[:, :, [0, 1, 2]])
    self.screen.blit(self.surface, (0, 0))

    pygame.display.flip()


class Display3D:
  def __init__(self, system, config):
    self.system = system
    self.config = config 

    # data queues
    self.q_pose = Queue()
    self.q_points = Queue()
    self.q_image = Queue()
