#!/usr/bin/env python3

import sys
sys.path.append('lib')
import pygame
from pygame.locals import DOUBLEBUF

from multiprocessing import Process, Queue
import pypangolin as pangolin
import OpenGL.GL as gl
import numpy as np

class Display2D:
  def __init__(self, system, config):
    W, H = config.image_width, config.image_height 

    pygame.init()
    self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
    self.surface = pygame.Surface(self.screen.get_size())
    self.system = system

  def update(self):
    for _  in pygame.event.get():
      pass

    img = self.system.get_current_image()

    pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1)[:, :, [0, 1, 2]])
    self.screen.blit(self.surface, (0, 0))

    pygame.display.flip()


class Display3D:
  def __init__(self, system, config):
    self.system = system
    self.config = config 

    self.state = None # current image, points

    self.view()

  def view(self):
    pangolin.CreateWindowAndBind("Viewer", 1024, 768)
    
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    width, height = 400, 250

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))

    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(
      pangolin.Attach(0.0),
      pangolin.Attach(1.0),
      pangolin.Attach(0.0),
      pangolin.Attach(1.0),
      -640.0 / 480.0)

    self.dcam.SetHandler(pangolin.Handler3D(self.scam))

    self.dimg = pangolin.Display('image')
    self.dimg.SetBounds(
        pangolin.Attach(0.0),
        pangolin.Attach(height / 768.),
        pangolin.Attach(0.0),
        pangolin.Attach(width / 1024.),
        1024 / 768.)

    self.texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = np.ones((height, width, 3), 'uint8')

    self.dcam.Activate(self.scam)

  def refresh(self):
    # update image, points
    if self.system.points is not None:
      self.state = [None]*2
      self.state[0] = self.system.current.image
      self.state[1] = self.system.points

    if self.state == None:
      return

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)

    image, points = self.state

    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    self.dcam.Activate(self.scam)

    # draw points
    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawPoints(points)

    # draw image
    self.texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    self.dimg.Activate()
    gl.glColor3f(1.0, 1.0, 1.0)
    self.texture.RenderToViewport()

    pangolin.FinishFrame()

