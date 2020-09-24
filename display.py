#!/usr/bin/env python3

import sys
sys.path.append('lib')
import pygame
from pygame.locals import DOUBLEBUF

from multiprocessing import Process, Queue
#import pypangolin as pangolin
#import OpenGL.GL as gl
import numpy as np
import cv2

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

"""
class Display3D:
  def __init__(self, system, config):
    self.system = system
    self.config = config 

    self.state = None

    self.image_width = 300
    self.image_height = 250

    self.viewer_init(1024, 768)

  def viewer_init(self, w, h):
    pangolin.CreateWindowAndBind('Map Viewer', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
        pangolin.ModelViewLookAt(-10, -300, -8,
                                 0, 0, 0,
                                 0, -1, 0))
    self.handler = pangolin.Handler3D(self.scam)

    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0), pangolin.Attach(0.0), pangolin.Attach(1.0), w/h)
    self.dcam.SetHandler(self.handler)
    self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
    self.dcam.Activate()

    self.dimg = pangolin.Display('image')
    self.dimg.SetBounds(
        pangolin.Attach(0.0),
        pangolin.Attach(self.image_height / 768.),
        pangolin.Attach(0.0),
        pangolin.Attach(self.image_width/ 1024.),
        1024 / 768.)

    self.texture = pangolin.GlTexture(self.image_width, self.image_height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = np.ones((self.image_height, self.image_width, 3), 'uint8')

  def refresh(self):
    # update image, points
    if self.system.points is not None:
      self.state = [np.array(self.system.current.image), np.array(self.system.points)]

    self.paint()

  def paint(self):
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    self.dcam.Activate(self.scam)

    if self.state is not None:
      image, points = self.state
      
      if points.shape[0] != 0:
        # draw points 
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(points)

      # draw image
      if image.ndim == 3:
        image = image[::-1, :, ::-1]
      else:
        image = np.repeat(image[::-1, :, np.newaxis], 3, axis=2)
      image = cv2.resize(image, (self.image_width, self.image_height))
      self.texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
      self.dimg.Activate()
      gl.glColor3f(1.0, 1.0, 1.0)
      self.texture.RenderToViewport()

    pangolin.FinishFrame()

"""
