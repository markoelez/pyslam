#!/usr/bin/env python3

import sys
sys.path.append('lib')
import pygame
from pygame.locals import DOUBLEBUF

from multiprocessing import Process, Queue
import OpenGL.GL as gl
import pypangolin as pangolin
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

    # data queues
    self.q_pose = Queue()
    self.q_points = Queue()
    self.q_image = Queue()

    # start
    self.view_thread = Process(target=self.view)
    self.view_thread.start()

    #self.view()

  def view(self):
    pangolin.CreateWindowAndBind('Viewer', 1024, 768)
    
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    width, height = 400, 250

    # Camera Render Object (for view / scene browsing)
    scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))

    #  view in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(
      pangolin.Attach(0.0),
      pangolin.Attach(1.0),
      pangolin.Attach(0.0),
      pangolin.Attach(1.0),
      -640.0 / 480.0)

    dcam.SetHandler(pangolin.Handler3D(scam))

    # image
    # width, height = 400, 130
    dimg = pangolin.Display('image')
    dimg.SetBounds(
        pangolin.Attach(0.0),
        pangolin.Attach(height / 768.),
        pangolin.Attach(0.0),
        pangolin.Attach(width / 1024.),
        1024 / 768.)

    texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = np.ones((height, width, 3), 'uint8')


    #while not pangolin.ShouldQuit():
    while 1:
      gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
      gl.glClearColor(1.0, 1.0, 1.0, 1.0)
      dcam.Activate(scam)

      print(self.q_image.empty())
      # show image
      if not self.q_image.empty():
        image = self.q_image.get()
        if image.ndim == 3:
          image = image[::-1, :, ::-1]
        else:
          image = np.repeat(image[::-1, :, np.newaxis], 3, axis=2)
        image = cv2.resize(image, (width, height))
      texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
      dimg.Activate()
      gl.glColor3f(1.0, 1.0, 1.0)
      texture.RenderToViewport()

  def update(self):
    #print("updating")

    #print(self.system.current.image)
    self.q_image.put(self.system.current.image)

    pangolin.FinishFrame()

