from enum import Enum
import cv2
import numpy as np

class Prep:

  ATARI_HEIGHT = 84
  ATARI_WIDTH = 84

  class Type(Enum):
    IDENTITY = 1
    EXPAND_DIM = 2
    ATARI = 3

  def __init__(self, type, atari_history_size=4):
    self.type = type

    self.atari_history_size = atari_history_size

    self.history = []

  def process(self, state):

    if self.type == self.Type.IDENTITY:
      return state, False

    elif self.type == self.Type.EXPAND_DIM:
      return np.expand_dims(state, axis=0), False

    elif self.type == self.Type.ATARI:

      state_y = cv2.cvtColor(state, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
      state_y = cv2.resize(state_y, (self.ATARI_HEIGHT, self.ATARI_WIDTH), interpolation=cv2.INTER_LINEAR)

      if len(self.history) < self.atari_history_size:
        self.history.append(state_y)
        return None, True
      else:
        del self.history[0]
        self.history.append(state_y)

        stack = np.empty((1, self.ATARI_HEIGHT, self.ATARI_WIDTH, self.atari_history_size))
        for idx, frame in enumerate(self.history):
          stack[0, :, :, idx] = frame

        return stack, False