

class Map:
    def __init__(self):
        self.frames = []
        self.frame_idx = 0
    
    def add_frame(self, f):
        self.frame_idx += 1
        self.frames.append(f)
        return self.frame_idx 

