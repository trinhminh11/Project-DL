import numpy as np
import pygame

class Ray:
    def __init__(self, pos, angle, length):
        self.pos = np.array(pos)
        angle = np.deg2rad(angle)
        self.dir = np.array([np.cos(angle), np.sin(angle)])
        self.base_color = [0, 0, 0]
        self.color = self.base_color
        self.base_length = length
        self.length = self.base_length
    
    def draw(self, surface):
        pygame.draw.line(
            surface, self.color, self.pos, self.pos + self.dir * self.length
        )
    
    def cast(self, walls):
        closest = None
        record = np.inf
        for wall in walls:
            x1, y1, x2, y2 = wall[0][0], wall[0][1], wall[1][0], wall[1][1]
            x3, y3, x4, y4 = self.pos[0], self.pos[1], self.pos[0] + self.base_length*self.dir[0], self.pos[1] + self.base_length*self.dir[1]
            
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                continue
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
            
            if t > 0 and t < 1 and 0 < u and u < 1:
                pt = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
                d = np.linalg.norm(pt - self.pos)
                if d < record:
                    record = d
                    closest = pt
        
        if closest is not None:
            self.color = [255, 0, 0]
            self.length = record
        else:
            self.color = self.base_color
            self.length = self.base_length

        return closest
