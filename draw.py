import pygame
import sys

screen = None

def plot(points, color):
    for p in points:
        pygame.draw.rect(screen, color, ((100*p[0] + 400), (100*p[1]+400),3,3))

def init():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((800,800))

def pyframe(points, velocity, truth):
    screen.fill((0,0,0))
    plot(points, (255,0,0))
    plot(velocity, (0,0,255))
    plot([truth], (0,255,0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
