import numpy as np
from models import HillClimbing as Hill
from models import LocalBeamSearch as Beam

def main():
    stepsizes  = [0.01, 0.05, 0.1, 0.2]
    beam_widths = [2, 4, 8, 16]
    low, high = 0, 10

    hill_1, hill_2 = Hill(f1, stepsizes), Hill(f2, stepsizes)
    beam_1, beam_2 = Beam(f1, beam_widths), Beam(f2, beam_widths)

    hill_1.start(low, high)
    hill_2.start(low, high)
    beam_1.start(low, high)
    beam_2.start(low, high)

def f1(x, y):
    return np.sin(x / 2) + np.cos(2 * y)

def f2(x, y):
    return -abs(x - 2) - abs(0.5*y + 1) + 3

if __name__ == '__main__':
    main()
