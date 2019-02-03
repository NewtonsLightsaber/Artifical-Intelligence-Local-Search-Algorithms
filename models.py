import numpy as np

class LocalSearch:
    def __init__(self, f, specs):
        self.f = f
        self.num_steps = [[] for _ in range(len(specs))]
        self.f_values = [[] for _ in range(len(specs))]

    def get_xy_0(self, low=None, high=None, size=None):
        if not (min is None or \
                max is None or \
                size is None):
            xy_0_list = np.random.uniform(
                low=low, high=high, size=(size, 2)
                ).tolist()
        else:
            raise Exception(
                'Must input low value, high value, and size of desired list'
                )
        return xy_0_list


class HillClimbing(LocalSearch):
    """
    Hill Climbing search algorithm for bivariate functions
    """
    def __init__(self, f, stepsizes):
        self.stepsizes = stepsizes
        super().__init__(f, stepsizes)

    def start(self, low, high):
        for i, stepsize in enumerate(self.stepsizes):
            xy_0_list = self.get_xy_0(low=low, high=high, size=100)

            for j, (x0, y0) in enumerate(xy_0_list):
                f_val, num_steps = self.hillclimb(
                    xy_0=(x0, y0),
                    low_high=(low, high),
                    stepsize=stepsize)

                self.save_num_steps(num_steps, i)
                self.save_f_value(f_val, i)

    def hillclimb(self, xy_0, low_high, stepsize):
        low, high = low_high
        x, y = xy_0
        cur_val = self.f(x, y)
        num_steps = 1

        while True:
            neighbours = self.get_neighbours(
                xy=(x, y),
                low_high=(low, high),
                stepsize=stepsize)

            vals = [self.f(x, y) for x, y in neighbours]
            next_val = max(vals)
            xi, yi = neighbours[vals.index(next_val)]

            if next_val <= cur_val:
                break
            else:
                x, y = xi, yi
                cur_val = next_val
                num_steps += 1

        return cur_val, num_steps

    def get_neighbours(self, xy, low_high, stepsize):
        x, y = xy
        low, high = low_high
        return (
            (x, min(y + stepsize, high)), # top
            (x, max(y - stepsize, low)), # bottom
            (max(x - stepsize, low), y), # left
            (min(x + stepsize, high), y), # right

            (min(x + stepsize, high), min(y + stepsize, high)), # top-right
            (min(x + stepsize, high), max(y - stepsize, low)), # bottom-right
            (max(x - stepsize, low), max(y - stepsize, low)), # bottom-left
            (max(x - stepsize, low), min(y + stepsize, high)), # top-left
        )

    def mean_num_steps(self):
        return [np.mean(num_steps) for num_steps in self.num_steps]

    def mean_f_values(self):
        return [np.mean(f_values) for f_values in self.f_values]

    def std_num_steps(self):
        return [np.std(num_steps) for num_steps in self.num_steps]

    def std_f_values(self):
        return [np.std(f_values) for f_values in self.f_values]

    def save_num_steps(self, num_steps, i):
        self.num_steps[i].append(num_steps)

    def save_f_value(self, f_value, i):
        self.f_values[i].append(f_value)


class LocalBeamSearch(LocalSearch):
    """
    Local Beam search parallelizing hill climbing
    """
    def __init__(self, f, beam_widths):
        self.beam_widths = beam_widths
        super().__init__(f, beam_widths)

    def start(self, low, high):
        pass
