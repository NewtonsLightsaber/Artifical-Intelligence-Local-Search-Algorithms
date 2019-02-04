import numpy as np

class LocalSearch:
    def __init__(self, f, specs):
        self.f = f
        self.num_steps = [[] for _ in range(len(specs))]
        self.f_values = [[] for _ in range(len(specs))]

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

    def get_xy_0(self, low, high, size):
        return np.random.uniform(low=low, high=high, size=(size, 2)).tolist()

    def mean_num_steps(self):
        """
        Return mean number of steps for each step size
        """
        return [np.mean(num_steps) for num_steps in self.num_steps]

    def mean_f_values(self):
        """
        Return mean f value (f being the function to maximize)
        for each step size
        """
        return [np.mean(f_values) for f_values in self.f_values]

    def std_num_steps(self):
        return [np.std(num_steps) for num_steps in self.num_steps]

    def std_f_values(self):
        return [np.std(f_values) for f_values in self.f_values]

    def save_num_steps(self, num_steps, i):
        self.num_steps[i].append(num_steps)

    def save_f_value(self, f_value, i):
        self.f_values[i].append(f_value)


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

            for x0, y0 in xy_0_list:
                x, y = x0, y0
                cur_val = self.f(x, y)
                num_steps = 1

                while True:
                    next_val, (xi, yi) = self.hillclimb(
                        x=x,
                        y=y,
                        low_high=(low, high),
                        stepsize=stepsize)

                    if next_val <= cur_val:
                        break
                    else:
                        x, y = xi, yi
                        cur_val = next_val
                        num_steps += 1

                self.save_num_steps(num_steps, i)
                self.save_f_value(cur_val, i)

    def hillclimb(self, x, y, low_high, stepsize):
        low, high = low_high

        neighbours = self.get_neighbours(
            xy=(x, y),
            low_high=(low, high),
            stepsize=stepsize)

        vals = [self.f(x, y) for x, y in neighbours]
        next_val = max(vals)
        xi, yi = neighbours[vals.index(next_val)]
        return next_val, (xi, yi)


class LocalBeamSearch(LocalSearch):
    """
    Local Beam search parallelizing hill climbing
    """
    def __init__(self, f, beam_widths):
        self.beam_widths = beam_widths
        super().__init__(f, beam_widths)

    def start(self, low, high, stepsize, repeat):
        for i, beam_width in enumerate(self.beam_widths):
            for _ in range(repeat):
                xy_0_list = self.get_xy_0(low=low, high=high, size=beam_width)
                f_val, num_steps = self.beamsearch(
                    xy_0_list=xy_0_list,
                    beam_width=beam_width,
                    low_high=(low, high),
                    stepsize=stepsize)

                self.save_num_steps(num_steps, i)
                self.save_f_value(f_val, i)

    def beamsearch(self, xy_0_list, beam_width, low_high, stepsize):
        beam = [(x, y, self.f(x, y)) for x, y in xy_0_list]
        num_steps = 1

        while True:
            #print('beam $d: ' % num_steps, end=''); print(beam)
            cur_vals = [f for x, y, f in beam]
            neighbours = [
                neighbour
                for x, y, f in beam
                for neighbour in self.get_neighbours((x, y), low_high, stepsize)
            ]
            next_states = [(x, y, self.f(x, y)) for x, y in neighbours]
            k_best = sorted(
                beam + next_states,
                key=lambda tup: tup[2],
                reverse=True
            )[:beam_width]
            k_best_vals = [f for x, y, f in k_best]

            print(k_best_vals)
            print(cur_vals)
            if max(k_best_vals) <= min(cur_vals):
                break
            else:
                beam = k_best
                num_steps += 1

        return max(cur_vals), num_steps
