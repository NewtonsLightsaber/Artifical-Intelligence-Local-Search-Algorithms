import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import cufflinks
from plotly.offline import iplot
from models import HillClimbing as Hill
from models import LocalBeamSearch as Beam

cufflinks.go_offline()

def main():
    stepsizes  = [0.01, 0.05, 0.1, 0.2]
    beam_widths = [2, 4, 8, 16]
    low, high = 0, 10

    hill_1, hill_2 = Hill(f1, stepsizes), Hill(f2, stepsizes)
    beam_1, beam_2 = Beam(f1, beam_widths), Beam(f2, beam_widths)

    hill_1.start(low, high)
    hill_2.start(low, high)

    #report_hill(hill_1)
    #report_hill(hill_2)

    stepsize_f1 = 0.1
    stepsize_f2 = 0.01

    beam_1.start(low, high, stepsize_f1)
    beam_2.start(low, high, stepsize_f2)

def report_beam(beam):
    pass

def table_beam(df):
    pass

def plot_beam(df):
    pass

def report_hill(hill):
    # Courtesy of Wolfram
    f1_max = 2

    df = pd.DataFrame({
        'step size': hill.stepsizes,
        'f value mean': hill.mean_f_values(),
        'f value standard deviation': hill.std_f_values(),
        'num steps mean': hill.mean_num_steps(),
        'num steps standard deviation': hill.std_num_steps(),
    })

    table_hill(df)
    plot_hill(df)

def table_hill(df):
    trace = go.Table(
        header=dict(values=['Step size',
                            'Mean f value', 'f value std deviation ',
                            'Mean num steps', 'num steps std deviation'],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['left'] * 5
        ),
        cells=dict(values=[df['step size'],
                            df['f value mean'],
                            df['f value standard deviation'],
                            df['num steps mean'],
                            df['num steps standard deviation']],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#EDFAFF'),
                    align = ['left'] * 5
        )
    )

    data = [trace]
    fig = dict(data=data)
    iplot(fig)

def plot_hill(df):
    df = df.set_index('step size')

    df['f value standard deviation'].iplot(
        mode='lines+markers',
        xTitle='Step size',
        title='f value standard deviation against step size'
    )

    df['f value mean'].iplot(
        mode='lines+markers',
        xTitle='Step size',
        title='f value mean against step size'
    )

def f1(x, y):
    return np.sin(x / 2) + np.cos(2 * y)

def f2(x, y):
    return -abs(x - 2) - abs(0.5*y + 1) + 3

if __name__ == '__main__':
    main()
