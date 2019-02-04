import pickle
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import cufflinks
from plotly.offline import iplot
from pathlib import Path
from models import HillClimbing as Hill
from models import LocalBeamSearch as Beam

cufflinks.go_offline()
project_dir = Path(__file__).resolve().parents[0]

def main():
    stepsizes  = [0.01, 0.05, 0.1, 0.2]
    beam_widths = [2, 4, 8, 16]
    low, high = 0, 10

    hill_1, hill_2 = Hill(f1, stepsizes), Hill(f2, stepsizes)
    beam_1, beam_2 = Beam(f1, beam_widths), Beam(f2, beam_widths)

    #hill_1.start(low, high)
    #hill_2.start(low, high)

    #report(hill_1)
    #report(hill_2)

    stepsize_f1 = 0.1
    stepsize_f2 = 0.01
    repeat = 100

    beam_1.start(low, high, stepsize_f1, repeat)
    beam_2.start(low, high, stepsize_f2, repeat)

    df1 = report(beam_1)
    df2 = report(beam_2)

    pickle.dump(df1, open(project_dir / 'data' / '3b_f1_df.pkl', 'wb'))
    pickle.dump(df2, open(project_dir / 'data' / '3b_f2_df.pkl', 'wb'))

def report(localSearch):
    stats = {
        'f value mean': localSearch.mean_f_values(),
        'f value standard deviation': localSearch.std_f_values(),
        'num steps mean': localSearch.mean_num_steps(),
        'num steps standard deviation': localSearch.std_num_steps(),
    }
    if isinstance(localSearch, Hill):
        stats['step size'] = localSearch.stepsizes
    else:
        stats['beam width'] = localSearch.beam_widths

    df = pd.DataFrame(stats)

    return df

    #table(localSearch, df)
    #plot(localSearch, df)

def table(localSearch, df):
    x_label = 'Step size' if isinstance(localSearch, Hill) else 'Beam width'
    x_val = df[x_label.lower()]

    trace = go.Table(
        header=dict(values=[x_label,
                            'Mean f max', 'f max std deviation ',
                            'Mean num steps', 'num steps std deviation'],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['left'] * 5
        ),
        cells=dict(values=[x_val,
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

def plot(localSearch, df):
    x_label = 'Step size' if isinstance(localSearch, Hill) else 'Beam width'
    df = df.set_index(x_label.lower())

    df['f value standard deviation'].iplot(
        mode='lines+markers',
        xTitle=x_label,
        title='f max standard deviation against %s' % x_label.lower()
    )

    df['f value mean'].iplot(
        mode='lines+markers',
        xTitle=x_label,
        title='Mean f max against %s' % x_label.lower()
    )

def f1(x, y):
    return np.sin(x / 2) + np.cos(2 * y)

def f2(x, y):
    return -abs(x - 2) - abs(0.5*y + 1) + 3

if __name__ == '__main__':
    main()
