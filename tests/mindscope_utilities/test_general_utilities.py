import numpy as np
import pandas as pd
from mindscope_utilities import event_triggered_response

def test_event_triggered_response():
    # make a time vector from -10 to 110
    t = np.arange(-10,110,0.01)

    # Make a dataframe with one column as time, and another column called 'sinusoid' defined as sin(2*pi*t)
    # The sinusoid column will have a period of 1
    df = pd.DataFrame({
        'time': t,
        'sinusoid': np.sin(2*np.pi*t)
    })

    # Make an event triggered response 
    etr = event_triggered_response(
        data = df,
        t = 'time',
        y = 'sinusoid',
        event_times = np.arange(100),
        t_before = 1,
        t_after = 1,
        output_sampling_rate = 10,
    )

    # Assert that the average value of the agrees with expectations
    assert np.isclose(etr.query('time == 0')['sinusoid'].mean(), 0, rtol=0.01)
    assert np.isclose(etr.query('time == 0.25')['sinusoid'].mean(), 1, rtol=0.01)
    assert np.isclose(etr.query('time == 0.5')['sinusoid'].mean(), 0, rtol=0.01)
    assert np.isclose(etr.query('time == 0.75')['sinusoid'].mean(), -1, rtol=0.01)
    assert np.isclose(etr.query('time == 1')['sinusoid'].mean(), 0, rtol=0.01)