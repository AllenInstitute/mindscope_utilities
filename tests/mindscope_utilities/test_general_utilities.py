import numpy as np
import pandas as pd
from mindscope_utilities import event_triggered_response
from mindscope_utilities import index_of_nearest_value
from mindscope_utilities import slice_inds_and_offsets

def test_event_triggered_response():
    # make a time vector from -10 to 110
    t = np.arange(-10,110,0.01)

    # Make a dataframe with one column as time, and another column called 'sinusoid' defined as sin(2*pi*t)
    # The sinusoid column will have a period of 1
    df = pd.DataFrame({
        'time': t,
        'sinusoid': np.sin(2*np.pi*t)
    })
    df_copy = df.copy(deep=True)

    # Make an event triggered response 
    etr = event_triggered_response(
        data = df,
        t = 'timestamps',
        y = 'sinusoid',
        event_times = np.arange(100),
        t_before = 1,
        t_after = 1,
        output_sampling_rate = 100,
    )

    # Assert that the average value of the agrees with expectations
    assert np.isclose(etr.query('time == 0')['sinusoid'].mean(), 0, rtol=0.01)
    assert np.isclose(etr.query('time == 0.25')['sinusoid'].mean(), 1, rtol=0.01)
    assert np.isclose(etr.query('time == 0.5')['sinusoid'].mean(), 0, rtol=0.01)
    assert np.isclose(etr.query('time == 0.75')['sinusoid'].mean(), -1, rtol=0.01)
    assert np.isclose(etr.query('time == 1')['sinusoid'].mean(), 0, rtol=0.01)

    # Assert that the dataframe is unchanged
    pd.testing.assert_frame_equal(df, df_copy)

def test_index_of_nearest_value():
    # create two timestamps series, of data and of events, using different sampling
    data_timestamps = np.arange(0, 100, 0.011)
    event_timestamps = np.arange(5, 95, 0.31)

    # get aligned indices
    event_aligned_ind = index_of_nearest_value(data_timestamps, event_timestamps)

    # assert length of the array
    assert len(event_aligned_ind) == 291
    # assert at least one index
    assert event_aligned_ind[15] == 877


def test_slice_inds_and_offsets():
    # create two timestamps series, of data and of events, using different sampling
    data_timestamps = np.arange(0, 100, 0.011)
    event_timestamps = np.arange(5, 95, 0.31)
    time_window = [-0.5, 1.5]

    # get event indices and offsets
    event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(data_timestamps=data_timestamps,
                                                                                             event_timestamps=event_timestamps,
                                                                                             time_window=time_window)
    # assert length of arrays
    assert len(event_indices) == 291
    assert len(trace_timebase) == 181
    # assert indices's values
    assert start_ind_offset == -45
    assert end_ind_offset == 136
    assert event_aligned_ind[15] == 877
    assert np.isclose(trace_timebase[15], -0.33, rtol=0.01)





