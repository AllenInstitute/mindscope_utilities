import numpy as np
import pandas as pd
from mindscope_utilities import event_triggered_response, get_time_array, index_of_nearest_value, slice_inds_and_offsets

def test_get_time_array_with_sampling_rate():
    '''
    tests the `get_time_array` function while passing a sampling_rate argument
    '''
    # this should give [-1, 1) with steps of 0.5, exclusive of the endpoint
    t_array = get_time_array(
        t_start=-1, 
        t_end=1, 
        sampling_rate=2,
        include_endpoint=False
    )
    assert (t_array == np.array([-1., -0.5,  0.,  0.5])).all()

    # this should give [-1, 1] with steps of 0.5, inclusive of the endpoint
    t_array = get_time_array(
        t_start=-1, 
        t_end=1, 
        sampling_rate=2,
        include_endpoint=True
    )
    assert (t_array == np.array([-1., -0.5,  0.,  0.5, 1.0])).all()

    # this should give [-1, 0.75) with steps of 0.5.
    # becuase the desired range (1.75) is not evenly divisible by the step size (0.5), the array should end before the desired endpoint
    t_array = get_time_array(
        t_start=-1, 
        t_end=0.75, 
        sampling_rate=2,
        include_endpoint=False
    )
    assert (t_array == np.array([-1., -0.5,  0.,  0.5])).all()

    # this should give [-1, 0.75) with steps of 0.5.
    # becuase the desired range (1.75) is not evenly divisible by the step size (0.5), the array should end before the desired endpoint
    t_array = get_time_array(
        t_start=-1, 
        t_end=0.75, 
        sampling_rate=2,
        include_endpoint=True
    )
    assert (t_array == np.array([-1., -0.5,  0.,  0.5])).all()


def test_get_time_array_with_step_size():
    '''
    tests the `get_time_array` function while passing a step_size argument
    '''
    # this should give [-1, 1) with steps of 0.5, exclusive of the endpoint
    t_array = get_time_array(
        t_start=-1, 
        t_end=1, 
        step_size=0.5,
        include_endpoint=False
    )
    assert (t_array == np.array([-1., -0.5,  0.,  0.5])).all()

    # this should give [-1, 1] with steps of 0.5, inclusive of the endpoint
    t_array = get_time_array(
        t_start=-1, 
        t_end=1, 
        step_size=0.5,
        include_endpoint=True
    )
    assert (t_array == np.array([-1., -0.5,  0.,  0.5, 1.0])).all()

    # this should give [-1, 0.75) with steps of 0.5.
    # becuase the desired range (1.75) is not evenly divisible by the step size (0.5), the array should end before the desired endpoint
    t_array = get_time_array(
        t_start=-1, 
        t_end=0.75, 
        step_size=0.5,
        include_endpoint=False
    )
    assert (t_array == np.array([-1., -0.5,  0.,  0.5])).all()

    # this should give [-1, 0.75) with steps of 0.5.
    # becuase the desired range (1.75) is not evenly divisible by the step size (0.5), 
    # the array should end before the desired endpoint
    t_array = get_time_array(
        t_start=-1, 
        t_end=0.75, 
        step_size=0.5,
        include_endpoint=True
    )
    assert (t_array == np.array([-1., -0.5,  0.,  0.5])).all()


def test_get_time_array_assertion_errors():
    '''
    tests that assertion errors are working correctly in `get_time_array`
    '''
    try:
        t_array = get_time_array(
            t_start=-1, 
            t_end=1, 
            sampling_rate=2,
            step_size=0.5,
            include_endpoint=False
        )
        assert False, 'it should not be possible to pass both sampling_rate and step_size'
    except AssertionError:
        # expect the AssertionError, so this test should pass
        pass

    try:
        t_array = get_time_array(
            t_start=-1, 
            t_end=1, 
            include_endpoint=False
        )
        assert False, 'must pass either a sampling_rate or step_size'
    except AssertionError:
        # expect the AssertionError, so this test should pass
        pass


def test_index_of_nearest_value():
    data_timestamps = np.arange(0, 1.2, 0.1)
    event_timestamps = np.array([0.21, 1.01, 1.0499, 1.05, 1.099])

    calculated_indices = index_of_nearest_value(data_timestamps, event_timestamps)

    expected_indices = np.array([2, 10, 10, 11, 11])

    assert np.all(calculated_indices == expected_indices)


def test_slice_inds_and_offsets():
    data_timestamps = np.arange(0, 6, 0.1)
    event_timestamps = [1.01, 2.05, 3.1, 4.2, 5.21]
    time_window = [-0.5, 0.5]
    event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(
        data_timestamps,
        event_timestamps,
        time_window,
        sampling_rate=None,
        include_endpoint=True
    )

    assert np.all(event_indices == np.array([10, 20, 31, 42, 52]))
    assert start_ind_offset == -5
    assert end_ind_offset == 6
    assert np.all(trace_timebase == np.array([-0.5, -0.4, -0.3, -0.2, -0.1,  0. ,  0.1,  0.2,  0.3,  0.4,  0.5]))


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
        t = 'time',
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