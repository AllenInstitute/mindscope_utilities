import numpy as np
import pandas as pd
from mindscope_utilities import *


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
    assert (t_array == np.array([-1., -0.5, 0., 0.5])).all()

    # this should give [-1, 1] with steps of 0.5, inclusive of the endpoint
    t_array = get_time_array(
        t_start=-1,
        t_end=1,
        sampling_rate=2,
        include_endpoint=True
    )
    assert (t_array == np.array([-1., -0.5, 0., 0.5, 1.0])).all()

    # this should give [-1, 0.75) with steps of 0.5.
    # becuase the desired range (1.75) is not evenly divisible by the step size (0.5), the array should end before the desired endpoint
    t_array = get_time_array(
        t_start=-1,
        t_end=0.75,
        sampling_rate=2,
        include_endpoint=False
    )
    assert (t_array == np.array([-1., -0.5, 0., 0.5])).all()

    # this should give [-1, 0.75) with steps of 0.5.
    # becuase the desired range (1.75) is not evenly divisible by the step size (0.5), the array should end before the desired endpoint
    t_array = get_time_array(
        t_start=-1,
        t_end=0.75,
        sampling_rate=2,
        include_endpoint=True
    )
    assert (t_array == np.array([-1., -0.5, 0., 0.5])).all()


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
    assert (t_array == np.array([-1., -0.5, 0., 0.5])).all()

    # this should give [-1, 1] with steps of 0.5, inclusive of the endpoint
    t_array = get_time_array(
        t_start=-1,
        t_end=1,
        step_size=0.5,
        include_endpoint=True
    )
    assert (t_array == np.array([-1., -0.5, 0., 0.5, 1.0])).all()

    # this should give [-1, 0.75) with steps of 0.5.
    # becuase the desired range (1.75) is not evenly divisible by the step size (0.5), the array should end before the desired endpoint
    t_array = get_time_array(
        t_start=-1,
        t_end=0.75,
        step_size=0.5,
        include_endpoint=False
    )
    assert (t_array == np.array([-1., -0.5, 0., 0.5])).all()

    # this should give [-1, 0.75) with steps of 0.5.
    # becuase the desired range (1.75) is not evenly divisible by the step size (0.5),
    # the array should end before the desired endpoint
    t_array = get_time_array(
        t_start=-1,
        t_end=0.75,
        step_size=0.5,
        include_endpoint=True
    )
    assert (t_array == np.array([-1., -0.5, 0., 0.5])).all()


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
        t_array = get_time_array(  # NOQA
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
    stim_timestamps = np.array([0.21, 1.01, 1.0499, 1.05, 1.099])

    calculated_indices = index_of_nearest_value(
        data_timestamps, stim_timestamps)

    expected_indices = np.array([2, 10, 10, 11, 11])

    assert np.all(calculated_indices == expected_indices)


def test_slice_inds_and_offsets():
    data_timestamps = np.arange(0, 6, 0.1)
    stim_timestamps = [1.01, 2.05, 3.1, 4.2, 5.21]
    time_window = [-0.5, 0.5]
    stim_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(
        data_timestamps,
        stim_timestamps,
        time_window,
        sampling_rate=None,
        include_endpoint=True
    )

    assert np.all(stim_indices == np.array([10, 20, 31, 42, 52]))
    assert start_ind_offset == -5
    assert end_ind_offset == 6
    assert np.all(trace_timebase == np.array(
        [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]))


def test_stim_triggered_response():
    # make a time vector from -10 to 110
    t = np.arange(-10, 110, 0.01)

    # Make a dataframe with one column as time, and another column called 'sinusoid' defined as sin(2*pi*t)
    # The sinusoid column will have a period of 1
    df = pd.DataFrame({
        'time': t,
        'sinusoid': np.sin(2 * np.pi * t)
    })
    df_copy = df.copy(deep=True)

    # Make an stim triggered response
    etr = stim_triggered_response(
        data=df,
        t='time',
        y='sinusoid',
        stim_times=np.arange(100),
        t_before=1,
        t_after=1,
        output_sampling_rate=100,
    )

    # Assert that the average value of the agrees with expectations
    assert np.isclose(etr.query('time == 0')['sinusoid'].mean(), 0, rtol=0.01)
    assert np.isclose(etr.query('time == 0.25')[
                      'sinusoid'].mean(), 1, rtol=0.01)
    assert np.isclose(etr.query('time == 0.5')[
                      'sinusoid'].mean(), 0, rtol=0.01)
    assert np.isclose(etr.query('time == 0.75')[
                      'sinusoid'].mean(), -1, rtol=0.01)
    assert np.isclose(etr.query('time == 1')['sinusoid'].mean(), 0, rtol=0.01)

    # Assert that the dataframe is unchanged
    pd.testing.assert_frame_equal(df, df_copy)

    
def test_response_probabilities_trial_number_limit():
    assert response_probabilities_trial_number_limit(1, 5) == 0.9

    assert response_probabilities_trial_number_limit(1, 50) == 0.99

    assert response_probabilities_trial_number_limit(1, 100) == 0.995

    assert response_probabilities_trial_number_limit(0.5, 100) == 0.5

    assert response_probabilities_trial_number_limit(0, 100) == 0.005


def test_dprime():

    d_prime = dprime(1.0, 0.0)
    assert d_prime == 4.6526957480816815

    d_prime = dprime(1.0, 0.0, limits=False)
    assert d_prime == 4.6526957480816815

    d_prime = dprime(1.0, 0.0, limits=(0.01, 0.99))
    assert d_prime == 4.6526957480816815

    d_prime = dprime(1.0, 0.0, limits=(0.0, 1.0))
    assert d_prime == np.inf

    d_prime = dprime(
        go_trials=[1, 1, 1, 1, 1, 1, 1],
        catch_trials=[0, 0],
        limits=True
    )
    assert d_prime == 2.1397235428816046

    d_prime = dprime(
        go_trials=[1, 1, 1, 1, 1, 1, 1],
        catch_trials=[0, 0],
        limits=False
    )
    assert d_prime == 4.6526957480816815

    d_prime = dprime(
        go_trials=[0, 1, 0, 1],
        catch_trials=[0, 1],
        limits=False
    )
    assert d_prime == 0.0

