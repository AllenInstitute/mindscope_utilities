import pandas as pd
import numpy as np


def get_time_array(t_start, t_end, sampling_rate=None, step_size=None, include_endpoint=True):  # NOQA E501
    '''
    A function to get a time array between two specified timepoints at a defined sampling rate  # NOQA E501
    Deals with possibility of time range not being evenly divisible by desired sampling rate  # NOQA E501
    Uses np.linspace instead of np.arange given decimal precision issues with np.arange (see np.arange documentation for details)  # NOQA E501

    Parameters:
    -----------
    t_start : float
        start time for array
    t_end : float
        end time for array
    sampling_rate : float
        desired sampling of array
        Note: user must specify either sampling_rate or step_size, not both
    step_size : float
        desired step size of array
        Note: user must specify either sampling_rate or step_size, not both
    include_endpoint : Boolean
        Passed to np.linspace to calculate relative time
        If True, stop is the last sample. Otherwise, it is not included.
            Default is True

    Returns:
    --------
    numpy.array
        an array of timepoints at the desired sampling rate

    Examples:
    ---------
    get a time array exclusive of the endpoint
    >>> t_array = get_time_array(
        t_start=-1, 
        t_end=1, 
        step_size=0.5,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5])


    get a time array inclusive of the endpoint
    >>> t_array = get_time_array(
        t_start=-1, 
        t_end=1, 
        step_size=0.5,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5, 1.0])


    get a time array where the range can't be evenly divided by the desired step_size
    in this case, the time array includes the last timepoint before the desired endpoint
    >>> t_array = get_time_array(
        t_start=-1, 
        t_end=0.75, 
        step_size=0.5,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5])


    Instead of passing the step_size, we can pass the sampling rate
    >>> t_array = get_time_array(
        t_start=-1, 
        t_end=1, 
        sampling_rate=2,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5])
    '''
    assert sampling_rate is not None or step_size is not None, 'must specify either sampling_rate or step_size'  # NOQA E501
    assert sampling_rate is None or step_size is None, 'cannot specify both sampling_rate and step_size'  # NOQA E501

    # value as a linearly spaced time array
    if not step_size:
        step_size = 1/sampling_rate
    # define a time array
    n_steps = (t_end - t_start) / step_size
    if n_steps != int(n_steps):
        # if the number of steps isn't an int, that means it isn't possible
        # to end on the desired t_after using the defined sampling rate
        # we need to round down and include the endpoint
        n_steps = int(n_steps)
        t_end_adjusted = t_start + n_steps*step_size
        include_endpoint = True
    else:
        t_end_adjusted = t_end

    if include_endpoint:
        # add an extra step if including endpoint
        n_steps += 1

    t_array = np.linspace(
        t_start,
        t_end_adjusted,
        int(n_steps),
        endpoint=include_endpoint
    )

    return t_array


def event_triggered_response(data, t, y, event_times, t_before=0, t_after=1, output_sampling_rate=10, include_endpoint=True, output_format='tidy'):  # NOQA E501
    '''
    Slices a timeseries relative to a given set of event times
    to build an event-triggered response.

    For example, If we have data such as a measurement of neural activity
    over time and specific events in time that we want to align
    the neural activity to, this function will extract segments of the neural
    timeseries in a specified time window around each event.

    The times of the events need not align with the measured
    times of the neural data.
    Relative times will be calculated by linear interpolation.

    Parameters:
    -----------
    data: Pandas.DataFrame
        Input dataframe in tidy format
        Each row should be one observation
        Must contains columns representing `t` and `y` (see below)
    t : string
        Name of column in data to use as time data
    y : string
        Name of column to use as y data
    event_times: list or array of floats
        Times of events of interest.
        Values in column specified by `y` will be sliced and interpolated
            relative to these times
    t_before : float
        time before each of event of interest to include in each slice
            (in same units as `t` column)
    t_after : float
        time after each event of interest to include in each slice
            (in same units as `t` column)
    output_sampling_rate : float
        desired sampling of output
            (input data will be interpolated to this sampling rate)
    include_endpoint : Boolean
        Passed to np.linspace to calculate relative time
        If True, stop is the last sample. Otherwise, it is not included.
            Default is True
    output_format : string
        'wide' or 'tidy' (default = 'tidy')
        if 'tidy'
            One column representing time
            One column representing event_number
            One column representing event_time
            One row per observation (# rows = len(time) x len(event_times))
        if 'wide', output format will be:
            time as indices
            One row per interpolated timepoint
            One column per event,
                with column names titled event_{EVENT NUMBER}_t={EVENT TIME}

    Returns:
    --------
    Pandas.DataFrame
        See description in `output_format` section above

    Examples:
    ---------
    An example use case, recover a sinousoid from noise:

    First, define a time vector
    >>> t = np.arange(-10,110,0.001)

    Now build a dataframe with one column for time,
    and another column that is a noise-corrupted sinuosoid with period of 1
    >>> data = pd.DataFrame({
            'time': t,
            'noisy_sinusoid': np.sin(2*np.pi*t) + np.random.randn(len(t))*3
        })

    Now use the event_triggered_response function to get a tidy
    dataframe of the signal around every event

    Events will simply be generated as every 1 second interval
    starting at 0, since our period here is 1
    >>> etr = event_triggered_response(
            data,
            x = 'time',
            y = 'noisy_sinusoid',
            event_times = np.arange(100),
            t_before = 1,
            t_after = 1,
            output_sampling_rate = 100
        )
    Then use seaborn to view the result
    We're able to recover the sinusoid through averaging
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> fig, ax = plt.subplots()
    >>> sns.lineplot(
            data = etr,
            x='time',
            y='noisy_sinusoid',
            ax=ax
        )
    '''
    # set up a dictionary with key 'time' and
    t_array = get_time_array(
        t_start=-1*t_before,
        t_end=t_after,
        sampling_rate=output_sampling_rate,
        include_endpoint=include_endpoint,
    )
    data_dict = {'time': t_array}

    # iterate over all event times
    data_time_indexed = data.set_index(t, inplace=False)

    for event_number, event_time in enumerate(np.array(event_times)):

        # get a slice of the input data surrounding each event time
        data_slice = data_time_indexed[y].loc[event_time - t_before: event_time + t_after]  # noqa: E501

        # update our dictionary to have a new key defined as
        # 'event_{EVENT NUMBER}_t={EVENT TIME}' and
        # a value that includes an array that represents the
        # sliced data around the current event, interpolated
        # on the linearly spaced time array
        data_dict.update({
            'event_{}_t={}'.format(event_number, event_time): np.interp(
                data_dict['time'],
                data_slice.index - event_time,
                data_slice.values
            )
        })

    # define a wide dataframe as a dataframe of the above compiled dictionary
    wide_etr = pd.DataFrame(data_dict)
    if output_format == 'wide':
        # return the wide dataframe if output_format is 'wide'
        return wide_etr.set_index('time')
    elif output_format == 'tidy':
        # if output format is 'tidy',
        # transform the wide dataframe to tidy format
        # first, melt the dataframe with the 'id_vars' column as "time"
        tidy_etr = wide_etr.melt(id_vars='time')

        # add an "event_number" column that contains the event number
        tidy_etr['event_number'] = tidy_etr['variable'].map(
            lambda s: s.split('event_')[1].split('_')[0]
        ).astype(int)

        # add an "event_time" column that contains the event time ()
        tidy_etr['event_time'] = tidy_etr['variable'].map(
            lambda s: s.split('t=')[1]
        ).astype(float)

        # drop the "variable" column, rename the "value" column
        tidy_etr = (
            tidy_etr
            .drop(columns=['variable'])
            .rename(columns={'value': y})
        )
        # return the tidy event triggered responses
        return tidy_etr
