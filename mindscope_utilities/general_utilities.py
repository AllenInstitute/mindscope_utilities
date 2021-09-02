import pandas as pd
import numpy as np
from scipy.stats import norm


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
        step_size = 1 / sampling_rate
    # define a time array
    n_steps = (t_end - t_start) / step_size
    if n_steps != int(n_steps):
        # if the number of steps isn't an int, that means it isn't possible
        # to end on the desired t_after using the defined sampling rate
        # we need to round down and include the endpoint
        n_steps = int(n_steps)
        t_end_adjusted = t_start + n_steps * step_size
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


def slice_inds_and_offsets(data_timestamps, event_timestamps, time_window, sampling_rate=None, include_endpoint=False):  # NOQA E501
    '''
    Get nearest indices to event timestamps, plus ind offsets (start:stop)
    for slicing out a window around the event from the trace.
    Parameters:
    -----------
    data_timestamps : np.array
        Timestamps of the datatrace.
    event_timestamps : np.array
        Timestamps of events around which to slice windows.
    time_window : list
        [start_offset, end_offset] in seconds
    sampling_rate : float, optional, default=None
        Sampling rate of the datatrace.
        If left as None, samplng rate is inferred from data_timestamps.

    Returns:
    --------
    event_indices : np.array
        Indices of events from the timestamps provided.
    start_ind_offset : int
    end_ind_offset : int
    trace_timebase :  np.array
    '''
    if sampling_rate is None:
        sampling_rate = 1 / np.diff(data_timestamps).mean()

    event_indices = index_of_nearest_value(data_timestamps, event_timestamps)
    trace_len = (time_window[1] - time_window[0]) * sampling_rate
    start_ind_offset = int(time_window[0] * sampling_rate)
    end_ind_offset = int(start_ind_offset + trace_len) + int(include_endpoint)
    trace_timebase = np.arange(
        start_ind_offset, end_ind_offset) / sampling_rate

    return event_indices, start_ind_offset, end_ind_offset, trace_timebase


def index_of_nearest_value(data_timestamps, event_timestamps):
    '''
    The index of the nearest sample time for each event time.

    Parameters:
    -----------
    sample_timestamps : np.ndarray of floats
        sorted 1-d vector of data sample timestamps.
    event_timestamps : np.ndarray of floats
        1-d vector of event timestamps.

    Returns:
    --------
    event_aligned_ind : np.ndarray of int
        An array of nearest sample time index for each event times.
    '''
    insertion_ind = np.searchsorted(data_timestamps, event_timestamps)
    # is the value closer to data at insertion_ind or insertion_ind-1?
    ind_diff = data_timestamps[insertion_ind] - event_timestamps
    ind_minus_one_diff = np.abs(data_timestamps[np.clip(
        insertion_ind - 1, 0, np.inf).astype(int)] - event_timestamps)

    event_indices = insertion_ind - (ind_diff > ind_minus_one_diff).astype(int)
    return event_indices


def event_triggered_response(data, t, y, event_times, t_start=None, t_end=None, t_before=None, t_after=None, output_sampling_rate=None, include_endpoint=True, output_format='tidy', interpolate=True):  # NOQA E501
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
    event_times: Panda.Series, numpy array or list of floats
        Times of events of interest. If pd.Series, the original index and index name will be preserved in the output
        Values in column specified by `y` will be sliced and interpolated
            relative to these times
    t_start : float
        start time relative to each event for desired time window
        e.g.:   t_start = -1 would start the window 1 second before each
                t_start = 1 would start the window 1 second after each event
        Note: cannot pass both t_start and t_before
    t_before : float
        time before each of event of interest to include in each slice
        e.g.:   t_before = 1 would start the window 1 second before each event
                t_before = -1 would start the window 1 second after each event
        Note: cannot pass both t_start and t_before
    t_end : float
        end time relative to each event for desired time window
        e.g.:   t_end = 1 would end the window 1 second after each event
                t_end = -1 would end the window 1 second before each event
        Note: cannot pass both t_end and t_after
    t_after : float
        time after each event of interest to include in each slice
        e.g.:   t_after = 1 would start the window 1 second after each event
                t_after = -1 would start the window 1 second before each event
        Note: cannot pass both t_end and t_after
    output_sampling_rate : float
        Desired sampling of output.
        Input data will be interpolated to this sampling rate if interpolate = True (default). # NOQA E501
        If passing interpolate = False, the sampling rate of the input timeseries will # NOQA E501
        be used and output_sampling_rate should not be specified.
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
    interpolate : Boolean
        if True (default), interpolates each response onto a common timebase
        if False, shifts each response to align indices to a common timebase

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
            t_start = -1,
            t_end = 1,
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
    # ensure that non-conflicting time values are passed
    assert t_before is not None or t_start is not None, 'must pass either t_start or t_before'  # noqa: E501
    assert t_after is not None or t_end is not None, 'must pass either t_start or t_before'  # noqa: E501

    assert t_before is None or t_start is None, 'cannot pass both t_start and t_before'  # noqa: E501
    assert t_after is None or t_end is None, 'cannot pass both t_after and t_end'  # noqa: E501

    if interpolate is False:
        assert output_sampling_rate is None, 'if interpolation = False, the sampling rate of the input timeseries will be used. Do not specify output_sampling_rate'  # NOQA E501

    # assign time values to t_start and t_end
    if t_start is None:
        t_start = -1 * t_before
    if t_end is None:
        t_end = t_after

    # get original stimulus_presentation_ids, preserve the column for .join() method
    if type(event_times) is pd.Series:  # if pd.Series, preserve the name of index column
        original_index = event_times.index.values
        if type(event_times.index.name) is str:
            original_index_column_name = event_times.index.name
        else:  # if index column does not have a name, name is original_index
            event_times.index.name = 'original_index'
            original_index_column_name = event_times.index.name
    # is list or array, turn into pd.Series
    elif type(event_times) is list or type(event_times) is np.ndarray:
        event_times = pd.Series(data=event_times,
                                name='event_times'
                                )
        # name the index column "original_index"
        event_times.index.name = 'original_index'
        original_index_column_name = event_times.index.name
        original_index = event_times.index.values

    # ensure that t_end is greater than t_start
    assert t_end > t_start, 'must define t_end to be greater than t_start'

    if output_sampling_rate is None:
        # if sampling rate is None,
        # set it to be the mean sampling rate of the input data
        output_sampling_rate = 1 / np.diff(data[t]).mean()

    # if interpolate is set to True,
    # we will calculate a common timebase and
    # interpolate every response onto that timebase
    if interpolate:
        # set up a dictionary with key 'time' and
        t_array = get_time_array(
            t_start=t_start,
            t_end=t_end,
            sampling_rate=output_sampling_rate,
            include_endpoint=include_endpoint,
        )
        data_dict = {'time': t_array}

        # iterate over all event times
        data_reindexed = data.set_index(t, inplace=False)

        for event_number, event_time in enumerate(np.array(event_times)):

            # get a slice of the input data surrounding each event time
            data_slice = data_reindexed[y].loc[event_time + t_start: event_time + t_end]  # noqa: E501

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

        # define a wide dataframe as a dataframe of the above compiled dictionary  # NOQA E501
        wide_etr = pd.DataFrame(data_dict)

    # if interpolate is False,
    # we will calculate a common timebase and
    # shift every response onto that timebase
    else:
        event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(  # NOQA E501
            np.array(data[t]),
            np.array(event_times),
            time_window=[t_start, t_end],
            sampling_rate=None,
            include_endpoint=True
        )
        all_inds = event_indices + \
            np.arange(start_ind_offset, end_ind_offset)[:, None]
        wide_etr = pd.DataFrame(
            data[y].values.T[all_inds],
            index=trace_timebase,
            columns=['event_{}_t={}'.format(event_index, event_time) for event_index, event_time in enumerate(event_times)]  # NOQA E501
        ).rename_axis(index='time').reset_index()

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

        tidy_etr[original_index_column_name] = tidy_etr['event_number'].apply(
            lambda row: original_index[row])

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


def dprime(hit_rate=None, fa_rate=None, go_trials=None, catch_trials=None, limits=False):
    '''
    calculates the d-prime for a given hit rate and false alarm rate

    https://en.wikipedia.org/wiki/Sensitivity_index

    Parameters
    ----------
    hit_rate : float or vector of floats
        rate of hits in the True class
    fa_rate : float or vector of floats
        rate of false alarms in the False class
    go_trials: vector of booleans
        responses on all go trials (hit = True, miss = False)
    catch_trials: vector of booleans
        responses on all catch trials (false alarm = True, correct reject = False)
    limits : boolean or tuple, optional
        limits on extreme values, which can cause d' to overestimate on low trial counts.
        False (default) results in limits of (0.01,0.99) to avoid infinite calculations
        True results in limits being calculated based on trial count (only applicable if go_trials and catch_trials are passed)
        (limits[0], limits[1]) results in specified limits being applied

    Note: user must pass EITHER hit_rate and fa_rate OR go_trials and catch trials

    Returns
    -------
    d_prime

    Examples
    --------
    With hit and false alarm rates of 0 and 1, if we pass in limits of (0, 1) we are
    allowing the raw probabilities to be used in the dprime calculation.
    This will result in an infinite dprime:

    >> dprime(hit_rate = 1.0, fa_rate = 0.0, limits = (0, 1))
    np.inf

    If we do not pass limits, the default limits of 0.01 and 0.99 will be used
    which will convert the hit rate of 1 to 0.99 and the false alarm rate of 0 to 0.01.
    This will prevent the d' calcluation from being infinite:

    >>> dprime(hit_rate = 1.0, fa_rate = 0.0)
    4.6526957480816815

    If the hit and false alarm rates are already within the limits, the limits don't apply
    >>> dprime(hit_rate = 0.6, fa_rate = 0.4)
    0.5066942062715994

    Alternately, instead of passing in pre-computed hit and false alarm rates,
    we can pass in a vector of results on go-trials and catch-trials.
    Then, if we call `limits = True`, the limits will be calculated based
    on the number of trials in both the go and catch trial vectors
    using the `trial_number_limit` function.

    For example, for low trial counts, even perfect performance (hit rate = 1, false alarm rate = 0)
    leads to a lower estimate of d', given that we have low confidence in the hit and false alarm rates;

    >>> dprime(
            go_trials = [1, 1, 1],
            catch_trials = [0, 0],
            limits = True
            )
    1.6419113162977828

    At the limit, if we have only one sample of both trial types, the `trial_number_limit`
    pushes our estimated response probability to 0.5 for both the hit and false alarm rates,
    giving us a d' value of 0:

    >>> dprime(
            go_trials = [1],
            catch_trials = [0],
            limits = True
            )
    0.0

    And with higher trial counts, the `trial_number_limit` allows the hit and false alarm
    rates to get asymptotically closer to 0 and 1, leading to higher values of d'.
    For example, for 10 trials of each type:

    >>> dprime(
            go_trials = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            catch_trials = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            limits = True
        )
    3.289707253902945

    Or, if we had 20 hit trials and 10 false alarm trials:

    >>> dprime(
            go_trials = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            catch_trials = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            limits = True
        )
    3.604817611491527

    Note also that Boolean vectors can be passed:

    >>> dprime(
            go_trials = [True, False, True, True, False],
            catch_trials = [False, True, False, False, False],
            limits = True
        )
    1.094968336708714
    '''

    assert hit_rate is not None or go_trials is not None, 'must either a specify `hit_rate` or pass a boolean vector of `go_trials`'
    assert fa_rate is not None or catch_trials is not None, 'must either a specify `fa_rate` or pass a boolean vector of `catch_trials`'

    assert hit_rate is None or go_trials is None, 'do not pass both `hit_rate` and a boolean vector of `go_trials`'
    assert fa_rate is None or catch_trials is None, 'do not pass both `fa_rate` and a boolean vector of `catch_trials`'

    assert not (hit_rate is not None and limits is True), 'limits can only be calculated if a go_trials vector is passed, not a hit_rate'
    assert not (fa_rate is not None and limits is True), 'limits can only be calculated if a catch_trials vector is passed, not a fa_rate'

    # calculate hit and fa rates as mean of boolean vectors
    if hit_rate is None:
        hit_rate = np.mean(go_trials)
    if fa_rate is None:
        fa_rate = np.mean(catch_trials)

    Z = norm.ppf

    if limits is False:
        # if limits are False, apply default
        limits = (0.01, 0.99)
    elif limits is True:
        # clip the hit and fa rate based on trial count
        hit_rate = response_probabilities_trial_number_limit(
            hit_rate, len(go_trials))
        fa_rate = response_probabilities_trial_number_limit(
            fa_rate, len(catch_trials))

    if limits is not True:
        # Limit values in order to avoid d' infinity
        hit_rate = np.clip(hit_rate, limits[0], limits[1])
        fa_rate = np.clip(fa_rate, limits[0], limits[1])

    # keep track of nan locations
    hit_rate = pd.Series(hit_rate)
    fa_rate = pd.Series(fa_rate)
    hit_rate_nan_locs = list(hit_rate[pd.isnull(hit_rate)].index)
    fa_rate_nan_locs = list(fa_rate[pd.isnull(fa_rate)].index)

    # fill nans with 0.0 to avoid warning about nans
    d_prime = Z(hit_rate.fillna(0)) - Z(fa_rate.fillna(0))

    # for every location in hit_rate and fa_rate with a nan, fill d_prime with a nan
    for nan_locs in [hit_rate_nan_locs, fa_rate_nan_locs]:
        d_prime[nan_locs] = np.nan

    if len(d_prime) == 1:
        # if the result is a 1-length vector, return as a scalar
        return d_prime[0]
    else:
        return d_prime


def response_probabilities_trial_number_limit(P, N):
    '''
    Calculates limits on response probability estimate based on trial count.
    An important point to note about d' is that the metric will be infinite with
    perfect performance, given that Z(0) = -infinity and Z(1) = infinity.
    Low trial counts exacerbate this issue. For example, with only one sample of a go-trial,
    the hit rate will either be 1 or 0. Macmillan and Creelman [1] offer a correction on the hit and false
    alarm rates to avoid infinite values whereby the response probabilities (P) are bounded by
    functions of the trial count, N:

        1/(2N) < P < 1 - 1/(2N)

    Thus, for the example of just a single trial, the trial-corrected hit rate would be 0.5.
    Or after only two trials, the hit rate could take on the values of 0.25, 0.5, or 0.75.

    [1] Macmillan, Neil A., and C. Douglas Creelman. Detection theory: A user's guide. Psychology press, 2004.
    Parameters
    ----------
    P : float
        response probability, bounded by 0 and 1
    N : int
        number of trials used in the response probability calculation

    Returns
    -------
    P_corrected : float
        The response probability after adjusting for the trial count

    Examples
    --------
    Passing in a response probability of 1 and trial count of 10 will lead to a reduced
    estimate of the response probability:

    >>> trial_number_limit(1, 10)
    0.95

    A higher trial count increases the bounds on the estimate:

    >>> trial_number_limit(1, 50)
    0.99

    At the limit, the bounds are 0 and 1:

    >>> trial_number_limit(1, np.inf)
    1.0

    The bounds apply for estimates near 0 also:

    >>> trial_number_limit(0, 1)
    0.5

    >>> trial_number_limit(0, 2)
    0.25

    >>> trial_number_limit(0, 3)
    0.16666666666666666

    Note that passing a probality that is inside of the bounds
    results in no change to the passed probability:

    >>> trial_number_limit(0.25, 3)
    0.25
    '''
    if N == 0:
        return np.nan
    if not pd.isnull(P):
        P = np.max((P, 1. / (2 * N)))
        P = np.min((P, 1 - 1. / (2 * N)))
    return P
