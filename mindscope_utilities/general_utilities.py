import pandas as pd
import numpy as np


def event_triggered_response(data, t, y, event_times, t_before=1, t_after=1, output_sampling_rate=10, include_endpoint=True, output_format='tidy'):  # NOQA E501
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
    # value as a linearly spaced time array
    step_size = 1/output_sampling_rate
    data_dict = {
        'time': np.linspace(
            -t_before,
            t_after,
            int((t_before + t_after) / step_size + int(include_endpoint)),
            endpoint=include_endpoint
        )
    }

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

def index_of_nearest_value(sample_times, event_times):
    '''
    The index of the nearest sample time for each event time.
    Args:
        sample_times (np.ndarray of float): sorted 1-d vector of sample timestamps
        event_times (np.ndarray of float): 1-d vector of event timestamps
    Returns
        (np.ndarray of int) nearest sample time index for each event time
    '''
    insertion_ind = np.searchsorted(sample_times, event_times)
    # is the value closer to data at insertion_ind or insertion_ind-1?
    ind_diff = sample_times[insertion_ind] - event_times
    ind_minus_one_diff = np.abs(sample_times[np.clip(insertion_ind - 1, 0, np.inf).astype(int)] - event_times)
    return insertion_ind - (ind_diff > ind_minus_one_diff).astype(int)

def slice_inds_and_offsets(ophys_times, event_times, window_around_timepoint_seconds, frame_rate=None):
    '''
    Get nearest indices to event times, plus ind offsets for slicing out a window around the event from the trace.
    Args:
        ophys_times (np.array): timestamps of ophys frames
        event_times (np.array): timestamps of events around which to slice windows
        window_around_timepoint_seconds (list): [start_offset, end_offset] for window
        frame_rate (float): we shouldn't need this. leave none to infer from the ophys timestamps
    '''
    if frame_rate is None:
        frame_rate = 1 / np.diff(ophys_times).mean()
    event_indices = index_of_nearest_value(ophys_times, event_times)
    trace_len = (window_around_timepoint_seconds[1] - window_around_timepoint_seconds[0]) * frame_rate
    start_ind_offset = int(window_around_timepoint_seconds[0] * frame_rate)
    end_ind_offset = int(start_ind_offset + trace_len)
    trace_timebase = np.arange(start_ind_offset, end_ind_offset) / frame_rate
    return event_indices, start_ind_offset, end_ind_offset, trace_timebase

def eventlocked_traces(response_traces, event_indices, start_ind_offset, end_ind_offset):
    '''
    Extract trace for each cell, for each event-relative window.
    Args:
        dff_traces (np.ndarray): shape (nSamples, nCells) with dff traces for each cell
        event_indices (np.ndarray): 1-d array of shape (nEvents) with closest sample ind for each event
        start_ind_offset (int): Where to start the window relative to each event ind
        end_ind_offset (int): Where to end the window relative to each event ind
    Returns:
        sliced_dataout (np.ndarray): shape (nSamples, nEvents, nCells)
    '''
    all_inds = event_indices + np.arange(start_ind_offset, end_ind_offset)[:, None]
    sliced_dataout = response_traces.T[all_inds]
    return sliced_dataout

def get_response_xr(session, traces, timestamps, event_times, event_ids, trace_ids,
                    window_around_timepoint_seconds=[-0.25, 0.75], frame_rate=None):
    event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(
        ophys_times=timestamps,
        event_times=event_times,
        window_around_timepoint_seconds=window_around_timepoint_seconds,
        frame_rate=frame_rate
    )
    sliced_dataout = eventlocked_traces(traces, event_indices, start_ind_offset, end_ind_offset)

    eventlocked_traces_xr = xr.DataArray(
        data=sliced_dataout,
        dims=("eventlocked_timestamps", "trial_id", "trace_id"),
        coords={
            "eventlocked_timestamps": trace_timebase,
            "trial_id": event_ids,
            "trace_id": trace_ids
        }
    )

    response_range = [0, 0.75] # in seconds
    baseline_range = [-.25, 0] # in seconds

    mean_response = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*response_range)}
    ].mean(['eventlocked_timestamps'])

    mean_baseline = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*baseline_range)}
    ].mean(['eventlocked_timestamps'])

    result = xr.Dataset({
        'eventlocked_traces': eventlocked_traces_xr,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
    })

    return result


def get_response_df(response_xr):
    '''
    Smash things into df format if you want.
    '''
    traces = response_xr['eventlocked_traces']
    mean_response = response_xr['mean_response']
    mean_baseline = response_xr['mean_baseline']
    stacked_traces = traces.stack(multi_index=('trial_id', 'trace_id')).transpose()
    stacked_response = mean_response.stack(multi_index=('trial_id', 'trace_id')).transpose()
    stacked_baseline = mean_baseline.stack(multi_index=('trial_id', 'trace_id')).transpose()

    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['eventlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)

    df = pd.DataFrame({
        'trial_id': stacked_traces.coords['trial_id'],
        'trace_id': stacked_traces.coords['trace_id'],
        'trace': list(stacked_traces.data),
        'trace_timestamps': list(trace_timestamps),
        'mean_response': stacked_response.data,
        'baseline_response': stacked_baseline.data,
    })
    return df
