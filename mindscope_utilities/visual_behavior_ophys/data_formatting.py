import pandas as pd
import numpy as np
import xarray as xr
from mindscope_utilities import slice_inds_and_offsets
from mindscope_utilities import get_eventlocked_traces

def build_tidy_cell_df(experiment, exclude_invalid_rois=True):
    '''
    Builds a tidy dataframe describing activity for every cell in experiment.
    Tidy format is defined as one row per observation.
    Thus, the output dataframe will be n_cells x n_timetpoints long

    Parameters:
    -----------
    experiment : AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py  # noqa E501
    exclude_invalid_rois : bool
        If True (default), only includes ROIs that are listed as `valid_roi = True` in the experiment.cell_specimen_table.
        If False, include all ROIs.
        Note that invalid ROIs are only exposed for internal AllenInstitute users, so passing `False` will not change behavior for external users

    Returns:
    --------
    Pandas.DataFrame
        Tidy Format (one observation per row) with the following columns:
            * timestamps (float) : the ophys timestamps
            * cell_roi_id (int) : the cell roi id
            * cell_specimen_id (int) : the cell specimen id
            * dff (float) : measured deltaF/F for every timestep
            * events (float) : extracted events for every timestep
            * filtered events (float) : filtered (convolved with half-gaussian) events for every timestep
    '''

    # make an empty list to populate with dataframes for each cell
    list_of_cell_dfs = []

    # query on valid_roi if exclude_invalid_rois == True
    if exclude_invalid_rois:
        cell_specimen_table = experiment.cell_specimen_table.query('valid_roi').reset_index()  # noqa E501
    else:
        cell_specimen_table = experiment.cell_specimen_table.reset_index()

    # iterate over each individual cell
    for idx, row in cell_specimen_table.iterrows():
        cell_specimen_id = row['cell_specimen_id']

        # build a tidy dataframe for this cell
        cell_df = pd.DataFrame({
            'timestamps': experiment.ophys_timestamps,
            'dff': experiment.dff_traces.loc[cell_specimen_id]['dff'] if cell_specimen_id in experiment.dff_traces.index else [np.nan] * len(experiment.ophys_timestamps),  # noqa E501
            'events': experiment.events.loc[cell_specimen_id]['events'] if cell_specimen_id in experiment.events.index else [np.nan] * len(experiment.ophys_timestamps),  # noqa E501
            'filtered_events': experiment.events.loc[cell_specimen_id]['filtered_events'] if cell_specimen_id in experiment.events.index else [np.nan] * len(experiment.ophys_timestamps),  # noqa E501
        })

        # Make the cell_roi_id and cell_specimen_id columns categorical.
        # This will reduce memory useage since the columns
        # consist of many repeated values.
        for cell_id in ['cell_roi_id', 'cell_specimen_id']:
            cell_df[cell_id] = np.int32(row[cell_id])
            cell_df[cell_id] = pd.Categorical(
                cell_df[cell_id],
                categories=cell_specimen_table[cell_id].unique()
            )

        # append the dataframe for this cell to the list of cell dataframes
        list_of_cell_dfs.append(cell_df)

    # concatenate all dataframes in the list
    tidy_df = pd.concat(list_of_cell_dfs)

    # return the tidy dataframe
    return tidy_df

def get_response_xr(dataset, event_type = "all", use_events = True, filter_events = False, time_window = [-0.25, 0.75], sampling_rate = None):
    '''
    Creates xr array of sliced aligned to events response traces
    from the dataset object (one experiment)

    One future change for this function can be to add an option
    of specifying either event types or just providing an array of event timestamps.
    Parameters:
    ____________
    dataset : obj
        Dataset object from one experiment, output of get_behavior_ophys_experiment() method.
        allensdk.brain_observatory.behavior.behavior_ophys_experiment.BehaviorOphysExperiment.
    event_type : str
        Types of events to look for, defaults to all stimulus presentations.
        Options: "all", "omission", "change".
        Other options could be added such as "prefered_stimulus" or stimulus_id
    use_events : Boolean
        If True, uses extracted events from datatraces, otherwise uses dff traces. Default = True.
    filter_events: Boolean
        If True, uses events that had been smoothed using a gaussian filter. Default = False.
    time_window: np.array
        Time window to slice out of the response traces in seconds, default = [-0.25, 0.75]
    sampling_rate : int or float
        Sampling rate of the data trace, default = None, uses datatrace timestamps to infer
        sampling rate.

    Returns:
    ____________
    responses_xr :  xrarray
        Xr array of aligned responses
    '''
    ## Get data ##
    stim_df = dataset.stimulus_presentations
    if event_type =='all':
        event_timestamps = stim_df.start_time.values
        event_ids = stim_df.index.values
    elif event_type == "change":
        event_timestamps = stim_df[stim_df['is_change'] == True].start_time.values
        event_ids = stim_df[stim_df['is_change'] == True].index.values
    elif event_type == "omission":
        event_timestamps = stim_df[stim_df['omitted'] == True].start_time.values
        event_ids = stim_df[stim_df['omitted'] == True].index.values
    else:
        raise NameError("Event type is not currently supported.")

    if filter_events:
        response_traces = dataset.events.filtered_events.values
        cell_specimen_ids = dataset.events.index.values
    elif use_events:
        response_traces =dataset.events.events.values
        cell_specimen_ids = dataset.events.index.values
    else:
        response_traces = dataset.dff_traces.dff.values
        cell_specimen_ids = dataset.dff_traces.index.values

    response_traces = np.vstack(response_traces)
    timestamps = dataset.ophys_timestamps


    ###############

    event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(
        data_timestamps=timestamps,
        event_timestamps=event_timestamps,
        time_window=time_window,
        sampling_rate=sampling_rate
    )
    sliced_dataout = get_eventlocked_traces(response_traces, event_indices, start_ind_offset, end_ind_offset)

    eventlocked_traces_xr = xr.DataArray(
        data=sliced_dataout,
        dims=("eventlocked_timestamps", "trial_id", "cell_specimen_id"),
        coords={
            "eventlocked_timestamps": trace_timebase,
            "trial_id": event_ids,
            "cell_specimen_id": cell_specimen_ids
        }
    )
    response_xr = xr.Dataset({
        'eventlocked_traces': eventlocked_traces_xr
    })
    return response_xr

def compute_trace_means(response_xr, response_window = [0, 0.75], baseline_window = [-0.25, 0]):
    '''
    Uses xrarray of aligned traces to compute mean response and mean baseline activity.

    Parameters:
    _____________
    response_xr: xr array
        Aligned responses, output of get_response_xr()
    response_window: np.array
        time window to use for response mean, in seconds, default = [0, 0.75].
    baseline_window: np.array
        time window to use for baseline mean, in seconds, default = [-0.25, 0].

    Returns:
    ______________
    response_xr : xr array
        Updated xr array of sliced responses with added means

    '''
    mean_response = response_xr.loc[
        {'eventlocked_timestamps': slice(*response_window)}
    ].mean(['eventlocked_timestamps'])

    mean_baseline = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*baseline_window)}
    ].mean(['eventlocked_timestamps'])

    response_xr = xr.Dataset({
        'eventlocked_traces': response_xr,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
    })

    return response_xr

def get_response_df(dataset, event_type = "all", use_events = True, filter_events = False, time_window = [-0.25, 0.75], sampling_rate = None):
    '''
    Convert response_xr to pandas Dataframe
    '''
    response_xr = get_response_xr(dataset, event_type=event_type, use_events=use_events, filter_events=filter_events,
                                  time_window=time_window, sampling_rate=sampling_rate)
    response_xr =

    traces = response_xr['eventlocked_traces']
    mean_response = response_xr['mean_response']
    mean_baseline = response_xr['mean_baseline']
    stacked_traces = traces.stack(multi_index=('trial_id', 'cell_specimen_ids')).transpose()
    stacked_response = mean_response.stack(multi_index=('trial_id', 'cell_specimen_id')).transpose()
    stacked_baseline = mean_baseline.stack(multi_index=('trial_id', 'cell_specimen_id')).transpose()

    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['eventlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)

    df = pd.DataFrame({
        'trial_id': stacked_traces.coords['trial_id'],
        'cell_specimen_id': stacked_traces.coords['cell_specimen_id'],
        'trace': list(stacked_traces.data),
        'trace_timestamps': list(trace_timestamps),
        'mean_response': stacked_response.data,
        'baseline_response': stacked_baseline.data,
    })
    return df
