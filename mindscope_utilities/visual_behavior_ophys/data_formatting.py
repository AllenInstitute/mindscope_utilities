import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
import mindscope_utilities

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


def get_event_timestamps(stimulus_presentations_df, event_type='all', onset='start_time'):
    '''
    Gets timestamps of events of interest from the stimulus_resentations df.

    Parameters:
    ___________
    stimulus_presentations_df: Pandas.DataFrame
        Output of stimulus_presentations with stimulus trial metadata
    event_type: str
        Event of interest. Event_type can be any column in the stimulus_presentations_df,
        including 'omissions' or 'change'. Default is 'all', gets all trials
    onset: str
        optons: 'start_time' - onset of the stimulus, 'stop_time' - offset of the stimulus
        stimulus_presentations_df has a multiple timestamps to align data to. Default = 'start_time'.

    Returns:
        event_times: array
        event_ids: array
    --------
    '''
    if event_type == 'all':
        event_times = stimulus_presentations_df[onset]
        event_ids = stimulus_presentations_df.index.values
    elif event_type == 'images':
        event_times = stimulus_presentations_df[stimulus_presentations_df['omitted'] == False] [onset]
        event_ids = stimulus_presentations_df[stimulus_presentations_df['omitted'] == False].index.values
    elif event_type == 'omissions' or event_type == 'omitted':
        event_times = stimulus_presentations_df[stimulus_presentations_df['omitted'] == True][onset]
        event_ids = stimulus_presentations_df[stimulus_presentations_df['omitted'] == True].index.values
    elif event_type == 'changes' or event_type == 'is_change':
        event_times = stimulus_presentations_df[stimulus_presentations_df['is_change'] == True][onset]
        event_ids = stimulus_presentations_df[stimulus_presentations_df['is_change'] == True].index.values
    else:
        event_times = stimulus_presentations_df[stimulus_presentations_df[event_type] == True][onset]
        event_ids = stimulus_presentations_df[stimulus_presentations_df[event_type] == True].index.values

    return event_times, event_ids


def get_stimulus_response_xr(experiment, data_type='dff', event_type='all', time_window=[-3, 3],
                             interpolate=True, compute_means=True, **kargs):
    '''
    Parameters:
    ___________
    experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py  # noqa E501
    data_type: str
        neural data type to extract, options are: dff (default), events, filtered_events
    event_type: str
        event type to align to, which can be found in columns of experiment.stimulus_presentations df.
        options are: 'all' (default) - gets all stimulus trials
                     'images' - gets only image presentations (changes and not changes)
                     'omissions' - gets only trials with omitted stimuli
                     'changes' - get only trials of image changes
    time_window: array
        array of two int or floats indicating the time window on sliced data, default = [-3, 3]
    interpolate: bool
        type of alignment. If True (default) - interpolates neural data to align timestamps
        with stimulus presentations. If False - shifts data to the nearest timestamps (currently unavailable)
    compute_menas: bool

    kwargs: key, value mappings
        Other keyword arguments are passed down to mindscope_utilities.event_triggered_response(),
        for interpolation method such as output_sampling_rate and include_endpoint.

    Returns:
    __________
    stimulus_response_xr: xarray
        Xarray of aligned neural data with multiple dimentions: cell_specimen_id,
        'eventlocked_timestamps', and 'trial_id'

    '''

    # load neural data
    neural_data = build_tidy_cell_df(experiment)

    # load stimulus_presentations table
    stimulus_presentations_df = experiment.stimulus_presentations

    # get event times and event ids (original order in the stimulus flow)
    event_times, event_ids = get_event_timestamps(stimulus_presentations_df, event_type)

    # all cell specimen ids in an experiment
    cell_ids = np.unique(neural_data['cell_specimen_id'].values)

    # collect aligned data
    sliced_dataout = []

    # align neural data using interpolation method
    for cell_id in tqdm(cell_ids):
        etr = mindscope_utilities.event_triggered_response(
            data=neural_data[neural_data['cell_specimen_id'] == cell_id],
            t='timestamps',
            y=data_type,
            event_times=event_times,
            t_start=time_window[0],
            t_end=time_window[1],
            output_format='wide',
            **kargs
        )

        # get timestamps array
        trace_timebase = etr.index.values

        # collect aligned data from all cell, all trials into one array
        sliced_dataout.append(etr.transpose().values)

    # convert to xarray
    sliced_dataout = np.array(sliced_dataout)
    stimulus_response_xr = xr.DataArray(
        data=sliced_dataout,
        dims=('cell_specimen_id', 'trial_id', 'eventlocked_timestamps'),
        coords={
            'eventlocked_timestamps': trace_timebase,
            'trial_id': event_ids,
            'cell_specimen_id': cell_ids
        }
    )

    if compute_means is True:
        stimulus_response_xr = compute_means_xr(stimulus_response_xr)

    return stimulus_response_xr

def compute_means_xr(stimulus_response_xr, time_window):
    '''

    :param stimulus_response_xr:
    :param time_window:
    :return:
    '''
    response_range = [0, time_window[1]]
    baseline_range = [time_window[0], 0]

    mean_response = stimulus_response_xr.loc[
        {'eventlocked_timestamps': slice(*response_range)}
    ].mean(['eventlocked_timestamps'])

    mean_baseline = stimulus_response_xr.loc[
        {'eventlocked_timestamps': slice(*baseline_range)}
    ].mean(['eventlocked_timestamps'])

    stimulus_response_xr = xr.Dataset({
        'eventlocked_traces': stimulus_response_xr,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
    })

    return stimulus_response_xr
