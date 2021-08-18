import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
import mindscope_utilities


def build_tidy_cell_df(ophys_experiment, exclude_invalid_rois=True):
    '''
    Builds a tidy dataframe describing activity for every cell in ophys_experiment.
    Tidy format is defined as one row per observation.
    Thus, the output dataframe will be n_cells x n_timetpoints long

    Parameters:
    -----------
    ophys_experiment : AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    exclude_invalid_rois : bool
        If True (default), only includes ROIs that are listed as `valid_roi = True` in the ophys_experiment.cell_specimen_table.
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
        cell_specimen_table = ophys_experiment.cell_specimen_table.query('valid_roi').reset_index()  # noqa E501
    else:
        cell_specimen_table = ophys_experiment.cell_specimen_table.reset_index()  # noqa E501

    # iterate over each individual cell
    for idx, row in cell_specimen_table.iterrows():
        cell_specimen_id = row['cell_specimen_id']

        # build a tidy dataframe for this cell
        cell_df = pd.DataFrame({
            'timestamps': ophys_experiment.ophys_timestamps,
            'dff': ophys_experiment.dff_traces.loc[cell_specimen_id]['dff'] if cell_specimen_id in ophys_experiment.dff_traces.index else [np.nan] * len(ophys_experiment.ophys_timestamps),  # noqa E501
            'events': ophys_experiment.events.loc[cell_specimen_id]['events'] if cell_specimen_id in ophys_experiment.events.index else [np.nan] * len(ophys_experiment.ophys_timestamps),  # noqa E501
            'filtered_events': ophys_experiment.events.loc[cell_specimen_id]['filtered_events'] if cell_specimen_id in ophys_experiment.events.index else [np.nan] * len(ophys_experiment.ophys_timestamps),  # noqa E501
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


def get_event_timestamps(
        stimulus_presentations_df,
        event_type='all',
        onset='start_time'):
    '''
    Gets timestamps of events of interest from the stimulus_resentations df.

    Parameters:
    ___________
    stimulus_presentations_df: Pandas.DataFrame
        Output of stimulus_presentations with stimulus trial metadata
    event_type: str
        Event of interest. Event_type can be any column in the stimulus_presentations_df,  # noqa E501
        including 'omissions' or 'change'. Default is 'all', gets all trials  # noqa E501
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
        event_times = stimulus_presentations_df[stimulus_presentations_df['omitted'] == False][onset]  # noqa E501
        event_ids = stimulus_presentations_df[stimulus_presentations_df['omitted'] == False].index.values  # noqa E501
    elif event_type == 'omissions' or event_type == 'omitted':
        event_times = stimulus_presentations_df[stimulus_presentations_df['omitted']][onset]  # noqa E501
        event_ids = stimulus_presentations_df[stimulus_presentations_df['omitted']].index.values  # noqa E501
    elif event_type == 'changes' or event_type == 'is_change':
        event_times = stimulus_presentations_df[stimulus_presentations_df['is_change']][onset]  # noqa E501
        event_ids = stimulus_presentations_df[stimulus_presentations_df['is_change']].index.values  # noqa E501
    else:
        event_times = stimulus_presentations_df[stimulus_presentations_df[event_type]][onset]  # noqa E501
        event_ids = stimulus_presentations_df[stimulus_presentations_df[event_type]].index.values  # noqa E501

    return event_times, event_ids


def get_stimulus_response_xr(ophys_experiment,
                             data_type='dff',
                             event_type='all',
                             time_window=[-3, 3],
                             interpolate=True,
                             compute_means=True,
                             compute_significance=False,
                             **kargs):
    '''
    Parameters:
    ___________
    ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    data_type: str
        neural data type to extract, options are: dff (default), events, filtered_events
    event_type: str
        event type to align to, which can be found in columns of ophys_experiment.stimulus_presentations df.
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
        'eventlocked_timestamps', and 'stimulus_presentations_id'

    '''

    # load neural data
    neural_data = build_tidy_cell_df(ophys_experiment)

    # load stimulus_presentations table
    stimulus_presentations_df = ophys_experiment.stimulus_presentations

    # get event times and event ids (original order in the stimulus flow)
    event_times, event_ids = get_event_timestamps(
        stimulus_presentations_df, event_type)

    # all cell specimen ids in an ophys_experiment
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
        dims=('cell_specimen_id', 'stimulus_presentations_id',
              'eventlocked_timestamps'),
        coords={
            'eventlocked_timestamps': trace_timebase,
            'stimulus_presentations_id': event_ids,
            'cell_specimen_id': cell_ids
        }
    )

    if compute_means is True:
        stimulus_response_xr = compute_means_xr(
            stimulus_response_xr, time_window=time_window)

    return stimulus_response_xr


def compute_means_xr(stimulus_response_xr, time_window):
    '''
    Computes means of responses and spontaneous (baseline) traces.
    Response by default starts at 0, while baseline
    trace by default ends at 0.

    Parameters:
    ___________
    stimulus_response_xr: xarray
        stimulus_response_xr from get_stimulus_response_xr
        with three main dimentions: cell_specimen_id,
        trail_id, and eventlocked_timestamps
    time_window: array
        time window in seconds, used for alignment arount events
        in get_stimulus_response_xr

    Returns:
    _________
        stimulus_response_xr with additional
        dimentions: mean_response and mean_baseline
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


def get_stimulus_response_df(ophys_experiment,
                             data_type='dff',
                             event_type='all',
                             time_window=[-3, 3],
                             interpolate=True,
                             compute_means=True,
                             compute_significance=False,
                             **kargs):
    '''
    Get stimulus aligned responses from one ophys_experiment.

    Parameters:
    ___________
    ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    data_type: str
        neural data type to extract, options are: dff (default), events, filtered_events
    event_type: str
        event type to align to, which can be found in columns of ophys_experiment.stimulus_presentations df.
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
        Default=True, computes mean response and spontaneous (baseline) values. Adds them as additional columns.
    compute_significance: bool
        Currently, not working. A placeholder for future addition of finding significant responses.

    kwargs: key, value mappings
        Other keyword arguments are passed down to mindscope_utilities.event_triggered_response(),
        for interpolation method such as output_sampling_rate and include_endpoint.

    Returns:
    ___________
    stimulus_response_df: Pandas.DataFrame


    '''

    stimulus_response_xr = get_stimulus_response_xr(
        ophys_experiment=ophys_experiment,
        data_type=data_type,
        event_type=event_type,
        time_window=time_window,
        interpolate=interpolate,
        compute_means=compute_means,
        compute_significance=compute_significance,
        **kargs)

    traces = stimulus_response_xr['eventlocked_traces']
    if compute_means is True:
        mean_response = stimulus_response_xr['mean_response']
        mean_baseline = stimulus_response_xr['mean_baseline']
        stacked_response = mean_response.stack(
            multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()  # noqa E501
        stacked_baseline = mean_baseline.stack(
            multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()  # noqa E501

    if compute_significance is True:
        p_vals_omission = stimulus_response_xr['p_value_omission']
        p_vals_stimulus = stimulus_response_xr['p_value_stimulus']
        p_vals_gray_screen = stimulus_response_xr['p_value_gray_screen']
        stacked_pval_omission = p_vals_omission.stack(
            multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()  # noqa E501
        stacked_pval_stimulus = p_vals_stimulus.stack(
            multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()  # noqa E501
        stacked_pval_gray_screen = p_vals_gray_screen.stack(
            multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()  # noqa E501

    stacked_traces = traces.stack(multi_index=(
        'stimulus_presentations_id', 'cell_specimen_id')).transpose()
    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['eventlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)

    if compute_means is False and compute_significance is False:
        stimulus_response_df = pd.DataFrame({
            'stimulus_presentations_id': stacked_traces.coords['stimulus_presentations_id'],  # noqa E501
            'cell_specimen_id': stacked_traces.coords['cell_specimen_id'],
            'trace': list(stacked_traces.data),
            'trace_timestamps': list(trace_timestamps),
        })
    elif compute_means is True and compute_significance is False:
        stimulus_response_df = pd.DataFrame({
            'stimulus_presentations_id': stacked_traces.coords['stimulus_presentations_id'],  # noqa E501
            'cell_specimen_id': stacked_traces.coords['cell_specimen_id'],
            'trace': list(stacked_traces.data),
            'trace_timestamps': list(trace_timestamps),
            'mean_response': stacked_response.data,
            'baseline_response': stacked_baseline.data,
        })
    elif compute_means is False and compute_significance is True:
        stimulus_response_df = pd.DataFrame({
            'stimulus_presentations_id': stacked_traces.coords['stimulus_presentations_id'],  # noqa E501
            'cell_specimen_id': stacked_traces.coords['cell_specimen_id'],
            'trace': list(stacked_traces.data),
            'trace_timestamps': list(trace_timestamps),
            'p_value_gray_screen': stacked_pval_gray_screen,
            'p_value_omission': stacked_pval_omission,
            'p_value_stimulus': stacked_pval_stimulus,
        })
    else:
        stimulus_response_df = pd.DataFrame({
            'stimulus_presentations_id': stacked_traces.coords['stimulus_presentations_id'],  # noqa E501
            'cell_specimen_id': stacked_traces.coords['cell_specimen_id'],
            'trace': list(stacked_traces.data),
            'trace_timestamps': list(trace_timestamps),
            'p_value_gray_screen': stacked_pval_gray_screen,
            'p_value_omission': stacked_pval_omission,
            'p_value_stimulus': stacked_pval_stimulus,
            'mean_response': stacked_response.data,
            'baseline_response': stacked_baseline.data
        })
    return stimulus_response_df


def add_rewards_to_stimulus_presentations(
    stimulus_presentations,
    rewards,
    time_window=[
        -3,
        3]):
    '''
    Append a column to stimulus_presentations which contains
    the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of
            stimulus presentations.
            Must contain: 'start_time'
        rewards (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        time_window (list with 2 elements): start and end of the range
            relative to the start of each stimulus
            to average the running speed.
    Returns:
        stimulus_presentations with a new column called "reward" that contains
        reward times that fell within the window relative to each stim time

    Example:
        # get visual behavior cache
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc  # noqa E501
        cache_dir = SOME_LOCAL_DIRECTORY
        cache = bpc.from_s3_cache(cache_dir=cache_dir)

        # load data for one experiment
        ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)

        # get necessary tables
        stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
        rewards = ophys_experiment.rewards.copy()

        # add rewards to stim presentations
        stimulus_presentations = add_rewards_to_stimulus_presentations(stimulus_presentations, rewards)  # noqa E501
    '''

    reward_times = rewards['timestamps'].values
    rewards_each_stim = stimulus_presentations.apply(
        lambda row: reward_times[
            ((reward_times > row["start_time"] +
              time_window[0]) & (
                reward_times < row["start_time"] +
                time_window[1]))],
        axis=1,
    )
    stimulus_presentations["rewards"] = rewards_each_stim
    return stimulus_presentations


def add_licks_to_stimulus_presentations(
    stimulus_presentations,
    licks,
    time_window=[
        -3,
        3]):
    '''
    Append a column to stimulus_presentations which
    contains the timestamps of licks that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): 
            dataframe of stimulus presentations.
            Must contain: 'start_time'
        licks (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        time_window (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.  # noqa E501
    Returns:
        stimulus_presentations with a new column called "licks" that contains
        lick times that fell within the window relative to each stim time


    Example:
        # get visual behavior cache
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc  # noqa E501
        cache_dir = SOME_LOCAL_DIRECTORY
        cache = bpc.from_s3_cache(cache_dir=cache_dir)

        # load data for one experiment
        ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)

        # get necessary tables
        stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
        licks = ophys_experiment.licks.copy()

        # add licks to stim presentations
        stimulus_presentations = add_licks_to_stimulus_presentations(stimulus_presentations, licks)
    '''

    lick_times = licks['timestamps'].values
    licks_each_stim = stimulus_presentations.apply(
        lambda row: lick_times[
            ((lick_times > row["start_time"] +
              time_window[0]) & (
                lick_times < row["start_time"] +
                time_window[1]))],
        axis=1,
    )
    stimulus_presentations["licks"] = licks_each_stim
    return stimulus_presentations


def get_trace_average(trace, timestamps, start_time, stop_time):
    """
    takes average value of a trace within a window
    designated by start_time and stop_time
    """
    values_this_range = trace[(
        (timestamps >= start_time) & (timestamps < stop_time))]
    return values_this_range.mean()


def add_mean_running_speed_to_stimulus_presentations(
        stimulus_presentations,
        running_speed,
        time_window=[-3, 3]):
    '''
    Append a column to stimulus_presentations which contains
    the mean running speed in a range relative to
    the stimulus start time.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of
            stimulus presentations.
            Must contain: 'start_time'
        running_speed (pd.DataFrame): dataframe of running speed.
            Must contain: 'speed', 'timestamps'
        time_window: array
            timestamps in seconds, relative to the start of each stimulus
            to average the running speed.
            default = [-3,3]
    Returns:
        stimulus_presentations with new column
        "mean_running_speed" containing the
        mean running speed within the specified window
        following each stimulus presentation.

    Example:
        # get visual behavior cache
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc  # noqa E501
        cache_dir = SOME_LOCAL_DIR
        cache = bpc.from_s3_cache(cache_dir=cache_dir)

        # load data for one experiment
        ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)

        # get necessary tables
        stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
        running_speed = ophys_experiment.running_speed.copy()

        # add running_speed to stim presentations
        stimulus_presentations = add_mean_running_speed_to_stimulus_presentations(stimulus_presentations, running_speed)  # noqa E501
    '''

    stim_running_speed = stimulus_presentations.apply(
        lambda row: get_trace_average(
            running_speed['speed'].values,
            running_speed['timestamps'].values,
            row["start_time"] + time_window[0],
            row["start_time"] + time_window[1]), axis=1,)
    stimulus_presentations["mean_running_speed"] = stim_running_speed
    return stimulus_presentations


def add_mean_pupil_area_to_stimulus_presentations(
    stimulus_presentations,
    eye_tracking,
    time_window=[
        -3,
        3]):
    '''
    Append a column to stimulus_presentations which contains
    the mean pupil area in a range relative to
    the stimulus start time.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.  # noqa E501
            Must contain: 'start_time'
        eye_tracking (pd.DataFrame): dataframe of eye tracking data.
            Must contain: 'pupil_area', 'timestamps'
        time_window (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the pupil area.
    Returns:
        stimulus_presentations table with new column "mean_pupil_area" with the
        mean pupil arae within the specified window
        following each stimulus presentation.

    Example:
        # get visual behavior cache
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc  # noqa E501
        cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_cache'
        cache = bpc.from_s3_cache(cache_dir=cache_dir)

        # load data for one experiment
        ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)

        # get necessary tables
        stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
        eye_tracking = ophys_experiment.eye_tracking.copy()

        # add pupil area to stim presentations
        stimulus_presentations = add_mean_pupil_area_to_stimulus_presentations(stimulus_presentations, eye_tracking)  # noqa E501
    '''
    stim_pupil_area = stimulus_presentations.apply(
        lambda row: get_trace_average(
            eye_tracking['pupil_area'].values,
            eye_tracking['timestamps'].values,
            row["start_time"] + time_window[0],
            row["start_time"] + time_window[1],
        ), axis=1,)
    stimulus_presentations["mean_pupil_area"] = stim_pupil_area
    return stimulus_presentations
