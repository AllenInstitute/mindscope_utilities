import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from warnings import warn
import mindscope_utilities
from allensdk.brain_observatory.behavior.trials_processing import calculate_reward_rate

repo_depr_warn_str = "The mindscope_utilities repo is deprecated. Please use the brain_observatory_utilities instead."
module_warn_str = "The {} module is deprecated. Please use {} in brain_observatory_utilities instead."

warn(repo_depr_warn_str, DeprecationWarning, stacklevel=2)
warn(module_warn_str.format("data_formatting", "datasets.behavior.data_formatting, or datasets.ophys.data_formatting"), DeprecationWarning, stacklevel=2)


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


def get_stim_timestamps(
        stimulus_presentation,
        stim_type='all',
        image_order=0,
        onset='start_time'):
    '''
    Gets timestamps of stimuli of interest from the stimulus_resentations df.

    Parameters:
    ___________
    stimulus_presentation: Pandas.DataFrame
        Output of stimulus_presentations with stimulus trial metadata
    stim_type: str
        stimulus of interest. stim_type can be any column in the stimulus_presentation,  # noqa E501
        including 'omissions' or 'change'. Default is 'all', gets all trials  # noqa E501
    image_order: int
        If stim_type has 'n' in it, image_order is the index of the image presentation  # noqa E501
        after change, omission, or both. Default is 0 (same as changes or omissions or both) # noqa E501
    onset: str
        optons: 'start_time' - onset of the stimulus, 'stop_time' - offset of the stimulus
        stimulus_presentationshas a multiple timestamps to align data to. Default = 'start_time'.

    Returns:
        stim_times: array
        stim_ids: array
    --------
    '''
    if 'n_after_change' not in stimulus_presentation.columns:
        stimulus_presentation = add_n_to_stimulus_presentations(stimulus_presentation)  # noqa E501

    if stim_type == 'all':
        stim_times = stimulus_presentation[onset]
        stim_ids = stimulus_presentation.index.values
    elif stim_type == 'images':
        stim_times = stimulus_presentation[stimulus_presentation['omitted'] == False][onset]  # noqa E501
        stim_ids = stimulus_presentation[stimulus_presentation['omitted'] == False].index.values  # noqa E501
    elif stim_type == 'omissions' or stim_type == 'omitted':
        stim_times = stimulus_presentation[stimulus_presentation['omitted']][onset]  # noqa E501
        stim_ids = stimulus_presentation[stimulus_presentation['omitted']].index.values  # noqa E501
    elif stim_type == 'changes' or stim_type == 'is_change':
        stim_times = stimulus_presentation[stimulus_presentation['is_change']][onset]  # noqa E501
        stim_ids = stimulus_presentation[stimulus_presentation['is_change']].index.values  # noqa E501
    elif stim_type == 'images-n-omissions':
        condition = (stimulus_presentation['n_after_omission']==image_order) & \
                    (stimulus_presentation['n_after_change'] > stimulus_presentation['n_after_omission'])  # noqa E501
        stim_times = stimulus_presentation[condition][onset]
        stim_ids = stimulus_presentation[condition].index.values
    elif stim_type == 'images-n-changes':
        condition = (stimulus_presentation['n_after_change']==image_order) & \
                    ((stimulus_presentation['n_after_omission'] > stimulus_presentation['n_after_change']) | # noqa E501  
                     (stimulus_presentation['n_after_omission'] == -1))  # for trials without omission
        stim_times = stimulus_presentation[condition][onset]
        stim_ids = stimulus_presentation[condition].index.values
    elif stim_type == 'images>n-omissions':
        condition = (stimulus_presentation['n_after_omission'] > image_order) & \
                    (stimulus_presentation['n_after_change'] > stimulus_presentation['n_after_omission'])  # noqa E501
        stim_times = stimulus_presentation[condition][onset]
        stim_ids = stimulus_presentation[condition].index.values
    elif stim_type == 'images>n-changes':
        condition = (stimulus_presentation['n_after_change'] > image_order) & \
                    ((stimulus_presentation['n_after_omission'] > image_order) |  # noqa E501
                     (stimulus_presentation['n_after_omission'] == -1))  # for trials without omission
        stim_times = stimulus_presentation[condition][onset]
        stim_ids = stimulus_presentation[condition].index.values
    elif stim_type == 'images-n-before-changes':
        condition = (stimulus_presentation['n_before_change'] == image_order) & \
                    (stimulus_presentation['n_after_omission'] == -1)  # Get trials without omission only
        stim_times = stimulus_presentation[condition][onset]
        stim_ids = stimulus_presentation[condition].index.values
    else:
        stim_times = stimulus_presentation[stimulus_presentation[stim_type]][onset]  # noqa E501
        stim_ids = stimulus_presentation[stimulus_presentation[stim_type]].index.values  # noqa E501

    return stim_times, stim_ids


def get_stimulus_response_xr(ophys_experiment,
                             data_type='dff',
                             stim_type='all',
                             image_order=0,
                             time_window=[-3, 3],
                             response_window_duration=0.5,
                             baseline_window_duration=0.5,
                             interpolate=True,
                             output_sampling_rate=None,
                             exclude_invalid_rois=True,
                             **kwargs):
    '''
    Parameters:
    ___________
    ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    data_type: str
        neural or behavioral data type to extract, options are: dff (default), events, filtered_events, running_speed, pupil_diameter, lick_rate
    stim_type: str
        stimulus type to align to, which can be found in columns of ophys_experiment.stimulus_presentations df.
        options are: 'all' (default) - gets all stimulus trials
                'images' - gets only image presentations (changes and not changes)
                'omissions' - gets only trials with omitted stimuli
                'changes' - get only trials of image changes
                'images-n-omissions' - gets only n-th image presentations after omission
                'images-n-changes' - gets only n-th image presentations after change (and omission > n if there is)
                'images>n-omissions' - gets > n-th image presentations after omission
                'images>n-changes' - gets > n-th image presentations after change (and omission > n if there is)
                'images-n-before-changes' - gets only n-th image presentations before change (from trials without omission only)
    image_order: int
        corresponds to n if stim_type has 'n' in it
        starts at 0 (same as change or omission), default = 0
    time_window: array
        array of two int or floats indicating the time window on sliced data, default = [-3, 3]
    response_window_duration: float
        time period, in seconds, relative to stimulus onset to compute the mean response
    baseline_window_duration: float
        time period, in seconds, relative to stimulus onset to compute the baseline response
        For slow GCaMP and dff traces, it is recommended to set the baseline_window_duration
        shorter than the response_window_duration
    interpolate: bool
        type of alignment. If True (default) - interpolates neural data to align timestamps
        with stimulus presentations. If False - shifts data to the nearest timestamps
    output_sampling_rate : float
        Desired sampling of output.
        Input data will be interpolated to this sampling rate if interpolate = True (default). # NOQA E501
        If passing interpolate = False, the sampling rate of the input timeseries will # NOQA E501
        be used and output_sampling_rate should not be specified.
    exclude_invalid_rois : bool
        If True, only ROIs deemed as 'valid' by the classifier will be returned. If False, 'invalid' ROIs will be returned
        This only works if the provided dataset was loaded using internal Allen Institute database, does not work for NWB files.
        In the case that dataset object is loaded through publicly released NWB files, only 'valid' ROIs will be returned


    kwargs: key, value mappings
        Other keyword arguments are passed down to mindscope_utilities.stim_triggered_response(),
        for interpolation method such as include_endpoint.

    Returns:
    __________
    stimulus_response_xr: xarray
        Xarray of aligned neural data with multiple dimensions: cell_specimen_id,
        'stimlocked_timestamps', and 'stimulus_presentations_id'

    '''

    # load stimulus_presentations table
    stimulus_presentations = ophys_experiment.stimulus_presentations
    stimulus_presentations = add_n_to_stimulus_presentations(stimulus_presentations)

    # get stimulus times and stimulus ids (original order in the stimulus flow)
    stim_times, stim_ids = get_stim_timestamps(
        stimulus_presentations, stim_type, image_order=image_order)

    if ('running' in data_type) or (
            'pupil' in data_type) or ('lick' in data_type):
        # for behavioral datastreams
        # set up variables to handle only one timeseries per stim instead of
        # multiple cell_specimen_ids
        # create a column to take the place of 'cell_specimen_id'
        unique_id_string = 'trace_id'
        unique_ids = [0]  # list to iterate over
    else:
        unique_id_string = 'cell_specimen_id'

    if 'running' in data_type:
        # running_speed attribute is already in tidy format
        data = ophys_experiment.running_speed.copy()
        # rename column so its consistent with data_type
        data = data.rename(columns={'speed': 'running_speed'})
        data[unique_id_string] = 0  # only one value because only one trace
    elif 'pupil' in data_type:
        # eye tracking attribute is in tidy format
        data = ophys_experiment.eye_tracking.copy()
        data = get_pupil_data(
            data,
            interpolate_likely_blinks=True,
            normalize_to_gray_screen=True,
            zscore=False,
            interpolate_to_ophys=False,
            stimulus_presentations=ophys_experiment.stimulus_presentations,
            ophys_timestamps=None)
        # normalize to gray screen baseline
        data[unique_id_string] = 0  # only one value because only one trace
    elif 'lick' in data_type:
        # create dataframe with info about licks for each stimulus timestamp
        data = get_licks_df(ophys_experiment)
        data[unique_id_string] = 0  # only one value because only one trace
    else:
        # load neural data
        data = build_tidy_cell_df(
            ophys_experiment,
            exclude_invalid_rois=exclude_invalid_rois)
        # all cell specimen ids in an ophys_experiment
        unique_ids = np.unique(data['cell_specimen_id'].values)

    # collect aligned data
    sliced_dataout = []

    # align data using interpolation method
    for unique_id in tqdm(unique_ids):
        etr = mindscope_utilities.stim_triggered_response(
            data=data[data[unique_id_string] == unique_id],
            t='timestamps',
            y=data_type,
            stim_times=stim_times,
            t_start=time_window[0],
            t_end=time_window[1],
            output_format='wide',
            interpolate=interpolate,
            output_sampling_rate=output_sampling_rate,
            **kwargs
        )

        # get timestamps array
        trace_timebase = etr.index.values

        # collect aligned data from all cell, all trials into one array
        sliced_dataout.append(etr.transpose().values)

    # convert to xarray
    sliced_dataout = np.array(sliced_dataout)
    stimulus_response_xr = xr.DataArray(
        data=sliced_dataout,
        dims=(unique_id_string, 'stimulus_presentations_id',
              'stimlocked_timestamps'),
        coords={
            'stimlocked_timestamps': trace_timebase,
            'stimulus_presentations_id': stim_ids,
            unique_id_string: unique_ids
        }
    )

    # get traces for significance computation
    if 'events' in data_type:
        traces_array = np.vstack(ophys_experiment.events[data_type].values)
    elif data_type == 'dff':
        traces_array = np.vstack(ophys_experiment.dff_traces['dff'].values)
    else:
        traces_array = data[data_type].values

    # compute mean activity following stimulus onset and during pre-stimulus
    # baseline
    stimulus_response_xr = compute_means_xr(
        stimulus_response_xr,
        response_window_duration=response_window_duration,
        baseline_window_duration=baseline_window_duration)

    # get mean response for each trial
    # input needs to be array of nConditions, nCells
    mean_responses = stimulus_response_xr.mean_response.data.T

    try:
        # get native sampling rate if one is not provided
        if output_sampling_rate is None:
            output_sampling_rate = 1 / data.groupby('cell_specimen_id').apply(lambda x: np.diff(x.timestamps).mean()).mean()
        # compute significance of each trial, returns array of nConditions,
        # nCells
        p_value_gray_screen = get_p_value_from_shuffled_spontaneous(
            mean_responses,
            ophys_experiment.stimulus_presentations,
            ophys_experiment.ophys_timestamps,
            traces_array,
            response_window_duration * output_sampling_rate,
            output_sampling_rate)
    except BaseException:
        p_value_gray_screen = np.zeros(mean_responses.shape)

    # put p_value_gray_screen back into same coordinates as xarray and make it
    # an xarray data array
    p_value_gray_screen = xr.DataArray(
        data=p_value_gray_screen.T,
        coords=stimulus_response_xr.mean_response.coords)

    # create new xarray with means and p-values
    stimulus_response_xr = xr.Dataset({
        'stimlocked_traces': stimulus_response_xr.stimlocked_traces,
        'mean_response': stimulus_response_xr.mean_response,
        'mean_baseline': stimulus_response_xr.mean_baseline,
        'p_value_gray_screen': p_value_gray_screen
    })

    return stimulus_response_xr


def compute_means_xr(stimulus_response_xr, response_window_duration=0.5, baseline_window_duration=0.2):
    '''
    Computes means of traces for stimulus response and pre-stimulus baseline.
    Response by default starts at 0, while baseline
    trace by default ends at 0.

    Parameters:
    ___________
    stimulus_response_xr: xarray
        stimulus_response_xr from get_stimulus_response_xr
        with three main dimentions: cell_specimen_id,
        trail_id, and stimlocked_timestamps
    response_window_duration:
        duration in seconds relative to stimulus onset to compute the mean responses
        in get_stimulus_response_xr
    baseline_window_duration
        duration in seconds relative to stimulus onset to compute the baseline responses
        in get_stimulus_response_xr
        For slow GCaMP and dff traces, it is recommended to set the baseline_window_duration
        shorter than the response_window_duration

    Returns:
    _________
        stimulus_response_xr with additional
        dimentions: mean_response and mean_baseline
    '''
    response_range = [0, response_window_duration]
    baseline_range = [-baseline_window_duration, 0]

    mean_response = stimulus_response_xr.loc[
        {'stimlocked_timestamps': slice(*response_range)}
    ].mean(['stimlocked_timestamps'])

    mean_baseline = stimulus_response_xr.loc[
        {'stimlocked_timestamps': slice(*baseline_range)}
    ].mean(['stimlocked_timestamps'])

    stimulus_response_xr = xr.Dataset({
        'stimlocked_traces': stimulus_response_xr,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
    })

    return stimulus_response_xr


def get_spontaneous_frames(
        stimulus_presentations,
        ophys_timestamps,
        gray_screen_period_to_use='before'):
    '''
        Returns a list of the frames that occur during the before and after spontaneous windows. This is copied from VBA. Does not use the full spontaneous period because that is what VBA did. It only uses 4 minutes of the before and after spontaneous period.

    Args:
        stimulus_presentations_df (pandas.DataFrame): table of stimulus presentations, including start_time and stop_time
        ophys_timestamps (np.array): timestamps of each ophys frame
        gray_screen_period_to_use (str): 'before', 'after', or 'both'
                                        whether to use the gray screen period before the session, after the session, or across both
    Returns:
        spontaneous_inds (np.array): indices of ophys frames during the gray screen period before or after the session, or both
    '''
    # exclude the very first minute of the session because the monitor has just turned on and can cause artifacts
    # spont_duration_frames = 4 * 60 * 60  # 4 mins * * 60s/min * 60Hz
    spont_duration = 4 * 60  # 4mins * 60sec

    # for spontaneous at beginning of session, get 4 minutes of gray screen
    # values prior to first stimulus
    # something weird happens when first stimulus is omitted, start time is at
    # beginning of session
    if stimulus_presentations.iloc[0].image_name == 'omitted':
        first_index = 1
    else:
        first_index = 0
    behavior_start_time = stimulus_presentations.iloc[first_index].start_time
    spontaneous_start_time_pre = behavior_start_time - spont_duration
    spontaneous_end_time_pre = behavior_start_time
    spontaneous_start_frame_pre = mindscope_utilities.index_of_nearest_value(
        ophys_timestamps, spontaneous_start_time_pre)
    spontaneous_end_frame_pre = mindscope_utilities.index_of_nearest_value(
        ophys_timestamps, spontaneous_end_time_pre)
    spontaneous_frames_pre = np.arange(
        spontaneous_start_frame_pre, spontaneous_end_frame_pre, 1)

    # for spontaneous epoch at end of session, get 4 minutes of gray screen
    # values after the last stimulus
    behavior_end_time = stimulus_presentations.iloc[-1].start_time
    spontaneous_start_time_post = behavior_end_time + 0.75
    spontaneous_end_time_post = spontaneous_start_time_post + spont_duration
    spontaneous_start_frame_post = mindscope_utilities.index_of_nearest_value(
        ophys_timestamps, spontaneous_start_time_post)
    spontaneous_end_frame_post = mindscope_utilities.index_of_nearest_value(
        ophys_timestamps, spontaneous_end_time_post)
    spontaneous_frames_post = np.arange(
        spontaneous_start_frame_post, spontaneous_end_frame_post, 1)

    if gray_screen_period_to_use == 'before':
        spontaneous_frames = spontaneous_frames_pre
    elif gray_screen_period_to_use == 'after':
        spontaneous_frames = spontaneous_frames_post
    elif gray_screen_period_to_use == 'both':
        spontaneous_frames = np.concatenate(
            [spontaneous_frames_pre, spontaneous_frames_post])
    return spontaneous_frames


def get_p_value_from_shuffled_spontaneous(mean_responses,
                                          stimulus_presentations,
                                          ophys_timestamps,
                                          traces_array,
                                          response_window_duration,
                                          ophys_frame_rate=None,
                                          number_of_shuffles=10000):
    '''
    Args:
        mean_responses (array): Mean response values, shape (nConditions, nCells)
        stimulus_presentations_df (pandas.DataFrame): Table of stimulus presentations, including start_time and stop_time
        ophys_timestamps (np.array): Timestamps of each ophys frame
        traces_arr (np.array): trace values, shape (nSamples, nCells)
        response_window_duration (int): Number of frames averaged to produce mean response values
        number_of_shuffles (int): Number of shuffles of spontaneous activity used to produce the p-value
    Returns:
        p_values (array): p-value for each response mean, shape (nConditions, nCells)
    '''

    from mindscope_utilities.general_utilities import stimlocked_traces

    spontaneous_frames = get_spontaneous_frames(
        stimulus_presentations,
        ophys_timestamps,
        gray_screen_period_to_use='before')
    shuffled_spont_inds = np.random.choice(
        spontaneous_frames, number_of_shuffles)

    if ophys_frame_rate is None:
        ophys_frame_rate = 1 / np.diff(ophys_timestamps).mean()

    trace_len = np.round(
        response_window_duration * ophys_frame_rate).astype(int)
    start_ind_offset = 0
    end_ind_offset = trace_len
    # get an x frame segment of each cells trace after each shuffled
    # spontaneous timepoint
    spont_traces = stimlocked_traces(
        traces_array,
        shuffled_spont_inds,
        start_ind_offset,
        end_ind_offset)
    # average over the response window (x frames) for each shuffle,
    # Returns (nShuffles, nCells) - mean repsonse following each shuffled
    # spont frame
    spont_mean = spont_traces.mean(axis=0)

    # Goal is to figure out how each response compares to the shuffled distribution, which is just
    # a searchsorted call if we first sort the shuffled.
    # for each cell, sort the spontaneous mean values (axis0 is shuffles,
    # axis1 is cells)
    spont_mean_sorted = np.sort(spont_mean, axis=0)
    response_insertion_ind = np.empty(
        mean_responses.shape)  # should be nConditions, nCells
    # in cases where there is only 1 unique ID (i.e. one neuron in FOV, or one
    # running or pupil trace), duplicate dims so the code below works
    if spont_mean_sorted.ndim == 1:
        spont_mean_sorted = np.expand_dims(spont_mean_sorted, axis=1)
    # loop through indices and figure out how many times the mean response is
    # greater than the spontaneous shuffles
    for ind_cell in range(mean_responses.shape[1]):
        response_insertion_ind[:, ind_cell] = np.searchsorted(
            spont_mean_sorted[:, ind_cell], mean_responses[:, ind_cell])
    # p value is 1 over the fraction times that a given mean response is larger than the 10,000 shuffle means
    # response_insertion_ind tells the index that the mean response would have to be placed in to maintain the order of the shuffled spontaneous
    # if that number is 10k, the mean response is larger than all the shuffles
    # dividing response_insertion_index by 10k gives you the fraction of times that mean response was greater than the shuffles
    # then divide by 1 to get p-value
    proportion_spont_larger_than_sample = 1 - \
        (response_insertion_ind / number_of_shuffles)
    p_values = proportion_spont_larger_than_sample
    # result = xr.DataArray(data=proportion_spont_larger_than_sample,
    #                       coords=mean_responses.coords)
    return p_values


def get_stimulus_response_df(ophys_experiment,
                             data_type='dff',
                             stim_type='all',
                             image_order=0,
                             time_window=[-3, 3],
                             response_window_duration=0.5,
                             baseline_window_duration=0.5,
                             interpolate=True,
                             output_sampling_rate=None,
                             exclude_invalid_rois=True,
                             **kwargs):
    '''
    Get stimulus aligned responses from one ophys_experiment.

    Parameters:
    ___________
    ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    data_type: str
        neural or behavioral data type to extract, options are: dff (default), events, filtered_events, running_speed, pupil_diameter, lick_rate
    stim_type: str
        event type to align to, which can be found in columns of ophys_experiment.stimulus_presentations df.
        options are: 'all' (default) - gets all stimulus trials
                     'images' - gets only image presentations (changes and not changes)
                     'omissions' - gets only trials with omitted stimuli
                     'changes' - get only trials of image changes
                     'images-n-omissions' - gets only n-th image presentations after omission
                     'images-n-changes' - gets only n-th image presentations after change (and omission > n if there is)
                     'images>n-omissions' - gets > n-th image presentations after omission
                     'images>n-changes' - gets > n-th image presentations after change (and omission > n if there is)
                     'images-n-before-changes' - gets only n-th image presentations before change (from trials without omission only)
    image_order: int
        corresponds to n if stim_type has 'n' in it
        starts at 0 (same as change or omission), default = 0
    time_window: array
        array of two int or floats indicating the time window on sliced data, default = [-3, 3]
    response_window_duration: float
        time period, in seconds, relative to stimulus onset to compute the mean response
    baseline_window_duration: float
        time period, in seconds, relative to stimulus onset to compute the baseline
        For slow GCaMP and dff traces, it is recommended to set the baseline_window_duration
        shorter than the response_window_duration
    interpolate: bool
        type of alignment. If True (default) - interpolates neural data to align timestamps
        with stimulus presentations. If False - shifts data to the nearest timestamps
    output_sampling_rate : float
        Desired sampling of output.
        Input data will be interpolated to this sampling rate if interpolate = True (default). # NOQA E501
        If passing interpolate = False, the sampling rate of the input timeseries will # NOQA E501
        be used and output_sampling_rate should not be specified.
     exclude_invalid_rois : bool
        If True, only ROIs deemed as 'valid' by the classifier will be returned. If False, 'invalid' ROIs will be returned
        This only works if the provided dataset was loaded using internal Allen Institute database, does not work for NWB files.
        In the case that dataset object is loaded through publicly released NWB files, only 'valid' ROIs will be returned

    kwargs: key, value mappings
        Other keyword arguments are passed down to mindscope_utilities.stim_triggered_response(),
        for interpolation method such as output_sampling_rate and include_endpoint.

    Returns:
    ___________
    stimulus_response_df: Pandas.DataFrame


    '''

    stimulus_response_xr = get_stimulus_response_xr(
        ophys_experiment=ophys_experiment,
        data_type=data_type,
        stim_type=stim_type,
        image_order=image_order,
        time_window=time_window,
        response_window_duration=response_window_duration,
        baseline_window_duration=baseline_window_duration,
        interpolate=interpolate,
        output_sampling_rate=output_sampling_rate,
        exclude_invalid_rois=exclude_invalid_rois,
        **kwargs)

    # set up identifier columns depending on whether behavioral or neural data is being used
    if ('lick' in data_type) or (
            'pupil' in data_type) or ('running' in data_type):
        # set up variables to handle only one timeseries per stim instead of
        # multiple cell_specimen_ids
        # create a column to take the place of 'cell_specimen_id'
        unique_id_string = 'trace_id'
    else:
        # all cell specimen ids in an ophys_experiment
        unique_id_string = 'cell_specimen_id'

    # get mean response after stimulus onset and during pre-stimulus baseline
    mean_response = stimulus_response_xr['mean_response']
    mean_baseline = stimulus_response_xr['mean_baseline']
    stacked_response = mean_response.stack(
        multi_index=('stimulus_presentations_id', unique_id_string)).transpose()  # noqa E501
    stacked_baseline = mean_baseline.stack(
        multi_index=('stimulus_presentations_id', unique_id_string)).transpose()  # noqa E501

    # get p_value for each stimulus response compared to a shuffled
    # distribution of gray screen values
    p_vals_gray_screen = stimulus_response_xr['p_value_gray_screen']
    stacked_pval_gray_screen = p_vals_gray_screen.stack(
        multi_index=('stimulus_presentations_id', unique_id_string)).transpose()  # noqa E501

    # get stim locked traces and timestamps from xarray
    traces = stimulus_response_xr['stimlocked_traces']
    stacked_traces = traces.stack(multi_index=(
        'stimulus_presentations_id', unique_id_string)).transpose()
    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['stimlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)

    # turn it all into a dataframe
    stimulus_response_df = pd.DataFrame({
        'stimulus_presentations_id': stacked_traces.coords['stimulus_presentations_id'],  # noqa E501
        unique_id_string: stacked_traces.coords[unique_id_string],
        'trace': list(stacked_traces.data),
        'trace_timestamps': list(trace_timestamps),
        'mean_response': stacked_response.data,
        'baseline_response': stacked_baseline.data,
        'p_value_gray_screen': stacked_pval_gray_screen,
    })

    # save frame rate, time window and other metadata for reference
    if output_sampling_rate is not None:
        stimulus_response_df['ophys_frame_rate'] = output_sampling_rate
    else:
        stimulus_response_df['ophys_frame_rate'] = ophys_experiment.metadata['ophys_frame_rate']
    stimulus_response_df['data_type'] = data_type
    stimulus_response_df['stim_type'] = stim_type
    stimulus_response_df['interpolate'] = interpolate
    stimulus_response_df['output_sampling_rate'] = output_sampling_rate
    stimulus_response_df['response_window_duration'] = response_window_duration

    return stimulus_response_df


def add_rewards_to_stimulus_presentations(
    stimulus_presentations,
    rewards,
    time_window=[
        0,
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
            ((reward_times > row["start_time"] + time_window[0]) & (reward_times < row["start_time"] + time_window[1]))],
        axis=1,
    )
    stimulus_presentations["rewards"] = rewards_each_stim
    return stimulus_presentations


def add_licks_to_stimulus_presentations(
    stimulus_presentations,
    licks,
    time_window=[
        0,
        0.75]):
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
            ((lick_times > row["start_time"] + time_window[0]) & (lick_times < row["start_time"] + time_window[1]))],
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
        stimulus_presentations, running_speed, time_window=[0, 0.75]):
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


def add_mean_pupil_to_stimulus_presentations(
    stimulus_presentations,
    eye_tracking,
    column_to_use='pupil_area',
    time_window=[
        0,
        0.75]):
    '''
    Append a column to stimulus_presentations which contains
    the mean pupil area, diameter, or radius in a range relative to
    the stimulus start time.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.  # noqa E501
            Must contain: 'start_time'
        eye_tracking (pd.DataFrame): dataframe of eye tracking data.
            Must contain: timestamps', 'pupil_area', 'pupil_width', 'likely_blinks'
        time_window (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the pupil area.
        column_to_use: column in eyetracking table to use to get mean, options: 'pupil_area', 'pupil_width', 'pupil_radius', 'pupil_diameter'
                        if 'pupil_diameter' or 'pupil_radius' are provided, they will be calculated from 'pupil_area'
                        if 'pupil_width' is provided, the column 'pupil_width' will be directly used from eye_tracking table
    Returns:
        stimulus_presentations table with new column "mean_pupil_"+column_to_use with the
        mean pupil value within the specified window following each stimulus presentation.

    Example:
        # get visual behavior cache
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc  # noqa E501
        cache_dir = r'\\allen\\programs\braintv\\workgroups\nc-ophys\visual_behavior\\platform_paper_cache'
        cache = bpc.from_s3_cache(cache_dir=cache_dir)

        # load data for one experiment
        ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)

        # get necessary tables
        stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
        eye_tracking = ophys_experiment.eye_tracking.copy()

        # add pupil area to stim presentations
        stimulus_presentations = add_mean_pupil_to_stimulus_presentations(stimulus_presentations, eye_tracking, column_to_use='pupil_area')  # noqa E501
    '''

    eye_tracking = get_pupil_data(
        eye_tracking,
        interpolate_likely_blinks=True,
        normalize_to_gray_screen=True,
        zscore=False,
        interpolate_to_ophys=False,
        ophys_timestamps=None,
        stimulus_presentations=stimulus_presentations)

    eye_tracking_timeseries = eye_tracking[column_to_use].values
    mean_pupil_around_stimulus = stimulus_presentations.apply(
        lambda row: get_trace_average(
            eye_tracking_timeseries,
            eye_tracking['timestamps'].values,
            row["start_time"] + time_window[0],
            row["start_time"] + time_window[1],
        ), axis=1,)
    stimulus_presentations["mean_" + column_to_use] = mean_pupil_around_stimulus
    return stimulus_presentations


def add_reward_rate_to_stimulus_presentations(stimulus_presentations, trials):
    '''
    Parameters:
    ____________
    trials: Pandas.DataFrame
        ophys_experiment.trials
    stimulus_presentation: Pandas.DataFrame
        ophys_experiment.stimulus_presentations

    Returns:
    ___________
    stimulus_presentation: Pandas.DataFrame
        with 'reward_rate_trials' column

    'reward_rate_trials' is calculated by the SDK based on the rolling reward rate over trials (not stimulus presentations)
    https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/trials_processing.py#L941
    '''

    last_time = 0
    reward_rate_by_frame = []
    if 'reward_rate' not in trials:
        trials['reward_rate'] = calculate_reward_rate(
            trials['response_latency'].values, trials['start_time'], window=.5)

    trials = trials[trials['aborted'] == False]  # noqa E712
    for change_time in trials.change_time.values:
        reward_rate = trials[trials.change_time == change_time].reward_rate.values[0]
        for start_time in stimulus_presentations.start_time:
            if (start_time < change_time) and (start_time > last_time):
                reward_rate_by_frame.append(reward_rate)
                last_time = start_time
    # fill the last flashes with last value
    for i in range(len(stimulus_presentations) - len(reward_rate_by_frame)):
        reward_rate_by_frame.append(reward_rate_by_frame[-1])

    stimulus_presentations['reward_rate_trials'] = reward_rate_by_frame
    return stimulus_presentations


def add_epochs_to_stimulus_presentations(
        stimulus_presentations,
        time_column='start_time',
        epoch_duration_mins=10):
    """
    Add column called 'epoch' with values as an index for the epoch within a session, for a given epoch duration.

    :param stimulus_presentations: dataframe with a column indicating stim start times
    :param time_column: name of column in dataframe indicating stim times
    :param epoch_duration_mins: desired epoch length in minutes
    :return: input dataframe with epoch column added
    """
    start_time = stimulus_presentations[time_column].values[0]
    stop_time = stimulus_presentations[time_column].values[-1]
    epoch_times = np.arange(start_time, stop_time, epoch_duration_mins * 60)
    stimulus_presentations['epoch'] = None
    for i, time in enumerate(epoch_times):
        if i < len(epoch_times) - 1:
            indices = stimulus_presentations[(stimulus_presentations[time_column] >= epoch_times[i]) & (
                stimulus_presentations[time_column] < epoch_times[i + 1])].index.values
        else:
            indices = stimulus_presentations[(
                stimulus_presentations[time_column] >= epoch_times[i])].index.values
        stimulus_presentations.loc[indices, 'epoch'] = i
    return stimulus_presentations


def add_trials_id_to_stimulus_presentations(stimulus_presentations, trials):
    """
    Add trials_id to stimulus presentations.
    Each stimulus will have associated 'trials_id'.
    'trials_id' is determined by comparing 'start_time 'from stimulus_presentations and trials.
    'trials_id' is last trials_id with start_time <= stimulus_time.

    :param: stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment object, must have 'start_time'
    :param trials: trials attribute of BehaviorOphysExperiment object, must have 'change_time'
    """
    # make a copy of trials with 'start_time' as index to speed lookup
    new_trials = trials.copy().reset_index().set_index('start_time')
    # add trials_id and trial_stimulus_index
    stimulus_presentations['trials_id'] = None
    for idx, row in stimulus_presentations.iterrows():
        # trials_id is last trials_id with start_time <= stimulus_time
        try:
            trials_id = new_trials.loc[:row['start_time']
                                       ].iloc[-1]['trials_id']
        except IndexError:
            trials_id = -1
        stimulus_presentations.at[idx, 'trials_id'] = trials_id
    return stimulus_presentations


def add_trials_data_to_stimulus_presentations_table(
        stimulus_presentations, trials):
    """
    Add trials_id to stimulus presentations table then join relevant columns of trials with stimulus_presentations
    :param: stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment object, must have 'start_time'
    :param trials: trials attribute of BehaviorOphysExperiment object, must have 'change_time'
    """
    # add trials_id and merge to get trial type information
    stimulus_presentations = add_trials_id_to_stimulus_presentations(
        stimulus_presentations, trials)
    # only keep certain columns
    trials = trials[['change_time',
                     'go',
                     'catch',
                     'aborted',
                     'auto_rewarded',
                     'hit',
                     'miss',
                     'false_alarm',
                     'correct_reject',
                     'response_time',
                     'response_latency',
                     'reward_time',
                     'reward_volume',
                     ]]
    # merge trials columns into stimulus_presentations
    stimulus_presentations = stimulus_presentations.reset_index().merge(
        trials, on='trials_id', how='left')
    stimulus_presentations = stimulus_presentations.set_index(
        'stimulus_presentations_id')
    return stimulus_presentations


def add_engagement_state_to_stimulus_presentations(
        stimulus_presentations, trials):
    """
    Add 'engaged' Boolean column and 'engagement_state' string ('engaged' or 'disengaged'
    using threshold of  1/90 rewards per second (~2/3 rewards per minute).
    Will merge trials data in to stimulus presentations if it has not been done already.

    :param stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment
    :param trials: trials attribute of BehaviorOphysExperiment object
    :return: stimulus_presentations with columns added: 'rewarded', 'reward_rate', 'reward_rate_per_second', 'engaged', 'engagement_state'
    """
    if 'reward_time' not in stimulus_presentations.keys():
        stimulus_presentations = add_trials_data_to_stimulus_presentations_table(
            stimulus_presentations, trials)

    # create Boolean column indicating whether the trial was rewarded or not
    stimulus_presentations['rewarded'] = [False if np.isnan(
        reward_time) else True for reward_time in stimulus_presentations.reward_time.values]
    # (rewards/stimulus)*(1 stimulus/.750s) = rewards/second
    stimulus_presentations['reward_rate_per_second'] = stimulus_presentations['rewarded'].rolling(
        window=320, min_periods=1, win_type='triang').mean() / .75  # units of rewards per second
    # (rewards/stimulus)*(1 stimulus/.750s)*(60s/min) = rewards/min
    stimulus_presentations['reward_rate'] = stimulus_presentations['rewarded'].rolling(
        window=320, min_periods=1, win_type='triang').mean() * (60 / .75)  # units of rewards/min

    reward_threshold = 2 / 3  # 2/3 rewards per minute = 1/90 rewards/second
    stimulus_presentations['engaged'] = [
        x > reward_threshold for x in stimulus_presentations['reward_rate'].values]
    stimulus_presentations['engagement_state'] = [
        'engaged' if engaged else 'disengaged' for engaged in stimulus_presentations['engaged'].values]

    return stimulus_presentations


def time_from_last(timestamps, stim_times, side='right'):
    '''
    For each timestamp, returns the time from the most recent other time (in stim_times)

    Args:
        timestamps (np.array): array of timestamps for which the 'time from last stim' will be returned
        stim_times (np.array): stimulus timestamps
    Returns
        time_from_last_stim (np.array): the time from the last stimulus for each timestamp

    '''
    last_stim_index = np.searchsorted(
        a=stim_times, v=timestamps, side=side) - 1
    time_from_last_stim = timestamps - stim_times[last_stim_index]
    # flashes that happened before the other thing happened should return nan
    time_from_last_stim[last_stim_index == -1] = np.nan

    return time_from_last_stim


def add_time_from_last_change_to_stimulus_presentations(
        stimulus_presentations):
    '''
    Adds a column to stimulus_presentations called 'time_from_last_change', which is the time, in seconds since the last image change

    ARGS: SDK session object
    MODIFIES: session.stimulus_presentations
    RETURNS: stimulus_presentations
    '''
    stimulus_times = stimulus_presentations["start_time"].values
    change_times = stimulus_presentations.query(
        'is_change')['start_time'].values
    time_from_last_change = time_from_last(stimulus_times, change_times)
    stimulus_presentations["time_from_last_change"] = time_from_last_change

    return stimulus_presentations


def add_n_to_stimulus_presentations(stimulus_presentations):
    """
    Adds a column to stimulus_presentations called 'n_after_change',
    which is the number of stimulus presentations that have occurred since the last change.
    It will also add a column called 'n_after_omission',
    which is the number of stimulus presentations that have occurred since the last omission,
    before the next change.
    If there is no omission, this value will be -1.
    Presentations before the first change or omission will have a value of -1.
    It will also add a column called 'n_before_change',
    which is the number of stimulus presentations that have occurred before the next change.
    Presentations after the last change will have a value of -1.
    Presentations before the first change will also have a value of -1.
    0 for 'n_after_change' and 'n_before_change' indicates the change itself.
    0 for 'n_after_omission' indicates the omission itself.

    Parameters
    ----------
    stimulus_presentations : pd.DataFrame
        stimulus_presentations table from BehaviorOphysExperiment

    Returns
    -------
    stimulus_presentations : pd.DataFrame
        stimulus_presentations table with 'n_after_change', 'n_after_omission', and 'n_before_change' columns added
    """

    change_ind = stimulus_presentations[stimulus_presentations['is_change']].index.values

    # Adding n_after_change
    n_after_change = np.zeros(len(stimulus_presentations)) - 1  # -1 indicates before the first change
    for i in range(1, len(change_ind)):
        n_after_change[change_ind[i - 1]: change_ind[i]] = np.arange(0, change_ind[i] - change_ind[i - 1]).astype(int)
    n_after_change[change_ind[i]:] = np.arange(0, len(stimulus_presentations) - change_ind[i]).astype(int)
    stimulus_presentations['n_after_change'] = n_after_change

    # Adding n_before_change
    n_before_change = np.zeros(len(stimulus_presentations)) - 1  # -1 indicates after the last and before the first change
    for i in range(len(change_ind) - 1):
        n_before_change[change_ind[i] + 1: change_ind[i + 1] + 1] = np.arange(change_ind[i + 1] - change_ind[i] - 1, -1, -1).astype(int)
    stimulus_presentations['n_before_change'] = n_before_change

    # Adding n_after_omission
    n_after_omission = np.zeros(len(stimulus_presentations)) - 1  # -1 indicates before the first omission or
                                                                  # from the next change till the next omission # noqa E114,E116
    # if there are no omissions, n_after_omission will be all -1
    # and 'omitted' will be added and assigned to False
    if 'omitted' in stimulus_presentations.columns:
        omission_ind = stimulus_presentations[stimulus_presentations['omitted']].index.values
        for i in range(len(omission_ind)):
            if change_ind[-1] > omission_ind[i]:  # if there is a change after the omission
                next_change_ind = change_ind[change_ind > omission_ind[i]][0]
                n_after_omission[omission_ind[i]: next_change_ind] = np.arange(0, next_change_ind - omission_ind[i]).astype(int)
            else:
                n_after_omission[omission_ind[i]:] = np.arange(0, len(stimulus_presentations) - omission_ind[i]).astype(int)
    else:
        stimulus_presentations['omitted'] = False
    stimulus_presentations['n_after_omission'] = n_after_omission

    return stimulus_presentations


def get_annotated_stimulus_presentations(
        ophys_experiment, epoch_duration_mins=10):
    """
    Takes in an ophys_experiment dataset object and returns the stimulus_presentations table with additional columns.
    Adds several useful columns to the stimulus_presentations table, including the mean running speed and pupil diameter for each stimulus,
    the times of licks for each stimulus, the rolling reward rate, an identifier for 10 minute epochs within a session,
    whether or not a stimulus was a pre-change or pre or post omission, and whether change stimuli were hits or misses
    :param ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    :return: stimulus_presentations attribute of BehaviorOphysExperiment, with additional columns added
    """
    stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
    stimulus_presentations = add_licks_to_stimulus_presentations(
        stimulus_presentations, ophys_experiment.licks, time_window=[0, 0.75])
    stimulus_presentations = add_mean_running_speed_to_stimulus_presentations(
        stimulus_presentations, ophys_experiment.running_speed, time_window=[0, 0.75])

    if hasattr('ophys_experiment', 'eye_tracking'):
        try:
            stimulus_presentations = add_mean_pupil_to_stimulus_presentations(
                stimulus_presentations,
                ophys_experiment.eye_tracking,
                column_to_use='pupil_width',
                time_window=[
                    0,
                    0.75])
        except Exception as e:
            print(
                'could not add mean pupil to stimulus presentations, length of eye_tracking attribute is', len(
                    ophys_experiment.eye_tracking))
            print(e)
    stimulus_presentations = add_reward_rate_to_stimulus_presentations(
        stimulus_presentations, ophys_experiment.trials)
    stimulus_presentations = add_epochs_to_stimulus_presentations(
        stimulus_presentations,
        time_column='start_time',
        epoch_duration_mins=epoch_duration_mins)
    stimulus_presentations = add_n_to_stimulus_presentations(stimulus_presentations)
    try:  # not all session types have catch trials or omissions
        stimulus_presentations = add_trials_data_to_stimulus_presentations_table(
            stimulus_presentations, ophys_experiment.trials)
        # add time from last change
        stimulus_presentations = add_time_from_last_change_to_stimulus_presentations(
            stimulus_presentations)
        # add pre-change
        stimulus_presentations['pre_change'] = stimulus_presentations['is_change'].shift(
            -1)
        # add licked Boolean
        stimulus_presentations['licked'] = [True if len(
            licks) > 0 else False for licks in stimulus_presentations.licks.values]
        stimulus_presentations['lick_on_next_flash'] = stimulus_presentations['licked'].shift(
            -1)
        # add engagement state based on reward rate - note this reward rate is
        # calculated differently than the SDK version
        stimulus_presentations = add_engagement_state_to_stimulus_presentations(
            stimulus_presentations, ophys_experiment.trials)
        # add omission annotation
        stimulus_presentations['pre_omitted'] = stimulus_presentations['omitted'].shift(
            -1)
        stimulus_presentations['post_omitted'] = stimulus_presentations['omitted'].shift(
            1)
    except Exception as e:
        print(e)

    return stimulus_presentations


def annotate_stimuli(dataset, inplace=False):
    '''
    adds the following columns to the stimulus_presentations table, facilitating calculation
    of behavior performance based entirely on the stimulus_presentations table:

    'trials_id': the corresponding ID of the trial in the trials table in which the stimulus occurred
    'previous_image_name': the name of the stimulus on the last flash (will list 'omitted' if last stimulus is omitted)
    'next_start_time': The time of the next stimulus start (including the time of the omitted stimulus if the next stimulus is omitted)
    'auto_rewarded': True for trials where rewards were delivered regardless of animal response
    'trial_stimulus_index': index of the given stimulus on the current trial. For example, the first stimulus in a trial has index 0, the second stimulus in a trial has index 1, etc
    'response_lick': Boolean, True if a lick followed the stimulus
    'response_lick_times': list of all lick times following this stimulus
    'response_lick_latency': time difference between first lick and stimulus
    'previous_response_on_trial': Boolean, True if there has been a lick to a previous stimulus on this trial
    'could_change': Boolean, True if the stimulus met the conditions that would have allowed
                    to be chosen as the change stimulus by camstim:
                        * at least the fourth stimulus flash in the trial
                        * not preceded by any licks on that trial

    Parameters:
    -----------
    dataset : BehaviorSession or BehaviorOphysSession object
        an SDK session object
    inplace : Boolean
        If True, operates on the dataset.stimulus_presentations object directly and returns None
        If False (default), operates on a copy and returns the copy

    Returns:
    --------
    Pandas.DataFrame (if inplace == False)
    None (if inplace == True)
    '''

    if inplace:
        stimulus_presentations = dataset.stimulus_presentations
    else:
        stimulus_presentations = dataset.stimulus_presentations.copy()

    # add previous_image_name
    stimulus_presentations['previous_image_name'] = stimulus_presentations['image_name'].shift(
    )

    # add next_start_time
    stimulus_presentations['next_start_time'] = stimulus_presentations['start_time'].shift(
        -1)

    # add trials_id and trial_stimulus_index
    stimulus_presentations['trials_id'] = None
    stimulus_presentations['trial_stimulus_index'] = None
    last_trial_id = -1
    trial_stimulus_index = 0

    # add response_lick, response_lick_times, response_lick_latency
    stimulus_presentations['response_lick'] = False
    stimulus_presentations['response_lick_times'] = None
    stimulus_presentations['response_lick_latency'] = None

    # make a copy of trials with 'start_time' as index to speed lookup
    trials = dataset.trials.copy().reset_index().set_index('start_time')

    # make a copy of licks with 'timestamps' as index to speed lookup
    licks = dataset.licks.copy().reset_index().set_index('timestamps')

    # iterate over every stimulus
    for idx, row in stimulus_presentations.iterrows():
        # trials_id is last trials_id with start_time <= stimulus_time
        try:
            trials_id = trials.loc[:row['start_time']].iloc[-1]['trials_id']
        except IndexError:
            trials_id = -1
        stimulus_presentations.at[idx, 'trials_id'] = trials_id

        if trials_id == last_trial_id:
            trial_stimulus_index += 1
        else:
            trial_stimulus_index = 0
            last_trial_id = trials_id
        stimulus_presentations.at[idx,
                                  'trial_stimulus_index'] = trial_stimulus_index

        # note the `- 1e-9` acts as a <, as opposed to a <=
        stim_licks = licks.loc[row['start_time']:row['next_start_time'] - 1e-9].index.to_list()

        stimulus_presentations.at[idx, 'response_lick_times'] = stim_licks
        if len(stim_licks) > 0:
            stimulus_presentations.at[idx, 'response_lick'] = True
            stimulus_presentations.at[idx,
                                      'response_lick_latency'] = stim_licks[0] - row['start_time']

    # merge in auto_rewarded column from trials table
    stimulus_presentations = stimulus_presentations.reset_index().merge(
        dataset.trials[['auto_rewarded']],
        on='trials_id',
        how='left',
    ).set_index('stimulus_presentations_id')

    # add previous_response_on_trial
    stimulus_presentations['previous_response_on_trial'] = False
    # set 'stimulus_presentations_id' and 'trials_id' as indices to speed
    # lookup
    stimulus_presentations = stimulus_presentations.reset_index(
    ).set_index(['stimulus_presentations_id', 'trials_id'])
    for idx, row in stimulus_presentations.iterrows():
        stim_id, trials_id = idx
        # get all stimuli before the current on the current trial
        mask = (stimulus_presentations.index.get_level_values(0) < stim_id) & (
            stimulus_presentations.index.get_level_values(1) == trials_id)
        # check to see if any previous stimuli have a response lick
        stimulus_presentations.at[idx,
                                  'previous_response_on_trial'] = stimulus_presentations[mask]['response_lick'].any()
    # set the index back to being just 'stimulus_presentations_id'
    stimulus_presentations = stimulus_presentations.reset_index(
    ).set_index('stimulus_presentations_id')

    # add could_change
    stimulus_presentations['could_change'] = False
    for idx, row in stimulus_presentations.iterrows():
        # check if we meet conditions where a change could occur on this
        # stimulus (at least 4th flash of trial, no previous change on trial)
        if row['trial_stimulus_index'] >= 4 and row['previous_response_on_trial'] is False and row[
                'image_name'] != 'omitted' and row['previous_image_name'] != 'omitted':
            stimulus_presentations.at[idx, 'could_change'] = True

    if inplace is False:
        return stimulus_presentations


def calculate_response_matrix(
        stimuli,
        aggfunc=np.mean,
        sort_by_column=True,
        engaged_only=True):
    '''
    calculates the response matrix for each individual image pair in the `stimulus` dataframe

    Parameters:
    -----------
    stimuli: Pandas.DataFrame
        From experiment.stimulus_presentations, after annotating as follows:
            annotate_stimuli(experiment, inplace = True)
    aggfunc: function
        function to apply to calculation. Default = np.mean
        other options include np.size (to get counts) or np.median
    sort_by_column: Boolean
        if True (default), sorts outputs by column means
    engaged_only: Boolean
        If True (default), calculates only on engaged trials
        Will throw an assertion error if True and 'engagement_state' column does not exist

    Returns:
    --------
    Pandas.DataFrame
        matrix of response probabilities for each image combination
        index = previous image
        column = current image
        catch trials are on diagonal

    '''
    stimuli_to_analyze = stimuli.query(
        'auto_rewarded == False and could_change == True and image_name != "omitted" and previous_image_name != "omitted"')
    if engaged_only:
        assert 'engagement_state' in stimuli_to_analyze.columns, 'stimuli must have column called "engagement_state" if passing engaged_only = True'
        stimuli_to_analyze = stimuli_to_analyze.query(
            'engagement_state == "engaged"')

    response_matrix = pd.pivot_table(
        stimuli_to_analyze,
        values='response_lick',
        index=['previous_image_name'],
        columns=['image_name'],
        aggfunc=aggfunc
    ).astype(float)

    if sort_by_column:
        sort_by = response_matrix.mean(axis=0).sort_values().index
        response_matrix = response_matrix.loc[sort_by][sort_by]

    response_matrix.index.name = 'previous_image_name'

    return response_matrix


def calculate_dprime_matrix(stimuli, sort_by_column=True, engaged_only=True):
    '''
    calculates the d' matrix for each individual image pair in the `stimulus` dataframe

    Parameters:
    -----------
    stimuli: Pandas.DataFrame
        From experiment.stimulus_presentations, after annotating as follows:
            annotate_stimuli(experiment, inplace = True)
    sort_by_column: Boolean
        if True (default), sorts outputs by column means
    engaged_only: Boolean
        If True (default), calculates only on engaged trials
        Will throw an assertion error if True and 'engagement_state' column does not exist

    Returns:
    --------
    Pandas.DataFrame
        matrix of d' for each image combination
        index = previous image
        column = current image
        catch trials are on diagonal

    '''
    if engaged_only:
        assert 'engagement_state' in stimuli.columns, 'stimuli must have column called "engagement_state" if passing engaged_only = True'

    response_matrix = calculate_response_matrix(
        stimuli,
        aggfunc=np.mean,
        sort_by_column=sort_by_column,
        engaged_only=engaged_only)

    d_prime_matrix = response_matrix.copy()
    for row in response_matrix.columns:
        for col in response_matrix.columns:
            d_prime_matrix.loc[row][col] = mindscope_utilities.dprime(
                hit_rate=response_matrix.loc[row][col],
                fa_rate=response_matrix[col][col],
                limits=False
            )
            if row == col:
                d_prime_matrix.loc[row][col] = np.nan

    return d_prime_matrix


def get_licks_df(ophys_experiment):
    '''
    Creates a dataframe containing columns for 'timestamps', 'licks', where values are from
    a binary array of the length of stimulus timestamps where frames with no lick are 0 and frames with a lick are 1,
    and a column called 'lick_rate' with values of 'licks' averaged over a 6 frame window to get licks per 100ms,
    Can be used to plot stim triggered average lick rate
    Parameters:
    -----------
    ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501

    Returns:
    --------
    Pandas.DataFrame with columns 'timestamps', 'licks', and 'lick_rate' in units of licks / 100ms

    '''
    timestamps = ophys_experiment.stimulus_timestamps.copy()
    licks = ophys_experiment.licks.copy()
    lick_array = np.zeros(timestamps.shape)
    lick_array[licks.frame.values] = 1
    licks_df = pd.DataFrame(data=timestamps, columns=['timestamps'])
    licks_df['licks'] = lick_array
    licks_df['lick_rate'] = licks_df['licks'].rolling(
        window=6, min_periods=1, win_type='triang').mean()

    return licks_df


def get_pupil_data(
        eye_tracking,
        interpolate_likely_blinks=False,
        normalize_to_gray_screen=False,
        zscore=False,
        interpolate_to_ophys=False,
        ophys_timestamps=None,
        stimulus_presentations=None):
    """
    Takes eye_tracking attribute of AllenSDK BehaviorOphysExperiment objection and optionally
    removes 'likely_blinks' from all columns in dataframe,
    interpolates over NaNs resulting from removing likely_blinks if interpolate = True,
    normalizes to the 5 minute gray screen period at the beginning of each ophys session,
    z-scores the timeseries, and/or aligns to ophys timestamps

    :param eye_tracking: eye_tracking attribute of AllenSDK BehaviorOphysExperiment object
    :param interpolate_likely_blinks: Boolean, whether or not to interpolate points where likely_blinks occured
    :param normalize_to_gray_screen: Boolean, whether or not to normalize eye_tracking values to the 5 minute gray screen period
    :param zscore: Boolean, whether or not to z-score the eye tracking values
    :param interpolate_to_ophys: Boolean, whether or not to interpolate eye tracking timestamps on to ophys timestamps
    :param ophys_timestamps: ophys_timestamps attribute of AllenSDK BehaviorOphysExperiment object, required to interpolate to ophys
    :param stimulus_presentations: stimulus_presentations attribute of AllenSDK BehaviorOphysExperiment object,
                                    required to normaliz to gray screen period


    :return:
    """
    import scipy

    # set index to timestamps so they dont get overwritten by subsequent
    # operations
    eye_tracking = eye_tracking.set_index('timestamps')

    # add timestamps column in addition to index so it can be used as a column as well
    eye_tracking['timestamps'] = eye_tracking.index.values

    # set all timepoints that are likely blinks to NaN for all eye_tracking
    # columns
    if True in eye_tracking.likely_blink.unique(
    ):  # only can do this if there are likely blinks to filter out
        eye_tracking.loc[eye_tracking['likely_blink'], :] = np.nan

    # interpolate over likely blinks, which are now NaNs
    if interpolate_likely_blinks:
        eye_tracking = eye_tracking.interpolate()

    # divide all columns by average value during gray screen period prior to behavior session
    if normalize_to_gray_screen:
        assert stimulus_presentations is not None, 'must provide stimulus_presentations if normalize_to_gray_screen is True'
        spontaneous_frames = get_spontaneous_frames(
            stimulus_presentations,
            eye_tracking.timestamps.values,
            gray_screen_period_to_use='before')
        for column in eye_tracking.keys():
            if (column != 'timestamps') and (column != 'likely_blink'):
                gray_screen_mean_value = np.nanmean(
                    eye_tracking[column].values[spontaneous_frames])
                eye_tracking[column] = eye_tracking[column] / \
                    gray_screen_mean_value
    # z-score pupil data
    if zscore:
        for column in eye_tracking.keys():
            if (column != 'timestamps') and (column != 'likely_blink'):
                eye_tracking[column] = scipy.stats.zscore(
                    eye_tracking[column], nan_policy='omit')

    # interpolate to ophys timestamps
    if interpolate_to_ophys:
        assert ophys_timestamps is not None, 'must provide ophys_timestamps if interpolate_to_ophys is True'
        eye_tracking_ophys_time = pd.DataFrame(
            {'timestamps': ophys_timestamps})
        for column in eye_tracking.keys():
            if (column != 'timestamps') and (column != 'likely_blink'):
                f = scipy.interpolate.interp1d(
                    eye_tracking['timestamps'],
                    eye_tracking[column],
                    bounds_error=False)
                eye_tracking_ophys_time[column] = f(
                    eye_tracking_ophys_time['timestamps'])
                eye_tracking_ophys_time[column].fillna(
                    method='ffill', inplace=True)
        eye_tracking = eye_tracking_ophys_time

    return eye_tracking
