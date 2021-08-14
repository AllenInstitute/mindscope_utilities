import pandas as pd
import numpy as np


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
        event_times = stimulus_presentations_df[stimulus_presentations_df['omitted'] == False][onset]
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