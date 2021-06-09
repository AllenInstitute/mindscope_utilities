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
        cell_roi_id = row['cell_roi_id']

        # build a tidy dataframe for this cell
        cell_df = pd.DataFrame({
            'timestamps': experiment.ophys_timestamps,
            'cell_roi_id': [cell_roi_id] * len(experiment.ophys_timestamps),
            'cell_specimen_id': [cell_specimen_id] * len(experiment.ophys_timestamps),  # noqa E501
            'dff': experiment.dff_traces.loc[cell_specimen_id]['dff'] if cell_specimen_id in experiment.dff_traces.index else [np.nan] * len(experiment.ophys_timestamps),  # noqa E501
            'events': experiment.events.loc[cell_specimen_id]['events'] if cell_specimen_id in experiment.events.index else [np.nan] * len(experiment.ophys_timestamps),  # noqa E501
            'filtered_events': experiment.events.loc[cell_specimen_id]['filtered_events'] if cell_specimen_id in experiment.events.index else [np.nan] * len(experiment.ophys_timestamps),  # noqa E501
        })

        # append the dataframe for this cell to the list of cell dataframes
        list_of_cell_dfs.append(cell_df)

    # concatenate all dataframes in the list
    tidy_df = pd.concat(list_of_cell_dfs)

    # return the tidy dataframe
    return tidy_df
