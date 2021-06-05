import pandas as pd


def build_tidy_cell_df(experiment):
    '''
    Builds a tidy dataframe describing activity for every cell in experiment.
    Tidy format is defined as one row per observation.
    Thus, the output dataframe will be n_cells x n_timetpoints long

    Parameters:
    -----------
    experiment : AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py  # noqa E501

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

    # iterate over each individual cell
    for cell_specimen_id in experiment.dff_traces.reset_index()['cell_specimen_id']:  # noqa E501

        # build a tidy dataframe for this cell
        cell_df = pd.DataFrame({
            'timestamps': experiment.ophys_timestamps,
            'cell_roi_id': [experiment.dff_traces.loc[cell_specimen_id]['cell_roi_id']] * len(experiment.ophys_timestamps),  # noqa E501
            'cell_specimen_id': [cell_specimen_id] * len(experiment.ophys_timestamps),  # noqa E501
            'dff': experiment.dff_traces.loc[cell_specimen_id]['dff'],
            'events': experiment.events.loc[cell_specimen_id]['events'],
            'filtered_events': experiment.events.loc[cell_specimen_id]['filtered_events'],  # noqa E501
        })

        # append the dataframe for this cell to the list of cell dataframes
        list_of_cell_dfs.append(cell_df)

    # concatenate all dataframes in the list
    tidy_df = pd.concat(list_of_cell_dfs)

    # return the tidy dataframe
    return tidy_df
