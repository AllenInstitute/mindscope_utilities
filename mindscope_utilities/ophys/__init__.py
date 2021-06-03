import pandas as pd


def get_cell_timeseries_dict(session, cell_specimen_id):
    '''
    for a given cell_specimen ID, this function creates a dictionary with the following keys
    * timestamps: ophys timestamps
    * cell_roi_id
    * cell_specimen_id
    * dff
    * events
    * filtered events
    This is useful for generating a tidy dataframe
    arguments:
        session object
        cell_specimen_id
    returns
        dict
    '''
    cell_dict = {
        'timestamps': session.ophys_timestamps,
        'cell_roi_id': [session.dff_traces.loc[cell_specimen_id]['cell_roi_id']] * len(session.ophys_timestamps),
        'cell_specimen_id': [cell_specimen_id] * len(session.ophys_timestamps),
        'dff': session.dff_traces.loc[cell_specimen_id]['dff'],
        'events': session.events.loc[cell_specimen_id]['events'],
        'filtered_events': session.events.loc[cell_specimen_id]['filtered_events'],

    }

    return cell_dict


def build_tidy_cell_df(session):
    '''
    builds a tidy dataframe describing activity for every cell in session containing the following columns
    * timestamps: the ophys timestamps
    * cell_roi_id: the cell roi id
    * cell_specimen_id: the cell specimen id
    * dff: measured deltaF/F for every timestep
    * events: extracted events for every timestep
    * filtered events: filtered events for every timestep
    Takes a few seconds to build
    arguments:
        session
    returns:
        pandas dataframe
    '''
    return pd.concat([pd.DataFrame(get_cell_timeseries_dict(session, cell_specimen_id)) for cell_specimen_id in session.dff_traces.reset_index()['cell_specimen_id']]).reset_index(drop=True)
