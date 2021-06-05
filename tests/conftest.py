# NOTE: this file contains fixtures used by other test files.
import pandas as pd
import numpy as np
import pytest

class SimulatedExperiment(object):
    '''
    A class to simulate an AllenSDK BehaviorOphysExperiment object
    for testing purposes
    
    Parameters:
    -----------
    cell_specimen_ids : list of ints
        list of cell specimen IDs
    cell_roi_ids : list of ints
        list of cell ROI IDs
    timestamps : list of floats
        measurement timestamps
    dff : list of lists/arrays of floats
        dff measurements for every cell
        each sublist must be same length as timestamps
    events : list of lists/arrays of floats
        event measurements for every cell
        each sublist must be same length as timestamps
    filtered_events : list of lists/arrays of floats
        filtered events measurements for every cell
        each sublist must be same length as timestamps
        
    Returns:
    --------
    simulated AllenSDK BehaviorOphysExperiment object
    '''
    def __init__(self, cell_specimen_ids, cell_roi_ids, timestamps, dff, events, filtered_events):
        self.ophys_timestamps = np.array(timestamps)
        self.dff_traces = self.build_dff_traces(
            cell_specimen_ids,
            cell_roi_ids,
            dff
        )
        self.events = self.build_events(
            cell_specimen_ids,
            cell_roi_ids,
            events,
            filtered_events
        )
        
    def build_dff_traces(self, cell_specimen_ids, cell_roi_ids, dff):
        '''builds a dff_traces dataframe'''
        dff_traces = pd.DataFrame({
            'cell_specimen_id': cell_specimen_ids,
            'cell_roi_id': cell_roi_ids,
            'dff': dff
        })
        return dff_traces.set_index('cell_specimen_id')
    
    def build_events(self, cell_specimen_ids, cell_roi_ids, events, filtered_events):
        '''builds an events dataframe'''
        events = pd.DataFrame({
            'cell_specimen_id': cell_specimen_ids,
            'cell_roi_id': cell_roi_ids,
            'events': events,
            'filtered_events': filtered_events,
            'lambda': np.arange(len(cell_specimen_ids)),
            'noise_std': np.arange(len(cell_specimen_ids)),
        })
        return events.set_index('cell_specimen_id')

@pytest.fixture
def simulated_experiment_fixture():
   # build a simulated experiment
    timestamps = np.arange(0, 1, 0.01)
    cell_specimen_ids = [1, 2, 3]
    simulated_experiment = SimulatedExperiment(
        cell_specimen_ids = cell_specimen_ids,
        cell_roi_ids = [2*csid for csid in cell_specimen_ids],
        timestamps = timestamps,
        dff = [np.sin(2*np.pi*timestamps + ii*0.5*np.pi) for ii, cell in enumerate(cell_specimen_ids)],
        events = [np.sin(4*np.pi*timestamps+ ii*0.5*np.pi) for ii, cell in enumerate(cell_specimen_ids)],
        filtered_events = [np.cos(4*np.pi*timestamps+ ii*0.5*np.pi) for ii, cell in enumerate(cell_specimen_ids)]
    )
    return simulated_experiment
