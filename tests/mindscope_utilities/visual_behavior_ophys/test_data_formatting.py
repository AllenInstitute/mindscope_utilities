import pandas as pd
from mindscope_utilities import visual_behavior_ophys
import allensdk.brain_observatory.behavior.behavior_project_cache as cache


def test_build_tidy_cell_df(simulated_experiment_fixture):
    '''
    tests visual_behavior_ophys.data_formatting.build_tidy_cell_df
    uses `simulated_experiment_fixture` defined in conftest.py
    '''
    tidy_cell_df = visual_behavior_ophys.build_tidy_cell_df(
        simulated_experiment_fixture)

    ans_1 = pd.DataFrame([{
        'timestamps': 0.5,
        'cell_roi_id': 2,
        'cell_specimen_id': 1,
        'dff': 0.0,
        'events': 0.0,
        'filtered_events': 1.0
    }])
    ans_1['cell_specimen_id'] = pd.Categorical(
        ans_1['cell_specimen_id'], categories=[1, 2, 3])
    ans_1['cell_roi_id'] = pd.Categorical(
        ans_1['cell_roi_id'], categories=[2, 4, 6])

    cols = ['timestamps', 'cell_roi_id', 'cell_specimen_id',
            'dff', 'events', 'cell_specimen_id']
    actual_1 = tidy_cell_df.query(
        'cell_specimen_id == 1 and timestamps == 0.5').reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_1[cols], ans_1[cols])

    ans_2 = pd.DataFrame([{
        'timestamps': 0.5,
        'cell_roi_id': 4,
        'cell_specimen_id': 2,
        'dff': -1.0,
        'events': 1.0,
        'filtered_events': 0.0
    }])
    ans_2['cell_specimen_id'] = pd.Categorical(
        ans_2['cell_specimen_id'], categories=[1, 2, 3])
    ans_2['cell_roi_id'] = pd.Categorical(
        ans_2['cell_roi_id'], categories=[2, 4, 6])
    actual_2 = tidy_cell_df.query(
        'cell_specimen_id == 2 and timestamps == 0.5').reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_2[cols], ans_2[cols])


def test_get_event_timestamps(experiment_id):
    ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)
    stimulus_presentations = ophys_experiment.stimulus_presentations
    timestamp = 322.78641
    event_times = visual_behavior_ophys.get_event_timestamps(
        stimulus_presentations, 'changes')
    assert(event_times[0] == timestamp)
