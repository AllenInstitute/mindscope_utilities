import pandas as pd
# import numpy as np
from mindscope_utilities import visual_behavior_ophys


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


def test_annotate_stimuli(visual_behavior_ophys_test_experiment):
    '''
    uses `visual_behavior_ophys_test_experiment` defined in conftest.py
    '''
    experiment = visual_behavior_ophys_test_experiment

    annotated_stimuli = visual_behavior_ophys.annotate_stimuli(experiment)
    assert annotated_stimuli.loc[4794]['next_start_time'] == 3908.26225


def test_calculate_response_matrix(visual_behavior_ophys_test_experiment):
    '''
    uses `visual_behavior_ophys_test_experiment` defined in conftest.py
    '''
    experiment = visual_behavior_ophys_test_experiment

    annotated_stimuli = visual_behavior_ophys.annotate_stimuli(experiment)
    rm = visual_behavior_ophys.calculate_response_matrix(
        annotated_stimuli,
        sort_by_column=True,
        engaged_only=False
    )

    assert rm.loc['im106']['im106'] == 0.49411764705882355
    assert rm.loc['im035']['im106'] == 0.5


def test_calculate_dprime_matrix(visual_behavior_ophys_test_experiment):
    '''
    uses `visual_behavior_ophys_test_experiment` defined in conftest.py
    '''
    experiment = visual_behavior_ophys_test_experiment

    annotated_stimuli = visual_behavior_ophys.annotate_stimuli(experiment)
    dprime_matrix = visual_behavior_ophys.calculate_dprime_matrix(
        annotated_stimuli,
        sort_by_column=True,
        engaged_only=False
    )

    assert dprime_matrix.loc['im035']['im106'] == 0.014745406527902915
    assert pd.isnull(dprime_matrix.loc['im106']['im106'])
