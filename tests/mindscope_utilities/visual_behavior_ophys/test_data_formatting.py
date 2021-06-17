import pandas as pd
import numpy as np
from mindscope_utilities import visual_behavior_ophys


def test_build_tidy_cell_df(simulated_experiment_fixture):
    '''
    tests visual_behavior_ophys.data_formatting.build_tidy_cell_df
    uses `simulated_experiment_fixture` defined in conftest.py
    '''
    tidy_cell_df = visual_behavior_ophys.build_tidy_cell_df(simulated_experiment_fixture)

    ans_1 = pd.DataFrame([{
        'timestamps': 0.5,
        'cell_roi_id': 2,
        'cell_specimen_id': 1,
        'dff': 0.0,
        'events': 0.0,
        'filtered_events': 1.0
    }])
    pd.testing.assert_frame_equal(
        tidy_cell_df.query('cell_specimen_id == 1 and timestamps == 0.5').reset_index(drop=True),
        ans_1
    )

    ans_2 = pd.DataFrame([{
        'timestamps': 0.5,
        'cell_roi_id': 4,
        'cell_specimen_id': 2,
        'dff': -1.0,
        'events': 1.0,
        'filtered_events': 0.0
    }])
    pd.testing.assert_frame_equal(
        tidy_cell_df.query('cell_specimen_id == 2 and timestamps == 0.5').reset_index(drop=True),
        ans_2
    )