import asyncio
import pandas as pd
import pytest
from pathlib import Path
import shutil
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

# Adjust the import path if your project structure is different
# This assumes tests/ is at the same level as src/
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.main_real import append_csv

# Pytest marker for async functions
pytestmark = pytest.mark.asyncio

# Helper function to create mock objects
def create_mock_objects():
    mock_chk = MagicMock()
    mock_chk.__getitem__.side_effect = lambda key: {'episode': 1}.get(key, 0) # Allows mock_chk['episode']

    mock_env = MagicMock()
    mock_env.time = datetime(2023, 1, 1, 12, 0, 0)
    mock_env.last_action = "env_action_1"

    mock_glue = MagicMock()
    mock_glue.num_steps = 10
    mock_glue.total_reward = 100.0

    mock_interaction = MagicMock()
    mock_interaction.o = "obs_1" # Observation
    mock_interaction.a = "agent_action_1" # Agent action
    mock_interaction.r = 0.5 # Reward
    mock_interaction.t = False # Terminal
    mock_interaction.extra = {"df": pd.DataFrame({'plant_id': [0], 'frame': [10]}), "some_extra_data": 123}

    return mock_chk, mock_env, mock_glue, mock_interaction

# Test case 1: New file creation
async def test_append_csv_new_file(tmp_path):
    mock_chk, mock_env, mock_glue, mock_interaction = create_mock_objects()
    raw_csv_path = tmp_path / "new_data.csv"
    img_name = "image1.jpg"

    # Data to be appended
    data_dict = {
        "time": [mock_env.time],
        "frame": [mock_glue.num_steps],
        "state_0": ["obs_1"], # Assuming expand("state", "obs_1") -> {"state_0": "obs_1"}
        "action_0": ["env_action_1"], # Assuming expand("action", "env_action_1") -> {"action_0": "env_action_1"}
        "agent_action": [mock_interaction.a],
        "reward": [mock_interaction.r],
        "terminal": [mock_interaction.t],
        "steps": [mock_glue.num_steps],
        "image_name": [img_name],
        "return": [None], # total_reward if terminal, else None
        "episode": [mock_chk['episode']],
        "some_extra_data": [mock_interaction.extra["some_extra_data"]],
        # Columns from interaction.extra["df"] after merge
        "plant_id": [0],
    }
    # Remove keys that would not be present if their interaction values are None
    if mock_interaction.r is None:
        del data_dict["reward"]


    # We need to carefully construct the expected DataFrame based on append_csv's logic
    # For simplicity in this example, we'll mock interaction.extra["df"] to be simple
    # and assume expand function behavior.
    # A more robust test would replicate the expand logic or mock it.

    # Mimic the DataFrame creation inside append_csv before merge
    # This is a simplified version. The actual df construction in append_csv is more complex.
    expected_df_before_merge_data = {}
    for key, val_list in data_dict.items():
        if key not in ["plant_id"]: # plant_id comes from the merge
             expected_df_before_merge_data[key] = val_list
    
    # Create the DataFrame that gets passed to append_csv (simplified)
    # The actual df inside append_csv is created from various sources.
    # For this test, we'll construct the DataFrame that we expect to be written.
    # This means we need to simulate the merge that happens inside append_csv
    
    # Data that would be in the df created inside append_csv
    base_data_for_df = {
        "time": mock_env.time,
        "frame": mock_glue.num_steps,
        "state_0": mock_interaction.o, # Simplified: assuming expand results in 'state_0'
        "action_0": mock_env.last_action, # Simplified
        "agent_action": mock_interaction.a,
        "reward": mock_interaction.r,
        "terminal": mock_interaction.t,
        "steps": mock_glue.num_steps,
        "image_name": img_name,
        "return": None, # as t is False
        "episode": mock_chk['episode'],
        "some_extra_data": mock_interaction.extra["some_extra_data"],
    }
    # Flatten np arrays if any, as per append_csv logic (simplified here)
    #expanded_info = expand_dict_with_expand_function_results
    #df_intermediate = pd.DataFrame(expanded_info_plus_other_fields)

    # For this test, let's define the final expected DataFrame directly
    # This requires knowing the exact column order and content post-merge
    final_expected_data = {
        'time': [datetime(2023, 1, 1, 12, 0, 0)],
        'frame': [10],
        'state_0': ['obs_1'],
        'action_0': ['env_action_1'],
        'agent_action': ['agent_action_1'],
        'reward': [0.5],
        'terminal': [False],
        'steps': [10],
        'image_name': ['image1.jpg'],
        'return': [None],
        'episode': [1],
        'some_extra_data': [123],
        'plant_id': [0] # This comes from interaction.extra["df"]
    }
    expected_df = pd.DataFrame(final_expected_data)
    # Ensure column order matches what pd.DataFrame would create, or sort if order is not critical for the test
    # For this example, let's assume the order in final_expected_data is the expected order.


    await append_csv(mock_chk, mock_env, mock_glue, raw_csv_path, img_name, mock_interaction)

    assert raw_csv_path.exists()
    df_read = pd.read_csv(raw_csv_path)
    
    # Convert time column to datetime for proper comparison if it's read as string
    if 'time' in df_read.columns:
        df_read['time'] = pd.to_datetime(df_read['time'])
    if 'time' in expected_df.columns:
        expected_df['time'] = pd.to_datetime(expected_df['time'])


    # Sort columns for comparison as df.to_csv might not preserve original df column order
    # depending on pandas version or if df is rebuilt.
    # However, the header in the CSV will be written based on df.columns.
    # For this test, let's assume the column order in expected_df is what we want.
    df_read = df_read[expected_df.columns]

    pd.testing.assert_frame_equal(df_read, expected_df, check_dtype=False) # check_dtype=False for flexibility with int/float


async def test_append_csv_existing_file_same_columns(tmp_path):
    mock_chk, mock_env, mock_glue, mock_interaction = create_mock_objects()
    raw_csv_path = tmp_path / "existing_data.csv"
    img_name_initial = "initial_img.jpg"
    img_name_new = "new_img.jpg"

    # Initial data
    initial_data = {
        'time': [datetime(2023, 1, 1, 10, 0, 0)], 'frame': [1], 'state_0': ['initial_obs'],
        'action_0': ['initial_env_action'], 'agent_action': ['initial_agent_action'],
        'reward': [0.1], 'terminal': [False], 'steps': [1], 'image_name': [img_name_initial],
        'return': [None], 'episode': [1], 'some_extra_data': [100], 'plant_id': [10]
    }
    initial_df = pd.DataFrame(initial_data)
    initial_df.to_csv(raw_csv_path, index=False)

    # New data to append (matches columns of initial_df)
    # Update mock objects for new data
    mock_env.time = datetime(2023, 1, 1, 12, 30, 0)
    mock_glue.num_steps = 11
    mock_interaction.o = "obs_2"
    mock_env.last_action = "env_action_2"
    mock_interaction.a = "agent_action_2"
    mock_interaction.r = 0.6
    mock_interaction.extra["df"] = pd.DataFrame({'plant_id': [11], 'frame': [11]}) # new plant_id for new data
    mock_interaction.extra["some_extra_data"] = 124


    await append_csv(mock_chk, mock_env, mock_glue, raw_csv_path, img_name_new, mock_interaction)

    # Expected combined data
    new_data_as_written = {
        'time': [mock_env.time], 'frame': [mock_glue.num_steps], 'state_0': [mock_interaction.o],
        'action_0': [mock_env.last_action], 'agent_action': [mock_interaction.a],
        'reward': [mock_interaction.r], 'terminal': [mock_interaction.t], 'steps': [mock_glue.num_steps],
        'image_name': [img_name_new], 'return': [None], 'episode': [mock_chk['episode']],
        'some_extra_data': [mock_interaction.extra["some_extra_data"]], 'plant_id': [11]
    }
    new_df_row = pd.DataFrame(new_data_as_written)
    expected_combined_df = pd.concat([initial_df, new_df_row], ignore_index=True)

    df_read = pd.read_csv(raw_csv_path)
    if 'time' in df_read.columns:
        df_read['time'] = pd.to_datetime(df_read['time'])
    if 'time' in expected_combined_df.columns:
        expected_combined_df['time'] = pd.to_datetime(expected_combined_df['time'])
    
    df_read = df_read[expected_combined_df.columns] # Ensure column order for comparison

    pd.testing.assert_frame_equal(df_read, expected_combined_df, check_dtype=False)
    assert not (tmp_path / (raw_csv_path.name + ".bak")).exists()


async def test_append_csv_existing_file_different_columns(tmp_path):
    mock_chk, mock_env, mock_glue, mock_interaction = create_mock_objects()
    raw_csv_path = tmp_path / "diff_cols_data.csv"
    img_name_initial = "initial_img_dc.jpg"
    img_name_new = "new_img_dc.jpg"

    # Initial data with specific columns
    initial_data = {
        'time': [datetime(2023, 1, 1, 10, 0, 0)], 'frame': [1], 'state_0': ['initial_obs_dc'],
        'old_column': ['old_value'] # This column won't be in the new data
    }
    initial_df = pd.DataFrame(initial_data)
    initial_df.to_csv(raw_csv_path, index=False)

    # New data with different columns
    mock_env.time = datetime(2023, 1, 1, 13, 0, 0)
    mock_glue.num_steps = 12
    mock_interaction.o = "obs_dc_2"
    mock_env.last_action = "env_action_dc_2"
    mock_interaction.a = "agent_action_dc_2"
    mock_interaction.r = 0.7
    mock_interaction.extra["df"] = pd.DataFrame({'plant_id': [12], 'frame': [12], 'new_plant_info': ['extra_plant_detail']})
    mock_interaction.extra["some_extra_data"] = 125
    # This new data will generate a DataFrame inside append_csv like:
    # time, frame, state_0, action_0, agent_action, reward, terminal, steps, image_name, return, episode, some_extra_data, plant_id, new_plant_info

    await append_csv(mock_chk, mock_env, mock_glue, raw_csv_path, img_name_new, mock_interaction)

    assert (tmp_path / (raw_csv_path.name + ".bak")).exists() # Check backup
    
    df_read = pd.read_csv(raw_csv_path)
    if 'time' in df_read.columns:
        df_read['time'] = pd.to_datetime(df_read['time'])

    # Expected: initial_df concatenated with the new df. Pandas will fill NaNs.
    # The new DataFrame created by append_csv will have its standard set of columns
    # + columns from interaction.extra["df"]
    new_data_generated_internally = {
        'time': mock_env.time, 'frame': mock_glue.num_steps, 'state_0': mock_interaction.o,
        'action_0': mock_env.last_action, 'agent_action': mock_interaction.a,
        'reward': mock_interaction.r, 'terminal': mock_interaction.t, 'steps': mock_glue.num_steps,
        'image_name': img_name_new, 'return': None, 'episode': mock_chk['episode'],
        'some_extra_data': mock_interaction.extra["some_extra_data"],
        'plant_id': 12, 'new_plant_info': 'extra_plant_detail'
    }
    new_df_row_internal = pd.DataFrame([new_data_generated_internally])
    
    expected_df_after_concat = pd.concat([initial_df, new_df_row_internal], ignore_index=True)
    if 'time' in expected_df_after_concat.columns:
         expected_df_after_concat['time'] = pd.to_datetime(expected_df_after_concat['time'])

    # Sort columns for robust comparison after concat
    df_read = df_read.reindex(sorted(df_read.columns), axis=1)
    expected_df_after_concat = expected_df_after_concat.reindex(sorted(expected_df_after_concat.columns), axis=1)

    pd.testing.assert_frame_equal(df_read, expected_df_after_concat, check_dtype=False)


async def test_append_csv_empty_dataframe_new_file(tmp_path):
    mock_chk, mock_env, mock_glue, mock_interaction = create_mock_objects()
    raw_csv_path = tmp_path / "empty_df_new.csv"
    img_name = "empty_img.jpg"

    # Simulate the df created inside append_csv being empty before merge
    # This means the initial construction results in an empty df
    # For the purpose of this test, we'll directly create an empty df and pass it
    # by ensuring interaction.extra["df"] is empty and other parts are structured to lead to empty.
    # However, append_csv constructs a non-empty base_df then merges.
    # Let's assume the DataFrame *passed into the CSV writing part* is empty.
    # The current append_csv builds a DataFrame with at least one row.
    # To test "empty DataFrame" meaningfully, we might need to assume the *input* to `df.to_csv` is empty.
    # The current structure of append_csv will always write 1 row from the main data,
    # then merges with interaction.extra["df"]. If interaction.extra["df"] is empty, it's like a left merge with an empty right table.

    # Let's refine the test: what if interaction.extra["df"] is empty?
    # The main part of the df (from env, glue, etc.) will still be there.
    mock_interaction.extra["df"] = pd.DataFrame(columns=['plant_id', 'frame']) # Empty but with columns

    await append_csv(mock_chk, mock_env, mock_glue, raw_csv_path, img_name, mock_interaction)
    
    assert raw_csv_path.exists()
    df_read = pd.read_csv(raw_csv_path)
    if 'time' in df_read.columns:
        df_read['time'] = pd.to_datetime(df_read['time'])

    # Expected: one row from main data, NaNs for columns that would come from an empty interaction.extra["df"]
    expected_data = {
        'time': [mock_env.time], 'frame': [mock_glue.num_steps], 'state_0': [mock_interaction.o],
        'action_0': [mock_env.last_action], 'agent_action': [mock_interaction.a],
        'reward': [mock_interaction.r], 'terminal': [mock_interaction.t], 'steps': [mock_glue.num_steps],
        'image_name': [img_name], 'return': [None], 'episode': [mock_chk['episode']],
        'some_extra_data': [mock_interaction.extra["some_extra_data"]],
        'plant_id': [None] # Or whatever merge results in for empty right DF
    }
    # The actual 'plant_id' column might be missing if interaction.extra["df"] was truly empty (no columns)
    # and the merge strategy. Given it has columns, it should result in NaN or similar.
    # If interaction.extra["df"] was pd.DataFrame(), the merge might not add 'plant_id' column.
    # Let's assume interaction.extra["df"] = pd.DataFrame({'plant_id': [], 'frame': []})
    # This will result in NaN for plant_id after the merge if no matching frames, or if it's empty.
    # The current append_csv does a left merge on 'frame'.
    # If mock_interaction.extra["df"] is empty, `plant_id` would be NaN for the new row.
    
    # Re-evaluate `interaction.extra["df"]` for this test:
    mock_interaction_empty_extra_df = mock_interaction
    mock_interaction_empty_extra_df.extra["df"] = pd.DataFrame(columns=['plant_id', 'some_plant_col', 'frame']).astype({'frame':int})

    await append_csv(mock_chk, mock_env, mock_glue, raw_csv_path, img_name, mock_interaction_empty_extra_df)
    
    df_read = pd.read_csv(raw_csv_path) # Re-read after correct call
    if 'time' in df_read.columns:
        df_read['time'] = pd.to_datetime(df_read['time'])

    expected_cols_from_empty_extra = {'plant_id': [None], 'some_plant_col': [None]}
    if not mock_interaction_empty_extra_df.extra["df"]["frame"].dtype == object: # Avoid warning with empty int series
         expected_cols_from_empty_extra = {'plant_id': [pd.NA], 'some_plant_col': [pd.NA]}


    final_expected_data_empty_extra = {
        'time': [mock_env.time], 'frame': [mock_glue.num_steps], 'state_0': [mock_interaction.o],
        'action_0': [mock_env.last_action], 'agent_action': [mock_interaction.a],
        'reward': [mock_interaction.r], 'terminal': [mock_interaction.t], 'steps': [mock_glue.num_steps],
        'image_name': [img_name], 'return': [None], 'episode': [mock_chk['episode']],
        'some_extra_data': [mock_interaction.extra["some_extra_data"]],
        **expected_cols_from_empty_extra
    }
    expected_df = pd.DataFrame(final_expected_data_empty_extra)
    if 'time' in expected_df.columns:
        expected_df['time'] = pd.to_datetime(expected_df['time'])
    
    # df_read might have more columns if interaction.extra["df"] had them, ensure we only check these
    # Or ensure expected_df has all columns that df_read would have.
    # The merge logic is: pd.merge(df, interaction.extra["df"], how="left", on=["frame"])
    # So, all columns from interaction.extra["df"] (except 'frame' if it's only for merge key) will be present.
    
    # Ensure expected_df has all columns that would be created
    all_expected_cols = list(final_expected_data_empty_extra.keys())
    for col in mock_interaction_empty_extra_df.extra["df"].columns:
        if col not in all_expected_cols and col != "frame": # frame is merge key
             all_expected_cols.append(col)
    
    # Recreate expected_df with the right columns and NaNs for those from empty extra["df"]
    complete_expected_data = {**final_expected_data_empty_extra}
    for col in mock_interaction_empty_extra_df.extra["df"].columns:
        if col != "frame" and col not in complete_expected_data : # frame is merge key
            complete_expected_data[col] = [pd.NA] # Or [None] if using older pandas

    expected_df = pd.DataFrame(complete_expected_data)
    if 'time' in expected_df.columns:
        expected_df['time'] = pd.to_datetime(expected_df['time'])


    # Align columns for comparison
    df_read = df_read.reindex(columns=expected_df.columns)

    pd.testing.assert_frame_equal(df_read, expected_df, check_dtype=False, check_ Scienze=False) # check_ Scienze for NaNs

# Note: The "empty dataframe" tests need careful consideration of how an empty DataFrame
# (specifically interaction.extra["df"]) interacts with the merge logic in append_csv.
# The main DataFrame constructed from non-extra sources in append_csv is unlikely to be empty.
# The tests above for "empty_dataframe_new_file" interpret it as interaction.extra["df"] being empty.
# True "empty df to write" (i.e. df.to_csv(empty_df)) is not what append_csv currently does.

# Further tests for empty interaction.extra["df"] with existing files (same/diff cols) would follow a similar pattern:
# 1. Setup initial CSV.
# 2. Call append_csv with mocks where interaction.extra["df"] is empty but has defined columns.
# 3. Assert file content (initial + 1 new row with NaNs for extra_df cols) and backup status.
# This comment serves as a placeholder for those more detailed scenarios if required.
# For now, test_append_csv_empty_dataframe_new_file covers the core "empty extra data" case.
