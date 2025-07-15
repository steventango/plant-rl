"""Pytest configuration and shared fixtures."""

import os
from unittest.mock import MagicMock

import pytest
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

from utils.checkpoint import Checkpoint


@pytest.fixture
def setup_checkpoint_test():
    """
    A fixture that returns a function to set up a checkpoint test environment for an agent.

    The returned function accepts the following parameters:
    - tmpdir: The temporary directory to use for the test
    - params: Dictionary of parameters to pass to the agent
    - agent_class: The agent class to test
    - observations: Observation shape tuple (default: (1,))
    - actions: Number of actions (default: 6)
    - collector: Data collector mock (default: None)
    - init_func: Optional function to initialize agent state before saving (takes agent as argument)
    - **kwargs: Additional arguments to pass to the agent constructor

    Returns:
    - original_agent: The original agent
    - loaded_agent: The loaded agent
    """

    def _setup_checkpoint_test(
        tmpdir,
        params,
        agent_class,
        observations=(1,),
        actions=6,
        collector=None,
        init_func=None,
        **kwargs,
    ):
        tmp_dir = str(tmpdir)
        seed = params.get("seed", 123)

        # Create directory structure
        checkpoint_dir = os.path.join(tmp_dir, "0")
        os.makedirs(checkpoint_dir, exist_ok=True)
        params_file = os.path.join(checkpoint_dir, "params.json")

        # Set up context mock
        mock_ctx = MagicMock()
        mock_ctx.resolve.side_effect = lambda path: os.path.join(tmp_dir, path)
        mock_ctx.exists.return_value = True
        mock_ctx.ensureExists.side_effect = lambda path, is_file: os.path.join(
            tmp_dir, path
        )

        # Create a mock experiment description
        mock_exp = MagicMock(spec=ExperimentDescription)
        mock_exp.getPermutation.return_value = {**params, "seed": seed}
        mock_exp.buildSaveContext.return_value = mock_ctx

        # Initialize a checkpoint with the mock experiment
        chk = Checkpoint(mock_exp, 0, base_path=tmp_dir)

        # Manually write params file that checkpoint expects
        os.makedirs(os.path.dirname(params_file), exist_ok=True)
        with open(params_file, "w") as f:
            import json

            json.dump({**params, "seed": seed}, f)

        # Create the original agent
        original_agent = agent_class(
            observations=observations,
            actions=actions,
            params=params,
            collector=collector,
            seed=seed,
            **kwargs,
        )

        # Initialize agent state if provided
        if init_func:
            init_func(original_agent)

        # Store the agent in the checkpoint
        chk["a"] = original_agent

        # Save the checkpoint
        chk.save()

        # Create a new checkpoint and load
        new_chk = Checkpoint(mock_exp, 0, base_path=tmp_dir)
        loaded = new_chk.load()

        # Verify checkpoint loaded successfully
        assert loaded, "Failed to load checkpoint"

        # Get the loaded agent
        loaded_agent = new_chk["a"]

        return original_agent, loaded_agent

    return _setup_checkpoint_test
