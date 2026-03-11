import subprocess
import sys
import os
import numpy as np
import neo
import quantities as pq
from pathlib import Path

# Add the script directory to path to allow direct import
SCRIPT_DIR = str(Path(__file__).parent.parent / "pipeline" / "stage02_processing" / "scripts")
sys.path.append(SCRIPT_DIR)
# Also add the pipeline directory to path so that 'utils' can be imported by the script
PIPELINE_DIR = str(Path(__file__).parent.parent / "pipeline")
sys.path.append(PIPELINE_DIR)

import background_subtraction

# Helper functions
def create_mock_data(path):
    # Create a simple neo block with an analogsignal
    ch_count = 4
    sample_count = 100
    # Mean of each channel will be its index + 1
    data = np.random.randn(sample_count, ch_count) + np.arange(1, ch_count + 1)
    
    asig = neo.AnalogSignal(data, units='mV', sampling_rate=1000*pq.Hz)
    asig.array_annotations = {
        'x_coords': np.arange(ch_count),
        'y_coords': np.zeros(ch_count)
    }
    asig.description = "Test Signal"
    
    segment = neo.Segment()
    segment.analogsignals.append(asig)
    
    block = neo.Block()
    block.segments.append(segment)
    
    writer = neo.io.get_io(str(path))
    writer.write(block)
    writer.close()
    return data

# Function tests

def test_background_subtraction_shape_frame(tmp_path):
    # Build a small test array
    # For the resulting matrix, dim 1 is the x-axis, dim 2 is the y-axis
    # (i.e., the order is reversed from Python indexing)
    xy_coords = np.array([[0,0], [1,0], [0,1], [1,1]])
    values = np.array([1, 2, 3, 4])

    frame = background_subtraction.shape_frame(values, xy_coords)
    
    assert frame.shape == (2, 2)
    assert frame[0, 0] == 1
    assert frame[0, 1] == 2
    assert frame[1, 0] == 3
    assert frame[1, 1] == 4

# CLI tests

def test_background_subtraction_cli(tmp_path):
    # Perform a run of the script using example data and check the output files
    input_file = tmp_path / "input.nix"
    output_file = tmp_path / "output.nix"
    output_file_image = tmp_path / "output.png"
    output_file_array = tmp_path / "output.npy"

    original_data = create_mock_data(input_file)
    
    # Call the script
    script_path = Path(SCRIPT_DIR) / "background_subtraction.py"
    
    # We need to set PYTHONPATH so the script can find 'utils'
    env = os.environ.copy()
    env["PYTHONPATH"] = PIPELINE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable,
        str(script_path),
        "--data", str(input_file),
        "--output", str(output_file),
        "--output_img", str(output_file_image),
        "--output_array", str(output_file_array),
    ]

    # Run script check successful execution
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # Verify output neo file
    reader = neo.io.get_io(str(output_file))
    block = reader.read_block()
    reader.close()

    new_asig = block.segments[0].analogsignals[0]
    processed_data = new_asig.as_array()

    # Check that mean was subtracted
    # original_data has means approx [1, 2, 3, 4]
    # processed_data should have means close to 0
    np.testing.assert_allclose(np.nanmean(processed_data, axis=0), 0, atol=1e-7)

    # Check if description was updated
    assert "The mean of each channel was subtracted" in new_asig.description

    # Check output image was created
    assert output_file_image.exists()
    assert output_file_image.stat().st_size > 0

    # Check output array file was created and contains the expected background frame
    assert output_file_array.exists()

    saved_frame = np.load(output_file_array)

    # Build expectation from scratch (without shape_frame())
    expected_background = np.nanmean(original_data, axis=0)
    expected_xy_coords = np.column_stack(
        (np.arange(original_data.shape[1]), np.zeros(original_data.shape[1], dtype=int))
    )
    expected_frame = background_subtraction.shape_frame(expected_background, expected_xy_coords)

    np.testing.assert_allclose(saved_frame, expected_frame, atol=1e-7)
