"""Anti-drift tests: the live engine and the keepsake renderer must share the
audio-character constants that define how a performance sounds, so a keepsake
can never come out sounding like a different mix than the live take. If someone
changes one copy and not config.py, these fail.
"""

import pytest

import config
import keepsake
from mrt_controller import PythonMRTController


@pytest.mark.unit
def test_sample_rate_is_single_source():
    assert config.SAMPLE_RATE == 48_000
    assert PythonMRTController.SAMPLE_RATE == config.SAMPLE_RATE
    assert keepsake.SAMPLE_RATE == config.SAMPLE_RATE


@pytest.mark.unit
def test_style_guidance_shared():
    assert PythonMRTController.CFG_BASE == config.STYLE_CFG == keepsake.CFG


@pytest.mark.unit
def test_temperature_shared():
    assert PythonMRTController.TEMPERATURE == config.TEMPERATURE == keepsake.TEMPERATURE


@pytest.mark.unit
def test_decay_watchdog_shared():
    assert PythonMRTController.DECAY_RMS == config.DECAY_RMS == keepsake.DECAY_RMS
    assert PythonMRTController.DECAY_CHUNKS == config.DECAY_CHUNKS == keepsake.DECAY_CHUNKS
