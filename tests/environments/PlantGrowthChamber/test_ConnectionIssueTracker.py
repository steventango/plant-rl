from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from environments.PlantGrowthChamber.PlantGrowthChamber import ConnectionIssueTracker

UTC = ZoneInfo("Etc/UTC")


def t(minute: int) -> datetime:
    return datetime(2024, 1, 1, 0, minute, 0, tzinfo=UTC)


class TestConnectionIssueTracker:
    def test_no_alert_before_threshold(self):
        tracker = ConnectionIssueTracker()
        # 3 consecutive failures (0, 1, 2 minutes) — below 4-minute threshold
        assert tracker.record_failure(t(0)) is False
        assert tracker.record_failure(t(1)) is False
        assert tracker.record_failure(t(2)) is False

    def test_alert_fires_at_threshold(self):
        tracker = ConnectionIssueTracker()
        assert tracker.record_failure(t(0)) is False
        assert tracker.record_failure(t(1)) is False
        assert tracker.record_failure(t(2)) is False
        # 4th minute crosses the 4-minute window → alert
        assert tracker.record_failure(t(4)) is True

    def test_alert_fires_only_once(self):
        tracker = ConnectionIssueTracker()
        tracker.record_failure(t(0))
        tracker.record_failure(t(4))  # alert fires here
        # subsequent failures do not re-alert
        assert tracker.record_failure(t(5)) is False
        assert tracker.record_failure(t(6)) is False

    def test_no_resolution_without_prior_alert(self):
        tracker = ConnectionIssueTracker()
        tracker.record_failure(t(0))
        tracker.record_failure(t(1))
        # less than threshold — no alert was sent
        assert tracker.record_success() is False

    def test_resolution_after_alert(self):
        tracker = ConnectionIssueTracker()
        tracker.record_failure(t(0))
        tracker.record_failure(t(4))  # alert fires
        assert tracker.record_success() is True

    def test_resolution_resets_tracker(self):
        tracker = ConnectionIssueTracker()
        tracker.record_failure(t(0))
        tracker.record_failure(t(4))
        tracker.record_success()
        # After resolution, a new 3-minute window should not alert
        assert tracker.record_failure(t(10)) is False
        assert tracker.record_failure(t(11)) is False
        assert tracker.record_failure(t(12)) is False

    def test_resolution_then_new_alert_cycle(self):
        tracker = ConnectionIssueTracker()
        tracker.record_failure(t(0))
        tracker.record_failure(t(4))
        tracker.record_success()
        # New failure window starts fresh
        assert tracker.record_failure(t(10)) is False
        assert tracker.record_failure(t(14)) is True

    def test_success_without_any_failure(self):
        tracker = ConnectionIssueTracker()
        assert tracker.record_success() is False

    def test_consecutive_successes_do_not_raise(self):
        tracker = ConnectionIssueTracker()
        assert tracker.record_success() is False
        assert tracker.record_success() is False

    def test_exact_threshold_boundary(self):
        tracker = ConnectionIssueTracker()
        tracker.record_failure(t(0))
        # Exactly at 4 minutes (t(0) + 4 min = t(4))
        assert tracker.record_failure(t(4)) is True

    def test_just_under_threshold(self):
        tracker = ConnectionIssueTracker()
        tracker.record_failure(t(0))
        # 3 minutes 59 seconds — just under threshold
        just_under = datetime(2024, 1, 1, 0, 3, 59, tzinfo=UTC)
        assert tracker.record_failure(just_under) is False
