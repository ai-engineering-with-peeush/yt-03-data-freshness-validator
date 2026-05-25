"""
test_freshness_validator.py
Tests for freshness_validator.py — run with: python -m unittest tests/test_freshness_validator.py
"""

import sys
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from freshness_validator import DataSource, FreshnessResult, FreshnessValidator


def make_source(name, last_updated_offset, expected_freshness=None, tz_aware=True):
    """Build a DataSource whose last_updated is `last_updated_offset` before now."""
    if expected_freshness is None:
        expected_freshness = timedelta(hours=1)
    now = datetime.now(tz=timezone.utc)
    last_updated = now - last_updated_offset
    if not tz_aware:
        last_updated = last_updated.replace(tzinfo=None)
    return DataSource(
        name=name,
        expected_freshness=expected_freshness,
        get_last_updated=lambda: last_updated,
    )


class TestFreshData(unittest.TestCase):

    def test_is_fresh_true_when_within_threshold(self):
        source = make_source("events_api", timedelta(minutes=30))
        result = FreshnessValidator().check(source)
        self.assertTrue(result.is_fresh)

    def test_staleness_is_zero_when_fresh(self):
        source = make_source("events_api", timedelta(minutes=30))
        result = FreshnessValidator().check(source)
        self.assertEqual(result.staleness, timedelta(0))

    def test_no_alert_fired_when_fresh(self):
        alert = MagicMock()
        source = make_source("events_api", timedelta(minutes=30))
        FreshnessValidator(alert_fn=alert).check(source)
        alert.assert_not_called()

    def test_message_contains_fresh_label(self):
        source = make_source("events_api", timedelta(minutes=30))
        result = FreshnessValidator().check(source)
        self.assertIn("[FRESH]", result.message)


class TestStaleData(unittest.TestCase):

    def test_is_fresh_false_when_past_threshold(self):
        source = make_source("events_api", timedelta(hours=2))
        result = FreshnessValidator().check(source)
        self.assertFalse(result.is_fresh)

    def test_alert_fired_when_stale(self):
        alert = MagicMock()
        source = make_source("events_api", timedelta(hours=2))
        FreshnessValidator(alert_fn=alert).check(source)
        alert.assert_called_once()

    def test_alert_receives_correct_result(self):
        alert = MagicMock()
        source = make_source("events_api", timedelta(hours=2))
        FreshnessValidator(alert_fn=alert).check(source)
        result_passed = alert.call_args[0][0]
        self.assertIsInstance(result_passed, FreshnessResult)
        self.assertEqual(result_passed.source_name, "events_api")
        self.assertFalse(result_passed.is_fresh)

    def test_staleness_duration_is_correct(self):
        # staleness = age (total time since last update), not age - threshold
        source = make_source("events_api", timedelta(hours=3), expected_freshness=timedelta(hours=1))
        result = FreshnessValidator().check(source)
        self.assertGreater(result.staleness, timedelta(hours=2, seconds=55))
        self.assertLess(result.staleness, timedelta(hours=3, seconds=5))

    def test_message_contains_stale_label(self):
        source = make_source("events_api", timedelta(hours=2))
        result = FreshnessValidator().check(source)
        self.assertIn("[STALE]", result.message)


class TestBoundary(unittest.TestCase):

    def test_data_at_exact_threshold_is_fresh(self):
        now = datetime.now(tz=timezone.utc)
        threshold = timedelta(hours=1)
        last_updated = now - threshold + timedelta(milliseconds=100)
        source = DataSource(
            name="boundary_test",
            expected_freshness=threshold,
            get_last_updated=lambda: last_updated,
        )
        result = FreshnessValidator().check(source)
        self.assertTrue(result.is_fresh)

    def test_data_one_second_past_threshold_is_stale(self):
        now = datetime.now(tz=timezone.utc)
        threshold = timedelta(hours=1)
        last_updated = now - threshold - timedelta(seconds=1)
        source = DataSource(
            name="boundary_test",
            expected_freshness=threshold,
            get_last_updated=lambda: last_updated,
        )
        result = FreshnessValidator().check(source)
        self.assertFalse(result.is_fresh)

    def test_timezone_naive_datetime_treated_as_utc(self):
        source = make_source("naive_tz_source", timedelta(minutes=10), tz_aware=False)
        result = FreshnessValidator().check(source)
        self.assertTrue(result.is_fresh)


class TestCheckAll(unittest.TestCase):

    def setUp(self):
        self.alert = MagicMock()
        self.validator = FreshnessValidator(alert_fn=self.alert)

    def test_returns_result_for_each_source(self):
        sources = [
            make_source("a", timedelta(minutes=10)),
            make_source("b", timedelta(minutes=20)),
            make_source("c", timedelta(hours=3)),
        ]
        results = self.validator.check_all(sources)
        self.assertEqual(len(results), 3)

    def test_only_stale_sources_trigger_alert(self):
        sources = [
            make_source("fresh_source", timedelta(minutes=10)),
            make_source("stale_source", timedelta(hours=3)),
        ]
        self.validator.check_all(sources)
        self.assertEqual(self.alert.call_count, 1)
        alerted_name = self.alert.call_args[0][0].source_name
        self.assertEqual(alerted_name, "stale_source")

    def test_results_preserve_source_order(self):
        sources = [
            make_source("alpha", timedelta(minutes=5)),
            make_source("beta", timedelta(hours=2)),
            make_source("gamma", timedelta(minutes=45)),
        ]
        results = self.validator.check_all(sources)
        self.assertEqual([r.source_name for r in results], ["alpha", "beta", "gamma"])

    def test_all_fresh_returns_true_when_all_pass(self):
        sources = [
            make_source("a", timedelta(minutes=5)),
            make_source("b", timedelta(minutes=10)),
        ]
        self.assertTrue(self.validator.all_fresh(sources))

    def test_all_fresh_returns_false_when_any_stale(self):
        sources = [
            make_source("a", timedelta(minutes=5)),
            make_source("b", timedelta(hours=5)),
        ]
        self.assertFalse(self.validator.all_fresh(sources))


class TestCheckAllWithErrors(unittest.TestCase):

    def test_failing_source_treated_as_stale(self):
        alert = MagicMock()
        validator = FreshnessValidator(alert_fn=alert)

        def boom():
            raise ConnectionError("API is down")

        source = DataSource(name="broken_api", expected_freshness=timedelta(hours=1), get_last_updated=boom)
        results = validator.check_all([source])
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].is_fresh)

    def test_error_in_one_source_does_not_skip_others(self):
        alert = MagicMock()
        validator = FreshnessValidator(alert_fn=alert)

        def boom():
            raise ConnectionError("API is down")

        sources = [
            DataSource(name="broken_api", expected_freshness=timedelta(hours=1), get_last_updated=boom),
            make_source("healthy_api", timedelta(minutes=5)),
        ]
        results = validator.check_all(sources)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].is_fresh)
        self.assertTrue(results[1].is_fresh)

    def test_error_message_contains_error_label(self):
        alert = MagicMock()
        validator = FreshnessValidator(alert_fn=alert)

        def boom():
            raise RuntimeError("timeout")

        source = DataSource(name="broken_api", expected_freshness=timedelta(hours=1), get_last_updated=boom)
        results = validator.check_all([source])
        self.assertIn("[ERROR]", results[0].message)


class TestCustomAlertCallback(unittest.TestCase):

    def test_custom_alert_receives_freshnessresult(self):
        received = []

        def my_alert(result):
            received.append(result)

        source = make_source("stale_source", timedelta(hours=5))
        FreshnessValidator(alert_fn=my_alert).check(source)
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], FreshnessResult)

    def test_custom_alert_not_called_for_fresh_data(self):
        received = []

        def my_alert(result):
            received.append(result)

        source = make_source("fresh_source", timedelta(minutes=5))
        FreshnessValidator(alert_fn=my_alert).check(source)
        self.assertEqual(len(received), 0)


if __name__ == "__main__":
    unittest.main()
