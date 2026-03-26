#!/usr/bin/env python3
"""Run lightweight reproducibility smoke checks."""

from __future__ import annotations

import unittest


def main() -> int:
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_reproducibility_smoke.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
