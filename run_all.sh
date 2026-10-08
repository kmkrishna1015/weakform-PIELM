#!/usr/bin/env bash
# Linux / macOS wrapper.  The driver itself is cross-platform: python run_all.py [options]
cd "$(dirname "$0")" && exec "${PYTHON:-python3}" run_all.py "$@"
