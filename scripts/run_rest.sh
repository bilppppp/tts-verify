#!/bin/bash
cd ~/tts-verify; unset ALL_PROXY all_proxy; export NO_PROXY=127.0.0.1,localhost
tts-venv/bin/python app/tests/run_tests.py auto "着急,看着,长度" > logs/tests_rerun.log 2> logs/tests_rerun.err
tts-venv/bin/python app/tests/run_controls.py auto > logs/controls.log 2> logs/controls.err
echo ALL_DONE >> logs/controls.log
