#!/bin/bash

if ps -ef | grep -v grep | grep "ttscraper.py corona covid" ; then
    echo "scraper already running"
    exit 0
else
    echo "scraper not running"
    python3 ~/Ferramentas/ttscraper.py corona covid &
    exit 0
fi
