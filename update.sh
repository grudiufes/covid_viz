#!/bin/bash

# this is supposed to be executed by cron once a week
# you can run it manually and may or may not specify a date in the format "dd/mm/aaaa"

cd ~/Dados/covid_viz
python3 updater.py $1
git add .
git add boletins/*
git commit -m "Update visualization"
git push
