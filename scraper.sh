#!/bin/bash

if [[ -z $(ps -ef | grep -v grep | grep "ttscraper.py corona covid") ]]; then
    xfce4-terminal -e "python3 $HOME/Ferramentas/ttscraper.py corona covid" 
fi
