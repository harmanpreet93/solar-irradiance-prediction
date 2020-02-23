#!/bin/bash
#while :
#do
#  rm -r ../log/*
#  rsync -avzh guest140@helios3.calculquebec.ca:/home/guest140/harman_remote/log /Users/harman/workspace/sem2/advanced_project/Solar-irradiance-Team08-IFT6759
#  sleep 10
#done
echo "Removing logs"
rm -r ../log/*
sleep 5
rsync -avzh guest140@helios3.calculquebec.ca:/home/guest140/harman_remote/log /Users/harman/workspace/sem2/advanced_project/Solar-irradiance-Team08-IFT6759
