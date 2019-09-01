#!/bin/sh



if [ $1 ];
then
    echo "Generating Model .."
    erb model.sdf.erb > model.sdf
fi

if [ $2 ];
then
    echo "Launching Model .."
    python launch_hand_model.py
fi
