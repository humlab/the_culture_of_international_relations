#!/bin/bash

shopt -s globstar

function clear_output()
{
    fullfile="$1"
    filename=$(basename -- "$fullfile")
    tmpname="/tmp/$filename"

    rm -f $tmpname

    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --log-level=WARN --output=$tmpname $fullfile

    if [ $? -ne 0 ]; then
        echo "fatal: conversion of $fullfile failed"
        exit 64
    fi

    if cmp --silent "$tmpname" "$fullfile"
    then
        echo "info: $fullfile is clean already!"
    else
        cp -f $tmpname $fullfile
        echo "info: $fullfile has been cleared"
    fi

    rm -f $tmpname
}

args=("$@")
n_args=${#args[@]}

if [ $n_args == 1 ] && [ "$args" == "--all" ]; then

    for i in **/*.ipynb; do # Whitespace-safe and recursive
        clear_output "$i"
    done

elif [ $n_args > 0 ]; then

    for (( i=0;i<$n_args;i++)); do

        filename=${args[${i}]}

        if [[ ! $filename =~ ^.*\.ipynb$ ]]; then
            echo "error: file must have an .ipynb extension"
            exit 64
        fi

        if [[ ! -f $filename ]]; then
            echo "error: file $filename does not exist"
            exit 64
        fi

        clear_output "$filename"

    done

else
    echo "usage: $0 [ --all | [FILE]... ]"
    exit 64
fi