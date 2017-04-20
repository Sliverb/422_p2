#!/bin/bash

num=0.0
total=0.0
const=1.0
for ((i=1; i<=50; i++)); do
    var=$( { TIMEFORMAT='%R';time python OAA_timer.py>>/dev/null; } 2>&1 )
    total=$(bc <<< "$total + $var")
    num=$(bc <<< "$num+ $const")
done

echo "OAA average time: ">>time.txt
echo "$(bc -l <<<"$total/$num")">>time.txt


num=0.0
total=0.0
const=1.0
for ((i=1; i<=50; i++)); do
    var=$( { TIMEFORMAT='%R';time python AVA_timer.py>>/dev/null; } 2>&1 )
    total=$(bc <<< "$total + $var")
    num=$(bc <<< "$num+ $const")
done

echo "AVA average time: ">>time.txt
echo "$(bc -l <<<"$total/$num")">>time.txt




