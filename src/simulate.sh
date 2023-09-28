#!/bin/sh

while getopts u:n:f: flag
do
    case "${flag}" in
        f) filepath=${OPTARG};; # $1
        n) name=${OPTARG};;
        d) debug=${OPTARG};;
    esac
done

FILE_DIR=./ #/Users/PatrickMcEwen/git_container/codesign/src # change path name for local computer
# arguments like this: ./simulate.sh benchmarks/models/<name> <name>
if [ $filepath ]; then
    # cd $FILE_DIR

    python instrument.py $filepath
    python instrumented_files/xformed-$name > instrumented_files/output.txt
    python simulate.py $filepath $debug
    #cd destiny/config
    #./destiny sample.cfg
fi