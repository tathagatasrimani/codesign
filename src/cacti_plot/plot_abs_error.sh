#!/bin/sh

cd ..

# Default values
cfg_name="mem_validate_cache"
dat_file="cacti/tech_params/90nm.dat"
cacheSize=131072
blockSize=64
cacheType="main memory"
busWidth=64

# Parse command-line arguments
while getopts "c:d:s:b:t:w:h" opt; do
  case $opt in
    c) cfg_name=$OPTARG ;;
    d) dat_file=$OPTARG ;;
    s) cacheSize=$OPTARG ;;
    b) blockSize=$OPTARG ;;
    t) cacheType=$OPTARG ;;
    w) busWidth=$OPTARG ;;
    h)
      echo "Usage: $0 [-c cfg_name] [-d dat_file] [-s cacheSize] [-b blockSize] [-t cacheType] [-w busWidth]"
      exit 0 ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1 ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1 ;;
  esac
done

# Call the Python script with the provided or default arguments
python cacti_util.py \
  -cfg_name "$cfg_name" \
  -dat_file "$dat_file" \
  -cacheSize "$cacheSize" \
  -blockSize "$blockSize" \
  -cacheType "$cacheType" \
  -busWidth "$busWidth"

cd cacti_plot
python cacti_plot.py
