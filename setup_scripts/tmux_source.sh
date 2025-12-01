## To use: source this script and call tmux_source with the path to the script to be sourced.
## e.g. From codesign root directory:
##      source setup_scripts/tmux_source.sh
##      tmux_source full_env_start.sh

tmux_source() {
    (
        set +e
        source "$1"
        rc=$?
        echo -e "\n[ tmux_source: subshell exited with code $rc ]"
        exit $rc
    )
}