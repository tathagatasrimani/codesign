tmux_source() {
    (
        set +e
        source "$1"
        rc=$?
        echo -e "\n[ tmux_source: subshell exited with code $rc ]"
        exit $rc
    )
}