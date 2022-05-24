#!/bin/bash

# =============================================================================
DESCRIPTION=$(cat <<-END
[DESCRIPTION]
END
)
# =============================================================================
# @author   : [AUTHOR]
# @date     : [DATE]
# @version  : 1.0
# =============================================================================

script=${0##*/}

if [ $# -ne 2 ]; then
    echo ""
    echo "$DESCRIPTION"
    echo ""
    echo "usage: bash $script IN_PATH OUT_PATH"
    echo "encapsule paths within '' if they contain spaces"
    exit 1
fi

# timestamp function
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

# waiting time spinner animation
# use '&>/dev/null' for suppressing other scripts outputs during animation
spinner() {
    local i sp n
    sp='/-\|'
    n=${#sp}
    printf ' '
    while sleep 0.1; do
        printf "%s\b" "${sp:i++%n:1}"
    done
}
show_spinner() {
    spinner &
}
hide_spinner() {
    kill "$!"
    printf "\b \b"
    wait "$!" 2>/dev/null
}

# file/path info (in_dir, full/reduced file name, ext)
get_file_info() {

    declare -A ret_dict=([in_dir]=$(dirname "$1")
                         [in_name_full]="$(basename -- $1)"
                         [in_ext]="${1#*.}"
                         [in_name]="$(basename -- $1 ${1#*.})")

    echo '('
    for key in  "${!ret_dict[@]}" ; do
        echo "[$key]=${ret_dict[$key]}"
    done
    echo ')'
}

# reads path without '/'
IN_PATH=${1%/}
OUT_PATH=${2%/}

# file info - access fields like ${IN_INFO[in_dir]}
declare -A IN_INFO="$(get_file_info $IN_PATH)"
declare -A OUT_INFO="$(get_file_info $IN_PATH)"
