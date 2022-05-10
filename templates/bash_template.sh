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

# reads path without '/'
IN_PATH=${1%/}
OUT_PATH=${2%/}

# file info
# IN_DIR=$(dirname "$IN_PATH")
# IN_NAME_FULL="$(basename -- $IN_PATH)"
# IN_EXT="${IN_PATH#*.}"
# IN_NAME="$(basename -- $IN_PATH $IN_EXT)"