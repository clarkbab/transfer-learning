#! /usr/bin/env bash

DESKTOP="A40782"
RESEARCH_ENV_IP="10.126.18.137"
RESEARCH_ENV_PORT="11112"

# TODO: Warn if not running after hours.
# TODO: RUN IN BATCHES. Take folder name, and count items then upload only in batches.
storescu -v -aet "DCMTK_$DESKTOP" -aec VMSDBDRES $RESEARCH_ENV_IP $RESEARCH_ENV_PORT $1
