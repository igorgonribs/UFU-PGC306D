#!/bin/bash

# Script para rodar todos os esperimentos

cd $(dirname $0) ; CWD=$(pwd)

format_time() {
  local total_seconds=$1
  local hours=$((total_seconds / 3600))
  local minutes=$(( (total_seconds % 3600) / 60 ))
  local seconds=$(( total_seconds % 60 ))
  printf "%02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
}

LOG_FILE="Execution.log"
log_message() {
  #echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

TOTAL_START_TIME=$SECONDS

START_TIME=$SECONDS
./MLP_01.py
END_TIME=$SECONDS
DURATION=$((END_TIME - START_TIME))
log_message "MLP_01.py Execution time: $(format_time $DURATION)"
sleep 5

START_TIME=$SECONDS
./MLP_02.py
END_TIME=$SECONDS
DURATION=$((END_TIME - START_TIME))
log_message "MLP_02.py Execution time: $(format_time $DURATION)"
sleep 5

START_TIME=$SECONDS
./MLP_03.py
END_TIME=$SECONDS
DURATION=$((END_TIME - START_TIME))
log_message "MLP_03.py Execution time: $(format_time $DURATION)"
sleep 5

START_TIME=$SECONDS
./MLP_04.py
END_TIME=$SECONDS
DURATION=$((END_TIME - START_TIME))
log_message "MLP_04.py Execution time: $(format_time $DURATION)"
sleep 5

START_TIME=$SECONDS
./CNN_05.py
END_TIME=$SECONDS
DURATION=$((END_TIME - START_TIME))
log_message "CNN_05.py Execution time: $(format_time $DURATION)"
sleep 5




TOTAL_END_TIME=$SECONDS
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
log_message "TOTAL Execution time: $(format_time $TOTAL_DURATION)"
exit 0
