#!/bin/bash

DEPLOY=$(./jq-osx-amd64 .'status' ~/TORUS_DL/logs/config.js)
#"up"=deployed, "down"=down
echo $DEPLOY

source $HOME/TORUS_DL/cron/boonlib.sh

#####
# Check to see if the directory exists, and if not then abort
function dir_exist() {
  if [ ! -d $1 ]
  then
    echo "The directory $1 does not exist - aborting"
    exit
  fi
}
#####
# Check to see if the directory exists, and if not create it
function check_dir() {
        if [ ! -d $1 ]; then mkdir -p $1; fi
}
####### - Start While Loop Here! Dont CRON anymore ###
while [[ $DEPLOY == '"up"' ]]
 do
	today=$(date '+%Y%m%d')
	echo "Beginning process..."
	LOGFILE=$HOME/logs/get_dl_data.$(date '+%Y%m%d').log

	ROOT_DIR=/Users/elizabethsmith/TORUS_DL/data

	LIDAR_DIR=${ROOT_DIR}/raw/dl/2019/201905/

	PROC_DIR=${ROOT_DIR}/nonQA_proc/dl/2019/201905/${today}

	dir_exist ${ROOT_DIR}

	#log "************************"
	#log "started"

	check_dir ${LIDAR_DIR}

	check_dir ${PROC_DIR}

	echo "Pulling data from Lidar ..."
	rsync -r -l -D --chmod=ugo=rwX -t -O -i clamps@192.168.50.13:/cygdrive/c/Lidar/Data/Proc/2019/201905/  ${LIDAR_DIR%/}/ > dump.txt

	#processing code
	echo "Processing Lidar data... "
	python process_test.py

	# Plotting code
	python plot_stare.py 
        python plot_vads.py
        python plot_rhi.py
        echo "Sleeping ..."
        sleep 60
	#update staus flag
	DEPLOY=$(./jq-osx-amd64 .'status' ~/TORUS_DL/logs/config.js)
done

while [[ $DEPLOY == '"down"' ]]
 do
       echo "status down!"
       echo "Pulling data from Lidar ..."
        rsync -r -l -D --chmod=ugo=rwX -t -O -i clamps@192.168.50.13:/cygdrive/c/Lidar/Data/Proc/2019/201905/  ${LIDAR_DIR%/}/ > dump.txt

        #processing code
        echo "Processing Lidar data... "
        python process_test.py
        sleep 30
        #update staus flag
        DEPLOY=$(./jq-osx-amd64 .'status' ~/TORUS_DL/logs/config.js)

 done
