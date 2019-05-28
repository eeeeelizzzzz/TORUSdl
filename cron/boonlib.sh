#!/bin/bash

##########################################################
#
#  BOONLIB - Boonleng's library
#  This is a collection of shell script functions for
#  various tasks. Tasks that I frequently do.
#
#  Copyright (c) 2009-2015 Boon Leng Cheong
#  Advanced Radar Research Center
#  The University of Oklahoma
#
##########################################################

##########################################################
#
#  l o g
#
#     appends the message in LOGFILE
#
#       o  log MESSAGE
#
##########################################################
function log() {
	if [ -z "$1" ]; then return; fi
	if [ -z "$LOGFILE" ]; then
		LOGFILE="$HOME/boonlib.log"
		log $1
		log "LOGFILE undefined. Using LOGFILE=$LOGFILE"
		return
	fi
	log_dir=${LOGFILE%/*}
	if [ "$log_dir" == "$LOGFILE" ]; then log_dir=""; fi
	if [[ ! -z "$log_dir" &&  ! -d "$log_dir" ]]; then
		mkdir -p $log_dir;
	fi
	if [[ ! -z "$DEBUG" && "$DEBUG" == "1" ]]; then
		echo "`date '+%m/%d %r'` : $@"
	fi
	echo "`date '+%m/%d %r'` : $@" >> $LOGFILE
}


##########################################################
#
#  s l o g
#
#     appends the message in LOGFILE, short timestamp
#
#       o  log MESSAGE
#
##########################################################
function slog() {
	if [ -z "$LOGFILE" ]; then
		LOGFILE="$HOME/boonlib.log"
		log $1
		log "LOGFILE undefined. Using LOGFILE=$LOGFILE"
		return
	fi
	log_dir=${LOGFILE%/*}
	if [ "$log_dir" == "$LOGFILE" ]; then log_dir=""; fi
	if [[ ! -z "$log_dir" &&  ! -d "$log_dir" ]]; then
		mkdir -p $log_dir;
	fi
	if [[ ! -z "$DEBUG" && "$DEBUG" == "1" ]]; then
		echo "`date '+%T'` : $@"
	fi
	echo "`date '+%T'` : $@" >> $LOGFILE
}


##########################################################
#
#  n u m 2 s t r 3
#
#     converts a string to a 3-digit groups with comma
#
#       o  num2str3 1234567890
#
##########################################################
function num2str3() {
	len=${#1}
	str3=""
	dig=0
	while [ "$len" -gt 0 ]; do
		len=$((len-1))
		str3="${1:$len:1}$str3"
		dig=$((dig+1))
		if [[ "$dig" -eq "3" && "$len" -gt 0 ]]; then
			dig=0
			str3=",$str3"
		fi
	done
	echo "$str3"
}


##########################################################
#
#  f i l e _ m a n a g e r
#
#     frees up space or limits the usage until the targeted
#     number is achieved
#
#       o  file_manager FREE/LIMIT PATH TARGET_BYTE [TOLERANCE]
#
##########################################################
function file_manager() {
	log "BOONLIB FILE_MANAGER -- initiated by $USER"
	if [ "$#" -lt "3" ]; then
		log "Need at least three arguments: Method, path, target space, (tolerance) [$#]"
		return
	fi
	method="$1"
	if [[ "$method" != "FREE" && "$method" != "LIMIT" ]]; then
		log "Method can only be FREE or LIMIT"
		return
	fi
	target_path="$2"
	if [ -z "$target_path" ]; then
		log "Target folder not specified."
		return
	fi		
	if [ ! -d "$target_path" ]; then
		log "Target folder does not exist."
		return
	fi
	target_space="$3"
	if [ -z "$target_space" ]; then
		log "Target space empty?"
		return
	fi
	tolerance="$4"
	if [ -z "$tolerance" ]; then tolerance=$((target_space/100)); fi
	log "Target folder: $target_path"
	if [ "$method" == "FREE" ]; then
		log "Target free > `num2str3 $((target_space/1024/1024))` MB"
		# NOTE: df returns free space in 1K blocks
		# Example, 1TB = 1024 GB = 1024*1024*1024 K-blocks
		#current=`df $target_path | tail -1 | awk {'print $4'}`
		current=`df $target_path | tail -1 | awk '{printf $(NF-2)}'`
	elif [ "$method" == "LIMIT" ]; then
		log "Target limit < `num2str3 $((target_space/1024/1024))` MB"
		# NOTE: du returns free space in 1K blocks
		# Example, 1TB = 1024 GB = 1024*1024*1024 K-blocks
		current=`nice -n 20 du -k -s $target_path/ | awk {'print $1'}`
	else
		log "This should not happened."
		return
	fi
	# Size in 1-K blocks
	target_space=$((target_space/1024))
	if [ "$current" -eq "$target_space" ]; then
		ineq="="
		echo "You hit the equal sign" | mail -s "Jackpot! from free_or_limit()" boonleng@ou.edu
	elif [ "$current" -gt "$target_space" ]; then
		ineq=">"
	elif [ "$current" -lt "$target_space" ]; then
		ineq="<"
	else
		ineq="?"
	fi
	log "Current: `num2str3 $((current/1024))` MB $ineq `num2str3 $((target_space/1024))` MB"
	num=0;
	# Get size to be in 1K-block
	#find -L $target_path -maxdepth 3 -type f -printf '%p,%k\n' | sort | while read line; do
	#	file=${line%%,*}
	#	size=${line##*,}
	nice -n 20 find $target_path -maxdepth 3 -type f -print | sort | while read line; do
		# file=${line%%,*}
		# size=${line##*,}
		file=$line
		# file size in 512 block
		size=`ls -s $file | awk {'print $1'}`
		# file size in 1K-block
		size=$((size/2))
		if [[ "$method" == "FREE" && "$current" -gt "$target_space" ]] || 
		   [[ "$method" == "LIMIT" && "$current" -lt "$target_space" ]]; then
			# At this point the condition is met
			if [ "$num" -gt 0 ]; then
				log "Erased $num file(s)"
				if [ "$method" == "FREE" ]; then
					log "Estimated free space: `num2str3 $((current/1024))` MB"
					current=`df $target_path | tail -1 | awk {'print $4'}`
					log "Actual free space: `num2str3 $((current/1024))` MB"
				else
					log "Estimated usage: `num2str3 $((current/1024))` MB"
					current=`nice -n 20 du -k -s $target_path/ | awk {'print $1'}`
					log "Actual usage: `num2str3 $((current/1024))` MB"
				fi
				if [[ "$method" == "FREE" && "$current" -lt "$((target_space-tolerance))" ]] || 
				   [[ "$method" == "LIMIT" && "$current" -gt "$((target_space+tolerance))" ]]; then
					log "Warning! Beyond tolerance. (`num2str3 $((tolerance/1024))` MB)"
					# Find the line with ----. That's where the log last started.
					n=`tail -n 300 $LOGFILE | grep -n -e "----" | tail -n 1`
					n=${n%%:*}
					tail -n $((300-n+1)) $LOGFILE
				fi
			else
				log "Nothing to erase"
			fi
			break
		else
			num=$((num+1))
			cmd="nice -n 20 rm -f $file"
			if [ "$method" == "FREE" ]; then
				current=$((current+size))
			else
				current=$((current-size))
			fi
			eval $cmd
			log $cmd,$current
		fi
	done
}

##########################################################
#
#  r e m o v e _ f i l e s
#
#     removes files in a specific folder until a targeted free space is achieved
#
#       o  remove_files PATH TARGET_FREE_SPACE TOLERANCE
#
##########################################################
function remove_files() {
	if [ "$#" -lt 2 ]; then
		log "Need at least two arguments: Target path and target free space."
		return;
	fi
	log "remove_files() -- depreciated --> file_manager() -- $USER"
	file_manager "FREE" "$1" "$2" "$((10*1024*1024))"
}

##########################################################
#
#  l i m i t _ u s a g e
#
#     removes files in a specific folder until a targeted usage
#
#       o  limit_usage PATH TARGETED_USAGE
#
##########################################################
function limit_usage() {
	if [ "$#" -lt 2 ]; then
		log "Need at least two arguments: Target path and target free space."
		return;
	fi
	log "limit_usage() -- deprecated --> file_manager() -- $USER"
	file_manager "LIMIT" "$1" "$2" "$((10*1024*1024))"
}


##########################################################
#
#  r e m o v e _ f i l e s _ b u t _ k e e p 
#  r e m o v e _ f o l d e r s _ b u t _ k e e p 
#
#     removes all files / folders in a folder but keep the last N files. The 
#     files / folders will be sorted alphabetically first before the last N
#     entries are determined.
#
#       o   erase_files_but_keep DIR NUM PATTERN
#       o   erase_folders_but_keep DIR NUM PATTERN
#
#     e.g.,
#
#     remove_files_but_keep './log' 3 '*.log'
#
#     erases all log files but keep the latest 3
#
##########################################################
function remove_files_but_keep() {
	remove_but_keep f $1 $2 $3
}

function remove_folders_but_keep() {
	remove_but_keep d $1 $2 $3
}

function remove_but_keep() {
	TYP=$1; if [ -z "${TYP}" ]; then
		echo "Must supply a mode"
		return
	fi
	DIR=$2; if [ -z "${DIR}" ]; then DIR="./"; fi
	NUM=$3; if [ -z "${NUM}" ]; then NUM=1000; fi
	PAT=$4; if [ -z "${PAT}" ]; then PAT='*'; fi
	log "remove_but_keep() -- $USER"
	log "T:${TYP}  D:${DIR}  N:${NUM}  P:${PAT}"
	count=0;
	while read f; do
		rm -rf $f
		log "Removed $f"
		count=$((count+1))
	done < <(find "${DIR}" -mindepth 1 -maxdepth 1 -type "${TYP}" -name "${PAT}" | sort |
		sed -n -e :a -e "1,${NUM}!{P;N;D;};N;ba")
	if [ "${count}" -gt 0 ]; then
		if [ "${TYP}" == "d" ]; then
			log "Removed ${count} folder(s)."
		else
			log "Removed ${count} files(s)."
		fi
	else
		log "Nothing removed."
	fi
}


##########################################################
#
#  r e m o v e _ e m p t y _ f o l d e r s
#
#     removes empty folders
#
#       o    remove_empty_dir DIR
#
##########################################################
function remove_empty_folders() {
	log "remove_empty_folders -- ${1} -- ${USER}"
	log "`find -L ${1} -depth -type d -empty`"
	find -L ${1} -depth -type d -empty -exec rmdir '{}' \;
}


##########################################################
#
#  w a r n _ i f _ l o w
#
#     generates a warning message if / is running low in space
#
#       o	warn_if_low DIR LOW_IN_BYTE
#
##########################################################
function warn_if_low() {
	DIR=$1; if [ -z "$DIR" ]; then DIR="/"; fi
	LOW=$2; if [ -z "$LOW" ]; then LOW="$((5*1024*1024*1024))"; fi
	LOW_KB=$((LOW/1024))
	LOW_MB=$((LOW/1024/1024))
	LOW_GB=$((LOW/1024/1024/1024))
	space=`df -k $DIR | tail -1 | awk '{printf $(NF-2)}'`
	if [ "$space" -lt "$LOW_KB" ]; then
		echo "Low in space for $DIR"
		if [ "$LOW_GB" -gt 0 ]; then
			echo "Available: $((space/1024/1024)) GB < $LOW_GB GB."
		else
			echo "Available: $((space/1024)) < $LOW_MB MB."
		fi
	fi
}


##########################################################
#
#  c h e c k _ p r o c e s s
#
#     checks for processes using ps and grep
#
#       o	check_process PROCESS_1 PROCESS_2 PROCESS_3 ...
#
##########################################################
function check_process() {
	if [ "$#" == "0" ]; then
		echo "No process to check?"
		return
	fi
	arg="\($1"
	for ((i=2;i<=$#;i++)); do
		cmd="c=\${${i}}"
		eval $cmd
		arg="$arg\|$c"
	done
	arg="${arg}\)"
	#echo "$arg"
	#ps x -u $USER -o pid,stat,cmd | grep -e "$arg" | grep -v "\(grep\|tail\|ssh\|bin/su\)"
	if [ `uname` == "Darwin" ]; then
		keywords="ruser,pid,stat,pcpu,pmem,command";
	else
		keywords="ruser,pid,stat,pcpu,pmem,nlwp,ucomm";
	fi
	ps ax -o $keywords | head -1
	ps ax -o $keywords | grep -e "$USER" | grep -e "$arg" | grep -v "\(grep\|tail\|ssh\|bin/su\)"
}


##########################################################
#
#  f e c h o
#
#     fills the 78th character in that line with |
#
#       o	fecho WHAT_YOU_WANT_TO_PRINT
#
##########################################################
function fecho() {
	str="$@"
	str="${str:0:78}"
	printf "%-78s|\n" "$str"
}


##########################################################
#
#  t e x t o u t
#
#     prints out the text with color
#
#       o	echo SOMETHING | textout TITLE COLOR
#       o   cat SOME_FILE | textout TITLE COLOR
#
##########################################################
function textout() {
	#   tput bold
	#   tput setab 4
	if [ "${#}" -ge 2 ]; then
#       tput setaf $2
		case "$2" in
			red)     c=1;;
			green)   c=2;;
			yellow)  c=3;;
			blue)    c=4;;
			magenta) c=5;;
			cyan)    c=6;;
			white)   c=7;;
			*)       c=$2;;
		esac
		echo -ne "\033[3${c}m"
	fi
	len=${#1}
	if [ "$#" -ge 1 ]; then
		LINE="============================================"
		fecho "${1}"
		fecho "${LINE:0:$len}"
	fi
	cat - | while read line; do
		fecho "${line}"
	done
	#tput sgr0
}


##########################################################
#
#  h e a d _ t a i l
#
#     shows the head and tail portions of a file list of a folder
#
#       o	head_tail DIR [COLOR]
#
##########################################################
function headtail() {
	DIR=${1}
	COLOR=${2}
	if [ -z ${COLOR} ]; then COLOR=yellow; fi
	if [ -d ${DIR} ]; then
		NUM=`ls ${DIR}/ | wc -l`
		SPACE=$(du -hs ${DIR} | awk {'print $1'})
		ls -lh ${DIR}/ | head -n 4 | tail -n 3 | textout "${DIR} (${NUM} --> ${SPACE})" ${COLOR}
		fecho ":          : :    :      :         :          :     :"
		ls -lh ${DIR}/ | tail -n 4 | textout
	fi
	tput sgr0
}


##########################################################
#
#  r e m o v e _ m i n u t e s _ o l d _ f i l e s
#
#     removes files older than N minutes
#
#       o	remove_old_files DIR MINUTES PATTERN
#
##########################################################
function remove_minutes_old_files() {
	DIR=${1}; if [ -z "${DIR}" ]; then DIR="./"; fi
	NUM=${2}; if [ -z "${NUM}" ]; then NUM=1440; fi
	PAT=${3}; if [ -z "${PAT}" ]; then PAT='*.tgz'; fi
	log "remove_old_files() -- ${USER}"
	log "HOME:${DIR}  MIN:${NUM}  PAT:${PAT}"
	count=0
	while read f; do
		rm -f ${f}
		log "Removed ${f}"
		count=$((count+1))
	done < <(find ${DIR} -maxdepth 2 -type f -mmin "+${NUM}" -name "${PAT}" | sort)
	if [ "$count" -gt 0 ]; then
		log "Removed $count file(s)."
	else
		log "Nothing removed."
	fi
}
