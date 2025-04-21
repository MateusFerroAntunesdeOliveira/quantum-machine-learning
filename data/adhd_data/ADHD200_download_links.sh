#! /bin/bash
# 
# Script to download files for the ADHD200 study
# 
# For this script to work you will need to have AWS's CLI tool installed. 
# More information here: https://aws.amazon.com/cli/
#
# Example usage of the script:
# bash ADHD200_download_links.sh -i ADHD200_site-<site>_list.csv -o FOLDER
#
# Created by Alexandre Franco (alexandre.franco@nki.rfmh.org)
# Feb 2023
#
# Last modified by Nathalia Esper (nathalia.esper@childmind.org) - Aug 2023


while getopts o:i: flag
do
	case ${flag} in
		o) output=${OPTARG};;
		i) input=${OPTARG};;
	esac
done

if [[ ! -d ${output} ]]; then
	echo Creating output directory...
	mkdir -p ${output}
fi

while IFS=, read -r filepath
do 
#	echo ${filepath}
	exactPath=${filepath#*/*/*/*/*/*/*/}
	outpath=${output}/${exactPath}
	echo ${outpath}
	if [ "${exactPath}" == "" ]; then
		echo ${filepath} ignored because it is a directory
	else
		aws s3 cp ${filepath} ${output}/${exactPath} --no-sign-request
		# Add this flag if running into firewall issues at your institution --no-verify-ssl
	fi
done < ${input}