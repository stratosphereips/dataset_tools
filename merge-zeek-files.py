#!/usr/bin/env python3
# Tool to merge two zeek logs files by 'inserting' one into the other and moving them in time

# Author: Sebastian Garcia, eldraco@gmail.com
# Stratosphere Laboratory, Czech Technical University in Prague
# www.stratosphereips.org

import argparse
from datetime import datetime
import os
from os import listdir
from os.path import isfile, join

# How it works
# - Read the folders with files. The first folder has the 'base' files. The second has the files to be inserted in the first.
# - Create the temp folder and the temp files
# - Merge the conn file according to the rules of merging
# - While merging the conn.log file, for each flow also merge the rest of the alternative flows from the other files
# - For each of the files in the second folder do
# - Find the uid in the conn.log file
# - Search the corresponding alternative flows (ssl, ssh, etc.)
# - Move the alternative flows to the same time as the conn.log flow
# - For the files.log and x509.log files, that don't have a uid, first read those

def merge_conn_files(firstfile, secondfile):
    """
    Insert the second zeek log file into the first zeek file.
    Steps
    1. Read the time of the first flow of the first file
    1. Read the time of the last flow of the first file
    2. Read the time of the first flow of the second file
    1. Read the time of the last flow of the second file
    3. Compute the difference
    4. Compute the new start of the second file: diff - time asked. So if diff = 4hs and time asked to move is 2hs, then we need to move -2hs
    5. Create new output file with the content of the first file
    6. Add lines of the second file to the output file (out of order) but only if the are < than the time of the last flow of the first file
    7. Sort the output file with | sort
    8. 
    """
    pass


def create_temp_folder_and_files(firstfolder):
    """
    Create the temp folder and files
    """
    if args.debug > 0:
        print(f'Creating the temp folders and files')

    merged_first_folder = firstfolder + '-merged'
    try:
        os.mkdir(merged_first_folder)
        if args.debug > 0:
            print(f'Created: {merged_first_folder}')
    except FileExistsError:
        # do nothing, just use it
        if args.debug > 0:
            print(f'Merged folder already existed')

    # List files
    # TODO ignore everything that does not end in .log
    files_firstfolder = [f for f in listdir(firstfolder) if isfile(join(firstfolder, f))]

    for files in files_firstfolder:
        f = open()




def merge_folders(firstfolder, secondfolder):
    """
    Read the two folders
    """

    create_temp_folder_and_files(firstfolder)


# Main
####################
if __name__ == '__main__':  
    print(f'Zeek logs merger. Inserts the second file in the first file.')
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Amount of verbosity. This shows more info about the results.', action='store', required=False, type=int, default=1)
    parser.add_argument('-d', '--debug', help='Amount of debugging. This shows more info about inner working.', action='store', required=False, type=int, default=0)
    parser.add_argument('-1', '--firstfolder', help='Name of the base Zeek folder where you want to insert flows into.', action='store', required=True, type=str)
    parser.add_argument('-2', '--secondfolder', help='Name of the Zeek folder that you want to be inserted into the base Zeek folder.', action='store', required=True, type=str)

    args = parser.parse_args()

    if args.verbose > 0:
        print(f'Merging folder {args.secondfolder} into folder {args.firstfolder}')

    merge_folders(args.firstfolder, args.secondfolder)
