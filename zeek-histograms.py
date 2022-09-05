#!/usr/bin/env python3
# Tool to print a histogram of amount of flows in zeek files. 
# It works with any type of zeek log file
# You can define the size of the bin as a float 

# Author: Sebastian Garcia, eldraco@gmail.com
# Stratosphere Laboratory, Czech Technical University in Prague
# www.stratosphereips.org

import argparse
from datetime import datetime
import json
import math
import time


def print_histogram_amount_file(zeekfile, bin_size):
    """
    Get a file, read it, count the amount of flows, print a histogram
    bin_size is in hours 
    """
    bins_data = {}
    max_value = 0

    # Process bins steps from h, m and s to seconds
    if 'h' in bin_size[-1]:
        bins_step = 3600 * float(bin_size.split('h')[0])
    elif 'm' in bin_size[-1]:
        bins_step = 60 * float(bin_size.split('m')[0])
    elif 's' in bin_size[-1]:
        bins_step = float(bin_size.split('s')[0])
    else:
        bins_step = float(bin_size)

    if args.debug > 0:
        print(f'[debug] Processing file {zeekfile}, bin={bin_size}')

    with open(zeekfile, 'r') as f:
        line = f.readline()
        # loop to ignore comments
        while line and line[0] == '#':
            line = f.readline()
        start_of_bin = False
        end_of_bin = False
        while line:
            # Process flows
            try:
                # Is it a json file?
                flow_time = float(json.loads(line)['ts'])
            except json.decoder.JSONDecodeError:
                # It may be a TAB separated file
                flow_time = float(line.split('\t')[0])
            if args.debug > 0:
                print(f'New flow: {flow_time}. Current start of bin:{start_of_bin}. Current end:{end_of_bin}')
            if not start_of_bin:
                start_of_bin = flow_time
                end_of_bin = start_of_bin + bins_step
                bins_data[start_of_bin] = 1
                if not max_value:
                    max_value = 1
                if args.debug > 0:
                    print(f'\t[+] Start and New bin. ts={flow_time}. Start of bin={start_of_bin}. End of bin={end_of_bin}. Amount in bin:{bins_data[start_of_bin]}')
            elif flow_time > start_of_bin and flow_time <= end_of_bin:
                bins_data[start_of_bin] += 1
                if args.debug > 0:
                    print(f'\t[+] Add Flow. ts={flow_time}. Start of bin={start_of_bin}. End of bin={end_of_bin}. Amount in bin:{bins_data[start_of_bin]}')
                if bins_data[start_of_bin] > max_value:
                    max_value = bins_data[start_of_bin]
            elif flow_time < start_of_bin:
                if args.debug > 2:
                    print(f'\tThis flow was out of order. Flowtime: {flow_time}. Bin Start: {start_of_bin}, Bin End: {end_of_bin}')
                # Find the bin of this flow
                found_its_bin = False
                if args.debug > 3:
                    print(f'\t\tChecking each bin')
                for key in bins_data.keys():
                    if args.debug > 3:
                        print(f'\t\tBin Start: {key}. End:{(key + bins_step)}')
                    if flow_time > key and flow_time <= (key + bins_step):
                        # The flow belongs to this bin
                        if args.debug > 3:
                            print(f'\t\tGoes Here. The out-of-order flow was added to bin with start: {key}')
                        found_its_bin = True
                        break
                    else:
                        if args.debug > 3:
                            print(f'\t\tNot in this bin {key}')
                        pass
                if not found_its_bin:
                    if args.debug > 3:
                        print(f'\t\tThe flow is not in any known bin. Lets create new ones')
                    # The flow is from before the fist bin.
                    # What is the diff in time between the flow and the first bin we created?
                    difference = start_of_bin - flow_time
                    # How many bins_step is this diff?
                    steps = difference / bins_step
                    # Convert that into a ceiling. math.ceil(3.8) = 4
                    bins_to_create = math.ceil(steps)
                    if args.debug > 4:
                        print(f'\t\t[*] Diff: {difference}. Steps:{steps}. to create: {bins_to_create}')
                    # We need to create 'bins_to_create' new bins in the past.
                    new_key = list(bins_data.keys())[0]
                    for i in range(bins_to_create):
                        to_create = i + 1
                        new_key =  new_key - bins_step
                        try:
                            data = bins_data[new_key]
                            # The bin was already there, just continue
                        except KeyError:
                            # The bin was not there, create it with no flows
                            bins_data[new_key] = 0
                        if args.debug > 3:
                            print(f'\t\tNew bin created with key: {new_key}')
                    bins_data[new_key] = 1
                    # Add the weird flow to the last created bin
                    if args.debug > 3:
                        print(f'\tCurrent bins: {bins_data}')

            elif flow_time > end_of_bin:
                # Create empty bins in the middle future
                # What is the diff in time between the flow and the first bin we created?
                difference = flow_time - start_of_bin
                # How many bins_step is this diff?
                steps = difference / bins_step
                # Convert that into a floor. math.floor(3.8) = 3
                bins_to_create = math.floor(steps) 
                if args.debug > 4:
                    print(f'\t\tThis flow is in the future. Current bins {bins_data}')
                    print(f'\t\t[*] Diff: {difference}. Steps:{steps}. to create: {bins_to_create}')
                # We need to create 'bins_to_create' new bins in the past.
                new_key = end_of_bin
                for i in range(bins_to_create):
                    to_create = i + 1
                    try:
                        # Do we already have this bin?
                        data = bins_data[new_key]
                        new_key = new_key + bins_step
                        # The bin was already there, just continue
                    except KeyError:
                        # The bin was not there, create it with no flows
                        bins_data[new_key] = 0
                # Add the flow to the last bin
                start_of_bin = new_key
                end_of_bin = start_of_bin + bins_step
                bins_data[new_key] = 1
                if args.debug > 0:
                    print(f'\t\t[+] New bin created. Start of bin={start_of_bin}. End of bin={end_of_bin}. Amount in bin: {bins_data[start_of_bin]}')
                    print(f'\tCurrent bins_data: {bins_data}')

            line = f.readline()
            # loop to ignore comments at the end
            while line and line[0] == '#':
                line = f.readline()
    # Print histogram
    if bins_data:
        print(f'Histogram of flows in the zeek file {zeekfile}. Bin size:{bin_size}\n')
        tz = time.tzname
        print(f'Current time zone in this system is: {tz[0]}. All flows ')
        prev_key = False
        for key in sorted(bins_data.keys()):
            hkey = datetime.fromtimestamp(float(key))
            if not prev_key:
                prev_key = key
                prev_hkey = datetime.fromtimestamp(float(prev_key))
            else:
                line_size = int(bins_data[prev_key] / max_value * args.maxasterics)
                asterics = '*' * line_size
                print(f'{prev_hkey} - {hkey}: {bins_data[prev_key]} {asterics}')
                prev_key = key
                prev_hkey = datetime.fromtimestamp(float(prev_key))
        final_key = prev_key + bins_step
        final_hkey = datetime.fromtimestamp(float(final_key))
        line_size = int(bins_data[prev_key] / max_value * args.maxasterics)
        asterics = '*' * line_size
        print(f'{prev_hkey} - {final_hkey}: {bins_data[prev_key]} {asterics}')
    else:
        print('The log file did not have any flows')


# Main
####################
if __name__ == '__main__':  
    print(f'Zeek logs histogram creator')
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Amount of verbosity. This shows more info about the results.', action='store', required=False, type=int, default=1)
    parser.add_argument('-d', '--debug', help='Amount of debugging. This shows more info about inner working.', action='store', required=False, type=int, default=0)
    parser.add_argument('-f', '--zeeklogfile', help='Do a histogram of amount of flows per bin in this Zeek file.', action='store', required=False, type=str)
    parser.add_argument('-b', '--histogrambinsize', help='Use this bin size in the histrogram. Use "h" for hours, "m" for minutes, "s" for seconds. No letter means seconds. Defaults to 1h.', action='store', required=False, type=str, default="1h")
    parser.add_argument('-m', '--maxasterics', help='Maximum amount of asterics for the histogram graph. Defaults to 100.', action='store', required=False, type=int, default=100)

    args = parser.parse_args()

    if args.zeeklogfile:
        print_histogram_amount_file(zeekfile=args.zeeklogfile, bin_size=args.histogrambinsize)
