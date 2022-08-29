#!/usr/bin/env python3
# Tool to merge two zeek logs files by 'inserting' one into the other and moving them in time

# Author: Sebastian Garcia, eldraco@gmail.com
# Stratosphere Laboratory, Czech Technical University in Prague
# www.stratosphereips.org

import argparse
from datetime import datetime


def print_histogram_amount_file(zeekfile, bin_size):
    """
    Get a file, read it, count the amount of flows, print a histogram
    bin_size is in hours 
    """
    if args.debug > 0:
        print(f'[debug] Processing file {zeekfile}, bin={bin_size}')
    bins_data = {}
    max_value = 0
    # Bins step is in seconds since ts is in seconds
    bins_step = 3600 * bin_size
    with open(zeekfile, 'r') as f:
        line = f.readline()
        # loop to ignore comments
        while line and line[0] == '#':
            line = f.readline()
        start_of_bin = False
        end_of_bin = False
        while line:
            # Process flows
            flow_time = float(line.split('\t')[0])
            if args.debug > 0:
                print(f'New flow: {flow_time}. Current start of bin:{start_of_bin}. Current end:{end_of_bin}')
            if not start_of_bin:
                start_of_bin = flow_time
                end_of_bin = start_of_bin + bins_step
                bins_data[start_of_bin] = 1
                if args.debug > 0:
                    print(f'\t[+] Start. ts={flow_time}. Start of bin={start_of_bin}. End of bin={end_of_bin}. Data:{bins_data[start_of_bin]}')
            elif flow_time <= end_of_bin:
                bins_data[start_of_bin] += 1
                if args.debug > 0:
                    print(f'\t[+] Add Flow. ts={flow_time}. Start of bin={start_of_bin}. End of bin={end_of_bin}. Data:{bins_data[start_of_bin]}')
                if bins_data[start_of_bin] > max_value:
                    max_value = bins_data[start_of_bin]
            elif flow_time > end_of_bin:
                start_of_bin = end_of_bin
                end_of_bin = start_of_bin + bins_step
                bins_data[start_of_bin] = 1
                if args.debug > 0:
                    print(f'\t[+] New bin. ts={flow_time}. Start of bin={start_of_bin}. End of bin={end_of_bin}. Data: {bins_data[start_of_bin]}')
            elif flow_time < start_of_bin:
                print(f'Oh no. A flow out of order. Please sort first')
                return True
            line = f.readline()
            # loop to ignore comments at the end
            while line and line[0] == '#':
                line = f.readline()
    # Print histogram
    print(f'Histogram of flows in the zeek file {zeekfile}. Bin size:{bin_size}hs\n')
    prev_key = False
    for key in bins_data:
        hkey = datetime.fromtimestamp(float(key))
        if not prev_key:
            prev_key = key
            prev_hkey = datetime.fromtimestamp(float(prev_key))
        else:
            line_size = int(bins_data[prev_key] / max_value * 100)
            asterics = '*' * line_size
            print(f'{prev_hkey} - {hkey}: {bins_data[prev_key]} {asterics}')
            prev_key = key
            prev_hkey = datetime.fromtimestamp(float(prev_key))
    final_key = prev_key + bins_step
    final_hkey = datetime.fromtimestamp(float(final_key))
    print(f'{prev_hkey} - {final_hkey}: {bins_data[prev_key]} {asterics}')


# Main
####################
if __name__ == '__main__':  
    print(f'Zeek logs merger and analyser')
    print('Zeek flows MUST be sorted!\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Amount of verbosity. This shows more info about the results.', action='store', required=False, type=int, default=1)
    parser.add_argument('-d', '--debug', help='Amount of debugging. This shows more info about inner working.', action='store', required=False, type=int, default=0)
    parser.add_argument('-1', '--firstfile', help='Name of the base Zeek file where you want to insert into.', action='store', required=False, type=str)
    parser.add_argument('-2', '--secondfile', help='Name of the Zeek file that you want to be inserted into the base Zeek file.', action='store', required=False, type=str)
    parser.add_argument('-H', '--histogram', help='Do a histogram of amount of flows per bin in this Zeek file.', action='store', required=False, type=str)
    parser.add_argument('-b', '--histogrambinsize', help='Use this bin size in the histrogram. In number of hours. Defaults to 1.', action='store', required=False, type=int, default=1)

    args = parser.parse_args()

    if args.firstfile and args.secondfile:
        # print(f'first {args.firstfile}, second {args.secondfile}')
        pass

    elif args.histogram:
        print_histogram_amount_file(zeekfile=args.histogram, bin_size=args.histogrambinsize)


