# Stratosphere Datasets Tools
[![Docker Image CI](https://github.com/stratosphereips/DatasetsTools/actions/workflows/docker-image.yml/badge.svg)](https://github.com/stratosphereips/DatasetsTools/actions/workflows/docker-image.yml)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/stratosphereips/DatasetsTools/main)
![Docker Pulls](https://img.shields.io/docker/pulls/stratosphereips/datatoolset?color=green)

A set of tools to work with the [Stratosphere datasets](https://www.stratosphereips.org/datasets-overview).

## Zeek Histogram Creator

The tool ```zeek-histograms.py``` creates histograms from any Zeek flow files. The tool supports bin sizes in hours, minutes and seconds (E.g.: 1h, 1m, or 1s). The flows do not have to be sorted before hand, the tool will recognize its time and place it on the proper bin.

Example:

```bash
$ python3 zeek-histograms.py -b 10s -f testing-datasets/test10-mixed-zeek-dir-conn.log
Zeek logs histogram creator
Histogram of flows in the zeek file testing-datasets/test10-mixed-zeek-dir-conn.log. Bin size:10s

Current time zone in this system is: CET. All flows
2020-10-06 17:32:36.785668 - 2020-10-06 17:32:46.785668:  1 **
2020-10-06 17:32:46.785668 - 2020-10-06 17:32:56.785668: 11 *******************************
2020-10-06 17:32:56.785668 - 2020-10-06 17:33:06.785668:  0
2020-10-06 17:33:06.785668 - 2020-10-06 17:33:16.785668:  7 ********************
2020-10-06 17:33:16.785668 - 2020-10-06 17:33:26.785668: 35 ****************************************************************************************************
2020-10-06 17:33:26.785668 - 2020-10-06 17:33:36.785668:  1 **
```
