# Stratosphere Datasets Tools
[![Docker Image CI](https://github.com/stratosphereips/DatasetsTools/actions/workflows/docker-image.yml/badge.svg)](https://github.com/stratosphereips/DatasetsTools/actions/workflows/docker-image.yml)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/stratosphereips/DatasetsTools/main)
![Docker Pulls](https://img.shields.io/docker/pulls/stratosphereips/datatoolset?color=green)

A set of tools to work with the [Stratosphere datasets](https://www.stratosphereips.org/datasets-overview):
* `zeek-histograms.py`: create histograms based on Zeek log files.
* `merge-zeek-files.py`: merge two Zeek log files.

## Zeek Histogram Creator

The tool ```zeek-histograms.py``` creates histograms from any Zeek flow files. The tool supports bin sizes in hours, minutes and seconds (E.g.: 1h, 1m, or 1s). The flows do not have to be sorted before hand, the tool will recognize its time and place it on the proper bin.

Example:

```bash
$ python3 zeek-histograms.py -b 10m -f dataset/001-zeek-scenario-malicious/conn.log

Zeek logs histogram creator
Histogram of flows in the zeek file dataset/001-zeek-scenario-malicious/conn.log. Bin size:10m

Current time zone in this system is: CET. All flows
1970-01-01 00:50:19.981745 - 1970-01-01 01:00:19.981745: 1
1970-01-01 01:00:19.981745 - 1970-01-01 01:10:19.981745: 318 ****************************************************************************************************
1970-01-01 01:10:19.981745 - 1970-01-01 01:20:19.981745: 166 ****************************************************
1970-01-01 01:20:19.981745 - 1970-01-01 01:30:19.981745: 152 ***********************************************
1970-01-01 01:30:19.981745 - 1970-01-01 01:40:19.981745: 152 ***********************************************
1970-01-01 01:40:19.981745 - 1970-01-01 01:50:19.981745: 160 **************************************************
1970-01-01 01:50:19.981745 - 1970-01-01 02:00:19.981745: 3
```

# Docker Image

To test the `datatoolset` image is working correctly, run the following command. The command will create a new container and run the `zeek-histograms` tool on a Zeek testing dataset: 
```bash
docker run --rm -it --name stratosphere_datatoolset stratosphereips/datatoolset:latest python3 zeek-histograms.py -b 10m -f dataset/001-zeek-scenario-malicious/conn.log
```

Use the public docker image with the latest version and run the tools directly on the container:

```bash
docker run -v /full/path/to/logs/:/datasetstool/testing-datasets --name stratosphere_datatoolset --rm -it stratosphereips/datatoolset:latest /bin/bash
```
