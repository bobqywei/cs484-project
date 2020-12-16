#!/bin/bash

files=(2011_09_26_calib.zip
2011_09_26_drive_0001
2011_09_26_drive_0005
2011_09_26_drive_0011
2011_09_26_drive_0014
2011_09_26_drive_0015
2011_09_26_drive_0017
2011_09_26_drive_0018
2011_09_26_drive_0019
2011_09_26_drive_0022
2011_09_26_drive_0028
2011_09_26_drive_0032
2011_09_26_drive_0035
2011_09_26_drive_0039
2011_09_26_drive_0051
2011_09_26_drive_0057
2011_09_26_drive_0061
2011_09_26_drive_0070
2011_09_26_drive_0079
2011_09_26_drive_0087
2011_09_26_drive_0091
2011_09_26_drive_0095
2011_09_26_drive_0104
2011_09_26_drive_0113
2011_09_28_calib.zip
2011_09_28_drive_0001
2011_09_29_calib.zip
2011_09_29_drive_0004
2011_09_29_drive_0026
2011_09_30_calib.zip
2011_09_30_drive_0020
2011_09_30_drive_0028
2011_09_30_drive_0033
2011_09_30_drive_0034
2011_10_03_calib.zip
2011_10_03_drive_0034
2011_10_03_drive_0042)

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                shortname=$i'_sync.zip'
                fullname=$i'/'$i'_sync.zip'
        else
                shortname=$i
                fullname=$i
        fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname
done
