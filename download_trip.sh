#!/bin/bash

trip=$1
 
trip_date=$(echo $trip | cut -d'T' -f 1)

trip_year=$(echo $trip_date | cut -d'-' -f 1)

trip_month=$(echo $trip_date | cut -d'-' -f 2)

trip_day=$(echo $trip_date | cut -d'-' -f 3)
 
s3_path="s3://trips-backup/trips/$trip_year/$trip_month/$trip_day/$trip.tar"

# Download the trip from S3

aws s3 cp $s3_path . && tar -xf "$trip.tar" && rm "$trip.tar"
 
# Download MM annotation

# rsync -az --info=progress2 -e 'ssh -i ~/itamar.ginsberg.pem' ubuntu@3.22.147.197:/home/volumes/trips/$trip_year/$trip_month/$trip_day/$trip/ $trip/ --exclude=/zoom* --exclude=/3d_images
 
 
# vehicle_id=$(cat $trip/aidriver_info.json | jq -r '.vehicle_id')

# Download Depth annotation images (rectification fix, etc.)

aws s3 cp s3://depth-annotation/${vehicle_id}/${trip}.tar.gz . && tar -xf "$trip.tar.gz" && rm "$trip.tar.gz"
 
# Download Depth annotation

# aws s3 cp s3://depth-annotation/${trip}.tar.gz . && tar -xf "$trip.tar.gz" && rm "$trip.tar.gz
