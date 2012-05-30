#!/bin/bash

export PYTHONPATH=:/usr/local/lib/python2.7/site-packages:/usr/local/lib/python2.7/site-packages:/usr/local/lib/python2.7/site-packages

# User access
export AWS_SECRET_ACCESS_KEY=eBbYhZwykjZka5HbS1LMFuwq8VlxfmiVrxkkKNin
export AWS_ACCESS_KEY_ID=AKIAIVTJV27C2UID5EWA


python_cmd=`which python`

$python_cmd ~/s3_multipart_upload_dir.py -f $1 $2
