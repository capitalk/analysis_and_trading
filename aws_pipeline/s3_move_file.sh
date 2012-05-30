#!/bin/bash

export PYTHONPATH=:/usr/local/lib/python2.7/site-packages:/usr/local/lib/python2.7/site-packages:/usr/local/lib/python2.7/site-packages

export AWS_SECRET_ACCESS_KEY=8J9VG9WlYCOmT6tq6iyC7h1K2rOk8v+q8FehsBdv
export AWS_ACCESS_KEY_ID=AKIAITZSJIMPWRM54I4Q


python_cmd=`which python`

$python_cmd ~/upload_and_queue.py -n true -f $1
