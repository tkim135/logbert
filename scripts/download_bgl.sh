#!/bin/bash

file="../.dataset/bgl/"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

cd $file
zipfile=BGL.tar.gz
wget -O $zipfile https://zenodo.org/record/3227177/files/${zipfile}?download=1 -P $file
tar -xvzf $zipfile
