#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
docker run -it --rm  --name lego_fab -p 8888:8888 -v "`echo $DIR`":/home/jovyan/ filippo/legofab
