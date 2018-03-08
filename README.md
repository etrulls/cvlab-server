# Server for remote services plug-in for Ilastik

This repository contains the server component of plug-in for Ilastik that
allows to use Ilastik in conjunction with services running on remote computers.
This allows us to leverage the annotation and visualization capabilities of
Ilastik while running the bulk of the computation with software components
running on external resources, free of dependencies. This work has been
developed within the context of the Human Brain Project.

The server is written in Python 3 using Flask. Please consult `reqs.txt` for a
list of requirements (probably overtuned, but given for reference). You may
generate the server credentials with:
```
openssl req -newkey rsa:2048 -nodes -keyout server.key -out server.crt
```

External services can be written in any language. We currently have:
* [CCboost component](https://github.com/etrulls/ccboost-service)
* [U-Net component](https://github.com/etrulls/unet-service)

Both are wrapped by Python code and must be symlinked into this folder. You may
use different virtual environments: please refer to `server.py` for details.

The server can be run with:
```
python server.py --port <NUMBER> [--verbose]
'''
