# Remote server for the Ilastik plug-in

This repository contains the server component of a plug-in for Ilastik that
allows to use it in conjunction with services running on remote computers.
This allows us to leverage the annotation and visualization capabilities of
Ilastik while executing the bulk of the computation on software components
running on external resources, free of dependencies. This work has been
developed by the [Computer Vision lab at EPFL](https://cvlab.epfl.ch) within
the context of the Human Brain Project.

The server is written in Python 3 using Flask. Please consult `reqs.txt` for a
list of requirements (probably overtuned, but given for reference). You may
generate the server credentials with:
```
openssl req -newkey rsa:2048 -new -nodes -x509 -days 3650 -keyout server.pem -out server.pem
```
and then copy the private key into `server.key` and the public key into
`server.crt`.

External services can be written in any language. We currently have:
* [CCboost component](https://github.com/etrulls/ccboost-service)
* [U-Net component](https://github.com/etrulls/unet-service)

Both are writte on or wrapped by Python code, and must be symlinked into this
folder. You may use different virtual environments: please refer to `server.py`
for details.

The server can then run with:
```
python server.py --port <NUMBER> [--verbose]
```
