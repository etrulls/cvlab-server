from flask import Flask, request, render_template
from flask_restful import Resource, Api
# from PIL import Image
from datetime import datetime
import numpy as np
# import json
import base64
import os
import h5py
# import zlib
import string
import ssl
import csv
import bcrypt
import subprocess
import sys
if sys.version_info[0] < 3:
    from cStringIO import StringIO
else:
    from io import StringIO

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('server.crt', 'server.key')

app = Flask(__name__)
api = Api(app)
# 1GB
app.config['MAX_CONTENT_LENGTH'] = 1000000000

std_base64chars = "+"
custom_base64chars = ")"


def checkUser(request):
    try:
        username = request.headers['username']
        password = request.headers['password']
    except:
        return False
    with open('users.csv') as csvfile:
        reader = csv.reader(csvfile)
        # reader = list(reader)
        entry = [line for line in reader if line[0] == username]
        if len(entry) != 0:
            hashed_password = entry[0][1]
            salt = entry[0][2]

            # hash password sent by client and compare
            combo_password = password.encode('utf-8') + salt
            new_hashed_password = bcrypt.hashpw(combo_password, salt)
            return hashed_password == new_hashed_password

        # user not in database
        else:
            return False


@app.route('/api/login', methods=['GET'])
def loginFun():
    if checkUser(request):
        return '', 200
    else:
        return '', 401


@app.route('/api/getProgress', methods=['GET'])
def progressFun():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        request.headers['username']
        file = open(dir_path + "/userLogs/rizzello/log.txt")
        txt = file.read()
        file.close()
        return txt, 200
    except:
        return "", 404


@app.route('/api/sendPicture', methods=['POST'])
def sendPictureFun():
    if checkUser(request):
        body = request.form.keys()[0]

        # for some reason the server doesn't receive + sings correctly
        # which is why I translate them to ')' and back to '+'
        # when sending data back and forth
        body = body.translate(
            string.maketrans(
                custom_base64chars,
                std_base64chars))

        # removed during transfer but necessary for padding
        body = body + '=='
        bodyClear = base64.b64decode(body)

        # get needed data from request
        labels = np.load(StringIO(bodyClear))['labels']
        image = np.load(StringIO(bodyClear))['image']
        username = request.headers['username']
        tag = datetime.now().strftime("%Y%m%d%H%M%S")

        # save labels and data in h5 format
        inputDirectory = os.getcwd() + '/userInput/' + username + "/" + tag + "/"
        if not os.path.isdir(inputDirectory):
            os.makedirs(inputDirectory)

        h5 = h5py.File(inputDirectory + 'data.h5', driver=None)
        if image.shape[3] == 1:
            # ilastik sends data in shape (x,y,z,1) when image is grayscale
            # we need (x,y,z)
            image = np.reshape(
                image,
                (image.shape[0],
                 image.shape[1],
                 image.shape[2]))
        h5.create_dataset('data', data=image)
        h5.close()
        h5 = h5py.File(inputDirectory + 'labels.h5', driver=None)
        if labels.shape[3] == 1:
            labels = np.reshape(
                labels,
                (labels.shape[0],
                 labels.shape[1],
                 labels.shape[2]))

        # transform labels from ilastik conventions to ccboost ones
        # ccboost: (255 = positive, 0 = negative, everything else ignored)
        my_dict = {0: 128, 1: 0, 2: 255}
        labels = np.vectorize(my_dict.get, otypes=[np.uint8])(labels)
        h5.create_dataset('data', data=labels)
        h5.close()

        dir_path = os.getcwd() + "/ccboost-service"

        # write cfg file based on request
        if not os.path.isdir(dir_path + '/config/' + username):
            os.mkdir(dir_path + '/config/' + username)

        model_name = "ccboost2000stumps"
        file = open(dir_path + "/config/" + username + "/" + tag + ".cfg", "w")
        file.write("dataset_name = \'" + tag + "\'\n")
        file.write("root = \'/cvlabdata1/home/rizzello/ccboost-service\'\n")
        file.write("stack = \'/cvlabdata1/home/rizzello/userInput/" + username + "/" + tag + "/data.h5\'\n") 
        file.write("labels = \'/cvlabdata1/home/rizzello/userInput/" + username + "/" + tag + "/labels.h5\'\n") 
        file.write("model_name = " + model_name + "\n")
        file.write("num_adaboost_stumps = 2000\n")
        file.close()

        subprocess.call(
            ["python",
             "ccboost-service/handler.py",
             "--train",
             dir_path + '/config/' + username + "/" + tag + ".cfg",
             "--username",
             username,
             "--tag",
             tag])

        # fetch result and send it back
        # h5 = h5py.File('ccboost-service/runs/inputDataTest/results/outputModel/out-0-ab-max.h5', driver=None)
        h5 = h5py.File(
            dir_path +
            "/runs/" +
            username +
            "/" +
            tag +
            "/results/" +
            model_name +
            "/out-0-ab-max.h5",
            driver=None)
        data = h5['data']
        data = np.reshape(
            data,
            (data.shape[0],
             data.shape[1],
             data.shape[2],
             1))

        f = StringIO()
        np.savez_compressed(f, image=data)
        f.seek(0)
        compressed_data = f.read()
        # compressedData64 = base64.b64encode(compressed_data)

        return compressed_data, 200
    else:
        # this shouldn't happen since the user can't send data before succesfully logging in
        return '', 401


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7171, ssl_context=context, threaded=True)
