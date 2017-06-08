from __future__ import print_function
from flask import Flask, request, render_template
from flask_restful import Resource, Api
from datetime import datetime
import numpy as np
import base64
import os
import h5py
import string
import ssl
import csv
import subprocess
import sys
from passlib.hash import sha256_crypt
import argparse
# import json
# from PIL import Image
# import zlib

# Set workspace folder to that holding the current file
# Services should be symlinked inside here
curr_path = os.path.dirname(os.path.realpath(__file__))


if sys.version_info[0] < 3:
    from cStringIO import StringIO as StringBytesIO
else:
    from io import BytesIO as StringBytesIO

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('server.crt', 'server.key')

app = Flask(__name__)
api = Api(app)
# 1GB
app.config['MAX_CONTENT_LENGTH'] = 1000000000

std_base64chars = "+"
custom_base64chars = ")"


def getImmediateSubdirectories(path):
    return [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]


def checkUser(request):
    try:
        username = request.headers['username']
        password = request.headers['password']
    except:
        return False
    with open('users.csv') as csvfile:
        reader = csv.reader(csvfile)
        entry = [line for line in reader if line[0] == username]
        if len(entry) != 0:
            hashed_password = entry[0][1]
            return sha256_crypt.verify(
                password,
                hashed_password)

        # user not in database
        else:
            return False


@app.route('/api/login', methods=['GET'])
def loginFun():
    if checkUser(request):
        username = request.headers['username']
        modelsPath = os.getcwd() + "/ccboost-service/workspace/" + username + "/models"  # Note : For now there is only service, future work might add more services than just ccboost
        # for example the user would choose the service he wants to use before logging in
        if not os.path.isdir(modelsPath):
            os.makedirs(modelsPath)
        modelList = getImmediateSubdirectories(modelsPath)
        dataPath = os.getcwd() + "/userInput/" + username
        if not os.path.isdir(dataPath):
            os.makedirs(dataPath)
        dataList = getImmediateSubdirectories(dataPath)

        # return np array of format [[dataset1,dataset2...],[modelA, modelB...]]
        # Ilastik slots are np arrays and returning this in that format makes things simpler
        # toReturn = np.array([dataList, modelsList])
        f = StringBytesIO()
        np.savez(f, data=dataList, models=modelList)
        f.seek(0)
        payload = f.read()
        return payload, 200
    else:
        return '', 401

@app.route('/api/deleteDataset', methods=['GET'])
def delDatasetFun():
    if checkUser(request):
        username = request.headers['username'].encode('ascii','ignore')
        datasetName = request.headers['dataset_name'].encode('ascii','ignore')
        if str.isalnum(datasetName) and str.isalnum(username): #username check might be redundant, but better safe than sorry.
            subprocess.call(["rm","-rf",curr_path+"/userInput/"+username+"/"+datasetName])
            subprocess.call(["rm","-rf", curr_path+"/ccboost-service/workspace/"+username+"/runs/"+ datasetName])
            return '', 200
        else:
            return 'user or dataset name contains illegal characters',400
    else:
        return '', 401


@app.route('/api/downloadDataset', methods=['GET'])
def downloadFun():
    if checkUser(request):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        username = request.headers['username']
        datasetName = request.headers['dataset_name']
        h5 = h5py.File(
            dir_path +
            "/userInput/" +
            username +
            "/" +
            datasetName +
            "/data.h5",
            driver=None)
        data = h5['data']
        f = StringBytesIO()
        np.savez_compressed(f, data=data)
        f.seek(0)
        compressed_data = f.read()
        return compressed_data, 200
    else:
        return '', 401


@app.route('/api/getProgress', methods=['GET'])
def progressFun():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        username = request.headers['username']
        file = open(dir_path + "/userLogs/" + username + "/log.txt")
        txt = file.read()
        file.close()
        return txt, 200
    except:
        return "", 404


@app.route('/api/train', methods=['POST'])
def trainFun():
    if checkUser(request):
        body = request.form.keys()[0]
        # keys = [k for k in request.form.keys()]
        # body = keys[0]

        # for some reason the server doesn't receive + signs correctly
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
        labels = np.load(StringBytesIO(bodyClear))['labels']
        image = np.load(StringBytesIO(bodyClear))['image']
        username = request.headers['username']
        datasetName = request.headers['datasetName']
        modelName = request.headers['modelName']

        # save labels and data in h5 format
        inputDirectory = os.getcwd() + '/userInput/' + username + "/" + datasetName + "/"
        if not os.path.isdir(inputDirectory):
            os.makedirs(inputDirectory)

        h5 = h5py.File(inputDirectory + 'data.h5', 'w', driver=None)
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
        h5 = h5py.File(inputDirectory + 'labels.h5', 'w', driver=None)
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

        dir_path = os.getcwd() + "/ccboost-service/workspace"

        # write cfg file based on request
        if not os.path.isdir(dir_path + "/" + username + "/config"):
            os.makedirs(dir_path + "/" + username + "/config")

        file = open(dir_path + "/" + username + "/config/" + datasetName + ".cfg", "w")
        file.write("dataset_name = \'" + datasetName + "\'\n")
        file.write("root = \'" + curr_path + "/ccboost-service\'\n")
        file.write("stack = \'" + curr_path + "/userInput/" + username + "/" + datasetName + "/data.h5\'\n")
        file.write("labels = \'" + curr_path + "/userInput/" + username + "/" + datasetName + "/labels.h5\'\n")
        file.write("model_name = " + modelName + "\n")
        file.write("num_adaboost_stumps = 2000\n")
        file.close()

        logPath = os.getcwd() + "/userLogs/" + username
        if not os.path.isdir(logPath):
            os.makedirs(logPath)
        logPath = logPath + "/log.txt"
        file = open(logPath, "w")
        file.write("starting processing \n")
        file.close()
        p = subprocess.Popen(
            ["python",
             "ccboost-service/handler.py",
             "--train",
             dir_path + "/" + username + "/config/" + datasetName + ".cfg",
             "--username",
             username], stdout=subprocess.PIPE, stderr=sys.stdout.fileno(), bufsize=1)
        for line in iter(p.stdout.readline, b''):
            file = open(logPath, "a")
            file.write(line)
            file.close(),
        p.stdout.close()
        p.wait()

        # fetch result and send it back
        h5 = h5py.File(
            dir_path +
            "/" +
            username +
            "/runs/" +
            datasetName +
            "/results/" +
            modelName +
            "/out-0-ab-max.h5",
            driver=None)
        data = h5['data']
        data = np.reshape(
            data,
            (data.shape[0],
             data.shape[1],
             data.shape[2],
             1))
        h5.close()

        f = StringBytesIO()
        np.savez_compressed(f, result=data)
        f.seek(0)
        compressed_data = f.read()
        # compressedData64 = base64.b64encode(compressed_data)

        os.remove(logPath)
        return compressed_data, 200
    else:
        # this shouldn't happen since the user can't send data before succesfully logging in
        return '', 401


@app.route('/api/testNewData', methods=['POST'])
def testNewFun():
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
        image = np.load(StringBytesIO(bodyClear))['image']
        username = request.headers['username']
        datasetName = request.headers['datasetName']
        modelName = request.headers['modelName']

        # save data in h5 format
        inputDirectory = os.getcwd() + '/userInput/' + username + "/" + datasetName + "/"
        if not os.path.isdir(inputDirectory):
            os.makedirs(inputDirectory)

        h5 = h5py.File(inputDirectory + 'data.h5', 'w', driver=None)
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

        dir_path = os.getcwd() + "/ccboost-service/workspace"

        # write cfg file based on request
        if not os.path.isdir(dir_path + "/" + username + "/config"):
            os.makedirs(dir_path + "/" + username + "/config")

        file = open(dir_path + "/" + username + "/config/" + datasetName + ".cfg", "w")
        file.write("root = \'" + curr_path + "/ccboost-service\'\n")
        file.write("dataset_name = \'" + datasetName + "\'\n")
        file.write("stack = \'" + curr_path + "/userInput/" + username + "/" + datasetName + "/data.h5\'\n")
        file.write("model_name = " + modelName + "\n")
        file.write("num_adaboost_stumps = 2000\n")
        file.close()

        logPath = os.getcwd() + "/userLogs/" + username
        if not os.path.isdir(logPath):
            os.makedirs(logPath)
        logPath = logPath + "/log.txt"
        # if os.path.isfile(logPath):
        #     os.remove(logPath)
        file = open(logPath, "w")
        file.write(">> Starting CCBOOST service\n")
        file.close()
        p = subprocess.Popen(
            ["python",
             "ccboost-service/handler.py",
             "--test",
             dir_path + "/" + username + "/config/" + datasetName + ".cfg",
             "--username",
             username], stdout=subprocess.PIPE, bufsize=1)
        for line in iter(p.stdout.readline, b''):
            file = open(logPath, "a")
            file.write(line)
            file.close(),
        p.stdout.close()
        p.wait()

        # fetch result and send it back
        h5 = h5py.File(
            dir_path +
            "/" +
            username +
            "/runs/" +
            datasetName +
            "/results/" +
            modelName +
            "/out-0-ab-max.h5",
            driver=None)
        data = h5['data']
        data = np.reshape(
            data,
            (data.shape[0],
             data.shape[1],
             data.shape[2],
             1))
        h5.close()

        f = StringBytesIO()
        np.savez_compressed(f, result=data)
        f.seek(0)
        compressed_data = f.read()
        # compressedData64 = base64.b64encode(compressed_data)

        os.remove(logPath)
        return compressed_data, 200
    else:
        # this shouldn't happen since the user can't send data before succesfully logging in
        return '', 401


@app.route('/api/testOldData', methods=['GET'])
def testOldFun():
    if checkUser(request):
        username = request.headers['username']
        datasetName = request.headers['datasetName']
        modelName = request.headers['modelName']

        dir_path = os.getcwd() + "/ccboost-service/workspace"

        # write cfg file based on request
        if not os.path.isdir(dir_path + "/" + username + "/config"):
            os.makedirs(dir_path + "/" + username + "/config")

        file = open(dir_path + "/" + username + "/config/" + datasetName + ".cfg", "w")
        file.write("root = \'" + curr_path + "/ccboost-service\'\n")
        file.write("dataset_name = \'" + datasetName + "\'\n")
        file.write("stack = \'" + curr_path + "/userInput/" + username + "/" + datasetName + "/data.h5\'\n")
        file.write("model_name = " + modelName + "\n")
        file.write("num_adaboost_stumps = 2000\n")
        file.close()

        logPath = os.getcwd() + "/userLogs/" + username
        if not os.path.isdir(logPath):
            os.makedirs(logPath)
        logPath = logPath + "/log.txt"
        file = open(logPath, "w")
        file.write("starting processing \n")
        file.close()
        p = subprocess.Popen(
            ["python",
             "ccboost-service/handler.py",
             "--test",
             dir_path + "/" + username + "/config/" + datasetName + ".cfg",
             "--username",
             username], stdout=subprocess.PIPE, bufsize=1)
        for line in iter(p.stdout.readline, b''):
            file = open(logPath, "a")
            file.write(line)
            file.close(),
        p.stdout.close()
        p.wait()

        # fetch result and send it back
        h5 = h5py.File(
            dir_path +
            "/" +
            username +
            "/runs/" +
            datasetName +
            "/results/" +
            modelName +
            "/out-0-ab-max.h5",
            driver=None)
        data = h5['data']
        data = np.reshape(
            data,
            (data.shape[0],
             data.shape[1],
             data.shape[2],
             1))
        h5.close()

        h5 = h5py.File(
            os.getcwd() +
            "/userInput/" +
            username +
            "/" +
            datasetName +
            "/data.h5",
            driver=None)
        image = h5['data']
        image = np.reshape(
            image,
            (image.shape[0],
             image.shape[1],
             image.shape[2],
             1))
        h5.close()

        f = StringBytesIO()
        np.savez_compressed(f, image=image, result=data)
        f.seek(0)
        compressed_data = f.read()
        # compressedData64 = base64.b64encode(compressed_data)

        os.remove(logPath)
        return compressed_data, 200
    else:
        # this shouldn't happen since the user can't send data before succesfully logging in
        return '', 401


if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser(description='CVLAB server')
    parser.add_argument(
        '--port',
        type=int,
        help='Server port',
        default=7170)
    params = parser.parse_args()

    # Run flask
    app.run(host='0.0.0.0', port=params.port, ssl_context=context, threaded=True)

