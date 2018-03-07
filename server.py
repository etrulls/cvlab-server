from __future__ import print_function
from flask import Flask, request, render_template
from flask_restful import Resource, Api
from datetime import datetime
import numpy as np
import base64
import os
import h5py
import ssl
import csv
import subprocess
import sys
from passlib.hash import sha256_crypt
import argparse
from io import BytesIO
# import deepdish as dd
import psutil

# Set workspace folder
curr_path = os.path.dirname(os.path.realpath(__file__))

# Services should be symlinked inside here
if not os.path.isdir(curr_path + "/ccboost-service"):
    raise RuntimeError("Cannot find service 'ccboost'. Forgot to symlink?")

# Create API
app = Flask(__name__)
api = Api(app)
# 1GB
app.config['MAX_CONTENT_LENGTH'] = 1000000000

# Fix for windows (client side)
std_base64chars = "+"
custom_base64chars = ")"


def getCCboostModels(path):
    folders = [name for name in os.listdir(path)
               if os.path.isdir(os.path.join(path, name))]

    # Folder must contain models
    folders = [
        name for name in folders if os.path.isfile(
            os.path.join(path, name, 'stumps.cfg'))]

    return folders


def getDatasets(path):
    folders = [name for name in os.listdir(path)
               if os.path.isdir(os.path.join(path, name))]

    # Folder must contain models
    folders = [
        name for name in folders if os.path.isfile(
            os.path.join(path, name, 'data.h5'))]

    return folders


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


# Log in and get list of available services
@app.route('/api/login', methods=['GET'])
def loginFun():
    # Log in and get list of services
    if checkUser(request):
        username = request.headers['username']

        # Services currently integrated
        # Do not swap the order: ccboost/train is used as the default for the UI
        services = [
            'CCboost (train)',
            'CCboost (test)',
            'U-Net GAD mouse (test)',
        ]

        # Retrieve uploaded data
        dataPath = curr_path + "/userInput/" + username
        if not os.path.isdir(dataPath):
            os.makedirs(dataPath)
        dataList = getDatasets(dataPath)

        # Get dataset size
        for i in range(len(dataList)):
            with h5py.File("{}/userInput/{}/{}/data.h5".format(curr_path, username, dataList[i]), 'r') as f:
                shape = f['data'].shape
                dataList[i] += " (" + "x".join([str(s) for s in shape]) + ")"

        # Retrieve ccboost models
        ccboostModelsPath = curr_path + "/ccboost-service/workspace/" + username + "/models"
        if not os.path.isdir(ccboostModelsPath):
            os.makedirs(ccboostModelsPath)
        ccboostModelList = getCCboostModels(ccboostModelsPath)

        # List GAD models
        unetGadModelList = ["GAD mouse CH1", "GAD mouse CH2"]

        # Return np array of list (Ilastik slots are np arrays)
        f = BytesIO()
        np.savez(f, services=services, data=dataList, ccboostModels=ccboostModelList, unetGadModels=unetGadModelList)
        f.seek(0)
        payload = f.read()
        return payload, 200
    else:
        return 'Not authorized', 401


@app.route('/api/deleteDataset', methods=['GET'])
def delDatasetFun():
    if checkUser(request):
        username = request.headers['username']
        datasetName = request.headers['dataset-name']

        # Better safe than sorry (we are deleting stuff)
        if str.isalnum(datasetName) and str.isalnum(username):
            # Subprocess calls are better
            # Server may hang for a bit otherwise and make the client unresponsive
            subprocess.call(["rm", "-rf", curr_path + "/userInput/" + username + "/" + datasetName])
            subprocess.call(["rm", "-rf", curr_path + "/ccboost-service/workspace/" + username + "/runs/" + datasetName])
            # os.system("rm -rf {}/userInput/{}/{}".format(curr_path, username, datasetName))
            # os.system("rm -rf {}/ccboost-service/workspace/{}/runs/{}".format(curr_path, username, datasetName))
            return 'Success', 200
        else:
            return 'User or dataset name contains illegal characters', 400
    else:
        return 'Unauthorized', 401


@app.route('/api/downloadDataset', methods=['GET'])
def downloadFun():
    if checkUser(request):
        username = request.headers['username']
        datasetName = request.headers['dataset-name']

        fn ="{}/userInput/{}/{}/data.h5".format(curr_path, username, datasetName)
        if os.path.isfile(fn):
            h5 = h5py.File(fn, driver=None)
            data = h5['data']
            f = BytesIO()
            np.savez_compressed(f, data=data)
            f.seek(0)
            payload = f.read()
            return payload, 200
        else:
            return 'No such file', 404
    else:
        return 'Unauthorized', 401


@app.route('/api/deleteModel', methods=['GET'])
def delModelFun():
    if checkUser(request):
        username = request.headers['username']
        modelName = request.headers['model-name']
        serviceName = request.headers['service-name']

        # Only CCboost models can be deleted for now
        if serviceName != 'CCboost (test)':
            return 'Cannot delete models for service "{}"'.format(serviceName), 401

        # Better safe than sorry (we are deleting stuff)
        if str.isalnum(modelName) and str.isalnum(username):
            subprocess.call(["rm", "-rf", curr_path + "/ccboost-service/workspace/" + username + "/models/" + modelName])
            return 'Success', 200
        else:
            return 'User or dataset name contains illegal characters', 400
    else:
        return 'Unauthorized', 401


@app.route('/api/getProgress', methods=['GET'])
def progressFun():
    if checkUser(request):
        try:
            username = request.headers['username']
            dataset_name = request.headers['dataset-name']
            model_name = request.headers['model-name']
            folder = curr_path + "/userLogs/" + username
            # print(folder)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            with open(folder + "/" + dataset_name + "-" + model_name + ".txt") as f:
                txt = f.read().encode('utf-8')
                # print('Debugging: "{}"'.format(txt))
            return txt, 200
        except:
            return 'Error opening file', 404
    else:
        return 'Unauthorized', 401


# For training we always need to send at least the labels
# So we're sending the image as well... This is lazy, TODO fix it
@app.route('/api/train', methods=['POST'])
def trainFun():
    if checkUser(request):
        body = list(request.form.keys())[0]

        # For some reason the server doesn't receive + signs correctly
        body = body.replace(custom_base64chars, std_base64chars)

        # Removed during transfer but necessary for padding
        body = body + '=='
        bodyClear = base64.b64decode(body)

        # Get basic headers
        username = request.headers['username']
        datasetName = request.headers['dataset-name']
        modelName = request.headers['model-name']

        # Save labels and data in h5 format
        inputDirectory = curr_path + '/userInput/' + username + "/" + datasetName + "/"
        if not os.path.isdir(inputDirectory):
            os.makedirs(inputDirectory)

        # Labels are mandatory
        labels = np.load(BytesIO(bodyClear))['labels']
        with h5py.File(inputDirectory + 'labels.h5', 'w', driver=None) as h5:
            if labels.shape[3] == 1:
                labels = np.reshape(
                    labels,
                    (labels.shape[0],
                     labels.shape[1],
                     labels.shape[2]))
            # Transform labels from ilastik conventions to ours
            # ccboost: (255 = positive, 0 = negative, everything else ignored)
            my_dict = {0: 128, 1: 0, 2: 255}
            labels = np.vectorize(my_dict.get, otypes=[np.uint8])(labels)
            h5.create_dataset('data', data=labels)

        # Save data if necessary
        # dataOnServer = request.headers['data-on-server']
        dataOnServer = False
        if not dataOnServer:
            image = np.load(BytesIO(bodyClear))['image']
            with h5py.File(inputDirectory + '/data.h5', 'w', driver=None) as h5:
                # Ilastik sends data in shape (x,y,z,1) when image is grayscale
                # We need (x,y,z)
                if image.shape[3] == 1:
                    image = np.reshape(
                        image,
                        (image.shape[0],
                         image.shape[1],
                         image.shape[2]))
                h5.create_dataset('data', data=image)

        # Service and parameters
        serviceName = request.headers['service-name']
        if serviceName == 'CCboost (train)':
            out = ccboost_train(username,
                                datasetName,
                                modelName,
                                request.headers['ccboost-mirror'],
                                request.headers['ccboost-num-stumps'],
                                request.headers['ccboost-inside-pixel'],
                                request.headers['ccboost-outside-pixel'])
        else:
            return 'Cannot recognize service "{}"'.format(serviceName), 500

        # fetch result and send it back
        h5 = h5py.File(out, driver=None)
        result = h5['data']
        result = np.reshape(
            result,
            (result.shape[0],
             result.shape[1],
             result.shape[2],
             1))
        h5.close()

        f = BytesIO()
        np.savez_compressed(f, result=result)
        f.seek(0)
        compressed_data = f.read()
        # compressedData64 = base64.b64encode(compressed_data)

        # os.remove(logPath)
        return compressed_data, 200
    else:
        return 'Not authorized', 401


@app.route('/api/testWithData', methods=['POST'])
def testNewFun():
    if checkUser(request):
        body = list(request.form.keys())[0]
        # Replace ')' with '+'
        body = body.replace(custom_base64chars, std_base64chars)

        # Removed during transfer but necessary for padding
        body = body + '=='
        bodyClear = base64.b64decode(body)

        # Get needed data from request
        image = np.load(BytesIO(bodyClear))['image']
        username = request.headers['username']
        datasetName = request.headers['dataset-name']
        modelName = request.headers['model-name']
        mirror = request.headers['ccboost-mirror']

        # Save data in h5 format
        inputDirectory = curr_path + '/userInput/' + username + "/" + datasetName + "/"
        if not os.path.isdir(inputDirectory):
            os.makedirs(inputDirectory)

        h5 = h5py.File(inputDirectory + 'data.h5', 'w', driver=None)
        # Ilastik sends data in shape (x,y,z,1) when image is grayscale
        # We need (x,y,z)
        if image.ndim == 4:
            if image.shape[3] == 1:
                image = np.reshape(
                    image,
                    (image.shape[0],
                     image.shape[1],
                     image.shape[2]))
        h5.create_dataset('data', data=image)
        h5.close()

        out = ccboost_test(username, datasetName, modelName, mirror)

        # fetch result and send it back
        h5 = h5py.File(out, driver=None)
        result = h5['data']
        result = np.reshape(
            result,
            (result.shape[0],
             result.shape[1],
             result.shape[2],
             1))
        h5.close()

        f = BytesIO()
        np.savez_compressed(f, result=result)
        f.seek(0)
        compressed_data = f.read()
        # compressedData64 = base64.b64encode(compressed_data)

        # os.remove(logPath)
        return compressed_data, 200
    else:
        return 'Not authorized', 401


# Same as above but no data or labels are sent
# So no need to retrieve the body, headers are sufficient
@app.route('/api/testWithoutData', methods=['GET'])
def testOldFun():
    if checkUser(request):
        username = request.headers['username']
        datasetName = request.headers['dataset-name']
        modelName = request.headers['model-name']
        mirror = request.headers['ccboost-mirror']

        out = ccboost_test(username, datasetName, modelName, mirror)

        # fetch result and send it back
        h5 = h5py.File(out, driver=None)
        result = h5['data']
        result = np.reshape(
            result,
            (result.shape[0],
             result.shape[1],
             result.shape[2],
             1))
        h5.close()

        # h5 = h5py.File(
        #     curr_path +
        #     "/userInput/" +
        #     username +
        #     "/" +
        #     datasetName +
        #     "/data.h5",
        #     driver=None)
        # image = h5['data']
        # image = np.reshape(
        #     image,
        #     (image.shape[0],
        #      image.shape[1],
        #      image.shape[2],
        #      1))
        # h5.close()

        f = BytesIO()
        # np.savez_compressed(f, image=image, result=data)
        np.savez_compressed(f, result=result)
        f.seek(0)
        compressed_data = f.read()
        # compressedData64 = base64.b64encode(compressed_data)

        # os.remove(logPath)
        return compressed_data, 200
    else:
        return 'Not authorized', 401


@app.route('/api/getUsage', methods=['GET'])
def getUsage():
    if checkUser(request):
        try:
            # Retrieve CPU and memory usage
            try:
                cpu_load = psutil.cpu_percent()
                cpu_count = psutil.cpu_count()
                vm = psutil.virtual_memory()
                mem_perc = vm.percent
                mem_total = vm.total

                s = 'Server load: {:.1f}% ({:d} cores)\nMemory: {:.1f}% (total: {:.1f} Gb)'.format(
                    cpu_load, cpu_count, mem_perc, mem_total / 1e9)
                return s, 200
            except:
                return 'Server: Could not retrieve', 500
        except:
            return 'Error opening file', 404
    else:
        return 'Unauthorized', 401


def ccboost_train(username, datasetName, modelName, mirror, numStumps, insidePixel, outsidePixel):
    dir_path = curr_path + "/ccboost-service/workspace"

    # Write cfg file based on request
    if not os.path.isdir(dir_path + "/" + username + "/config"):
        os.makedirs(dir_path + "/" + username + "/config")

    with open(dir_path + "/" + username + "/config/" + datasetName + ".cfg", "w") as f:
        f.write("dataset_name = \'" + datasetName + "\'\n")
        f.write("root = \'" + curr_path + "/ccboost-service\'\n")
        f.write("stack = \'" + curr_path + "/userInput/" + username + "/" + datasetName + "/data.h5\'\n")
        f.write("labels = \'" + curr_path + "/userInput/" + username + "/" + datasetName + "/labels.h5\'\n")
        f.write("model_name = " + modelName + "\n")
        f.write("num_adaboost_stumps = " + numStumps +"\n")
        f.write("mirror = " + mirror + "\n")
        f.write("ignore = " + insidePixel + ", " + outsidePixel)

    logPath = curr_path + "/userLogs/" + username
    if not os.path.isdir(logPath):
        os.makedirs(logPath)
    logPath = logPath + "/" + datasetName + "-" + modelName + ".txt"
    if os.path.isfile(logPath):
        os.remove(logPath)
    with open(logPath, "w") as f:
        f.write("-- Starting CCBOOST service --\n")
    cmd = ["python",
           "ccboost-service/handler.py",
           "--train",
           dir_path + "/" + username + "/config/" + datasetName + ".cfg",
           "--username",
           username]
    # print("DEBUGGING : {}".format(" ".join(cmd)))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stdout.fileno(), bufsize=1)
    for line in iter(p.stdout.readline, b''):
        if verbose:
            print(line.strip().decode("utf-8"))
        f = open(logPath, "a")
        f.write(line.decode("utf-8"))
        f.close(),
    p.stdout.close()
    p.wait()

    fn = "{}/{}/runs/{}/results/{}/out-0-ab-max.h5".format(
        dir_path, username, datasetName, modelName)
    return fn


def ccboost_test(username, datasetName, modelName, mirror):
    dir_path = curr_path + "/ccboost-service/workspace"

    # write cfg file based on request
    if not os.path.isdir(dir_path + "/" + username + "/config"):
        os.makedirs(dir_path + "/" + username + "/config")

    f = open(dir_path + "/" + username + "/config/" + datasetName + ".cfg", "w")
    f.write("root = \'" + curr_path + "/ccboost-service\'\n")
    f.write("dataset_name = \'" + datasetName + "\'\n")
    f.write("stack = \'" + curr_path + "/userInput/" + username + "/" + datasetName + "/data.h5\'\n")
    f.write("model_name = " + modelName + "\n")
    f.write("num_adaboost_stumps = 2000\n")
    f.write("mirror = " + mirror + "\n")
    f.close()

    logPath = curr_path + "/userLogs/" + username
    if not os.path.isdir(logPath):
        os.makedirs(logPath)
    logPath = logPath + "/" + datasetName + "-" + modelName + ".txt"
    if os.path.isfile(logPath):
        os.remove(logPath)
    with open(logPath, "w") as f:
        f.write("-- Starting CCBOOST service --\n")
    p = subprocess.Popen(
        ["python",
         "ccboost-service/handler.py",
         "--test",
         dir_path + "/" + username + "/config/" + datasetName + ".cfg",
         "--username",
         username], stdout=subprocess.PIPE, bufsize=1)
    for line in iter(p.stdout.readline, b''):
        if verbose:
            print(line.strip().decode("utf-8"))
        f = open(logPath, "a")
        f.write(line.decode("utf-8"))
        f.close()
    p.stdout.close()
    p.wait()

    fn = dir_path + "/" + username + "/runs/" + datasetName + \
        "/results/" + modelName + "/out-0-ab-max.h5"
    return fn


if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser(description="CVLAB server")
    parser.add_argument("--port", type=int, default=7777, help="Port")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    params = parser.parse_args()

    global verbose
    verbose = params.verbose

    # Run flask
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain('server.crt', 'server.key')
    app.run(host='0.0.0.0', port=params.port, ssl_context=context, threaded=True)
