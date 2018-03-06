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
        ]

        # Retrieve uploaded data
        dataPath = curr_path + "/userInput/" + username
        if not os.path.isdir(dataPath):
            os.makedirs(dataPath)
        dataList = getImmediateSubdirectories(dataPath)

        # Get dataset size
        for i in range(len(dataList)):
            with h5py.File("{}/userInput/{}/{}/data.h5".format(curr_path, username, dataList[i]), 'r') as f:
                shape = f['data'].shape
                dataList[i] += " (" + "x".join([str(s) for s in shape]) + ")"

        # TODO append the type of service to the model names
        # Retrieve ccboost models
        modelsPath = curr_path + "/ccboost-service/workspace/" + username + "/models"
        if not os.path.isdir(modelsPath):
            os.makedirs(modelsPath)
        modelList = getImmediateSubdirectories(modelsPath)

        # Return np array of list (Ilastik slots are np arrays)
        f = BytesIO()
        np.savez(f, services=services, data=dataList, models=modelList)
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


@app.route('/api/deleteModel', methods=['GET'])
def delModelFun():
    if checkUser(request):
        username = request.headers['username']
        modelName = request.headers['model-name']

        # Better safe than sorry (we are deleting stuff)
        if str.isalnum(modelName) and str.isalnum(username):
            subprocess.call(["rm", "-rf", curr_path + "/ccboost-service/workspace/" + username + "/models/" + modelName])
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
        dir_path = os.path.dirname(os.path.realpath(__file__))

        fn ="{}/userInput/{}/{}/data.h5".format(dir_path, username, datasetName)
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


@app.route('/api/train', methods=['POST'])
def trainFun():
    if checkUser(request):
        body = list(request.form.keys())[0]
        # keys = [k for k in request.form.keys()]
        # body = keys[0]

        # For some reason the server doesn't receive + signs correctly
        # We translate them to ')' and then back to '+'
        # body = body.translate(
        #     string.maketrans(
        #         custom_base64chars,
        #         std_base64chars))
        body = body.replace(custom_base64chars, std_base64chars)

        # Removed during transfer but necessary for padding
        body = body + '=='
        bodyClear = base64.b64decode(body)

        # Get needed data from request
        labels = np.load(BytesIO(bodyClear))['labels']
        image = np.load(BytesIO(bodyClear))['image']
        username = request.headers['username']
        datasetName = request.headers['dataset-name']
        modelName = request.headers['model-name']
        mirror = request.headers['ccboost-mirror']
        numStumps = request.headers['ccboost-num-stumps']
        insidePixel = request.headers['ccboost-inside-pixel']
        outsidePixel = request.headers['ccboost-outside-pixel']

        # Save labels and data in h5 format
        inputDirectory = curr_path + '/userInput/' + username + "/" + datasetName + "/"
        if not os.path.isdir(inputDirectory):
            os.makedirs(inputDirectory)

        h5 = h5py.File(inputDirectory + '/data.h5', 'w', driver=None)
        # Ilastik sends data in shape (x,y,z,1) when image is grayscale
        # We need (x,y,z)
        if image.shape[3] == 1:
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

        # Transform labels from ilastik conventions to ours
        # ccboost: (255 = positive, 0 = negative, everything else ignored)
        my_dict = {0: 128, 1: 0, 2: 255}
        labels = np.vectorize(my_dict.get, otypes=[np.uint8])(labels)
        h5.create_dataset('data', data=labels)
        h5.close()

        dir_path = curr_path + "/ccboost-service/workspace"

        # write cfg file based on request
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
        f = open(logPath, "w")
        f.write("starting processing \n")
        f.close()
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

        # fetch result and send it back
        h5 = h5py.File("{}/{}/runs/{}/results/{}/out-0-ab-max.h5".format(
            dir_path, username, datasetName, modelName), driver=None)
        data = h5['data']
        data = np.reshape(
            data,
            (data.shape[0],
             data.shape[1],
             data.shape[2],
             1))
        h5.close()

        f = BytesIO()
        np.savez_compressed(f, result=data)
        f.seek(0)
        compressed_data = f.read()
        # compressedData64 = base64.b64encode(compressed_data)

        # os.remove(logPath)
        return compressed_data, 200
    else:
        return 'Not authorized', 401


@app.route('/api/testNewData', methods=['POST'])
def testNewFun():
    if checkUser(request):
        body = list(request.form.keys())[0]
        # print(type(body))
        # dd.io.save("body.h5", {"body": body})

        # Replace ')' with '+'
        # body = body.translate(
        #     string.maketrans(
        #         custom_base64chars,
        #         std_base64chars))
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
        f = open(logPath, "w")
        f.write(">> Starting CCBOOST service\n")
        f.close()
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

        f = BytesIO()
        np.savez_compressed(f, result=data)
        f.seek(0)
        compressed_data = f.read()
        # compressedData64 = base64.b64encode(compressed_data)

        # os.remove(logPath)
        return compressed_data, 200
    else:
        return 'Not authorized', 401


@app.route('/api/testOldData', methods=['GET'])
def testOldFun():
    if checkUser(request):
        username = request.headers['username']
        datasetName = request.headers['dataset-name']
        modelName = request.headers['model-name']
        mirror = request.headers['ccboost-mirror']

        dir_path = curr_path + "/ccboost-service/workspace"

        # write cfg file based on request
        if not os.path.isdir(dir_path + "/" + username + "/config"):
            os.makedirs(dir_path + "/" + username + "/config")

        with open(dir_path + "/" + username + "/config/" + datasetName + ".cfg", "w") as f:
            f.write("root = \'" + curr_path + "/ccboost-service\'\n")
            f.write("dataset_name = \'" + datasetName + "\'\n")
            f.write("stack = \'" + curr_path + "/userInput/" + username + "/" + datasetName + "/data.h5\'\n")
            f.write("model_name = " + modelName + "\n")
            f.write("num_adaboost_stumps = 2000\n")
            f.write("mirror = " + mirror + "\n")

        logPath = curr_path + "/userLogs/" + username
        if not os.path.isdir(logPath):
            os.makedirs(logPath)
        logPath = logPath + "/" + datasetName + "-" + modelName + ".txt"
        if os.path.isfile(logPath):
            os.remove(logPath)
        f = open(logPath, "w")
        f.write("starting processing \n")
        f.close()
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
            f.close(),
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
            curr_path +
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

        f = BytesIO()
        np.savez_compressed(f, image=image, result=data)
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
