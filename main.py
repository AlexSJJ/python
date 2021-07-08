from flask import Flask, request, make_response, redirect, url_for, jsonify

import csv
import os
import os.path
import glob
import json
import traceback

from ml import createModel, getModel, getModelType
import datautil

UPLOAD_FOLDER = './data/'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/debug')
def index_debug():
    return app.send_static_file('index_debug.html')


@app.route('/csvdata', methods=['GET', 'POST'])
def csvdata():
    global _hdfs_config, _hdfs_client
    if request.method == 'POST':
        file_object = request.files['file']
        if file_object:
            print(request.files)

            file_name = secure_filename(file_object.filename)
            _hdfs_config["hdfs_data_set_name"] = file_name

            # 上传到服务器缓冲区
            if not os.path.exists(app.config['UPLOAD_FOLDER'] + file_name):
                file_object.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            sleep(1)
            # 构造hdfs_地址
            hdfs_path_to_folder = _hdfs_config["hdfs_root"] + _hdfs_config[
                "hdfs_folder_name"]
            # 从服务器缓冲区上传到hdfs, 子线程上传, 目前在flask中执行shell脚本的功能还无法较好的实现
            # import cmd_helper
            # cmd_helper.cmd_helper("hadoop fs -put {} {}".format(app.config['UPLOAD_FOLDER'] + file_name,
            #                                                     hdfs_path_to_folder + '/' + file_name))
            # 也可以调库
            try:
                _hdfs_client.upload(hdfs_path_to_folder + '/' + file_name, app.config['UPLOAD_FOLDER'] + file_name,
                                    n_threads=3)
            except hdfs.HdfsError:
                _is_finished["load_data_set"] = True
                return "File {} already exists!".format(hdfs_path_to_folder + '/' + file_name)
                pass
            # _hdfs_client.upload("/dataset/user_12345/analyzer.py", app.config['UPLOAD_FOLDER'] + file_name)
            _is_finished["load_data_set"] = True
            return html + "Upload finished!" + "hadoop fs -put {}  {}".format(app.config['UPLOAD_FOLDER'] + file_name,
                                                                              hdfs_path_to_folder + '/' + file_name)
        else:
            return "Cannot upload empty file."
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return json.dumps({})
#         return json.dumps({error:
#                            'You are not allowed to upload such a file.'})
    else:
        flist = [f.replace('./data'+os.sep, '').replace('.csv', '')
                 for f in glob.glob('./data/*.csv')]
        return json.dumps(flist)


@app.route('/data/<dataname>')
def getdata(dataname):
    fpath = './data/' + dataname + '.csv'

    headerOnly = False
    try:
        headerOnly = request.args['headerOnly']
        value = headerOnly.strip().upper()
        if value not in ("0", "FALSE", "F", "N", "NO", "NONE", ""):
            headerOnly = True
    except:
        pass

    if not os.path.isfile(fpath):
        return make_response('<h1>File %s does not exist!</h1>' % fpath)
    else:
        with open(fpath, 'rb') as csvfile:
            if headerOnly:
                return jsonify(name=dataname, csv=csvfile.readline())
            else:
                return jsonify(name=dataname, csv=csvfile.read().decode("UTF-8"))


@app.route('/ml/cls/<action>', methods=['GET'])
def mlclsop(action):
    try:
        if action == "create":
            method = request.args['type']
            model = createModel("Classification", method)
            return jsonify(result="Success", model=model.getId())

        elif action == "train":
            modelId = request.args['id']
            dataName = request.args['data']
            label = request.args['label']
            features = request.args['features'].split(",")

            model = getModel(modelId)

            datadf = datautil.load(dataName)
            labelData = datautil.getColValues(datadf, label)
            featureData = datautil.getColsValues(datadf, features)

            data = dict()
            data["features"] = featureData
            data["label"] = labelData
            model.train(data)

            return jsonify(result="Success", model=modelId)

        elif action == "predict":
            modelId = request.args['id']
            data = json.loads(request.args['data'])

            model = getModel(modelId)
            return jsonify(result="Success", predict=str(model.predict(data)))

        elif action == "predictViz":
            modelId = request.args['id']
            scale = request.args['scale']

            model = getModel(modelId)
            return jsonify(result="Success",
                           predict=str(model.predictViz(int(scale))))

        else:
            return jsonify(result="Failed",
                           msg="Do not support this action {}".format(action))
    except:
        traceback.print_exc()
        return jsonify(result="Failed", msg="Some Exception")


# TODO : lots of overlap with cls operation
@app.route('/ml/regression/<action>', methods=['GET'])
def mlregressionop(action):
    try:
        if action == "create":
            method = request.args['type']
            model = createModel("Regression", method)
            return jsonify(result="Success", model=model.getId())

        elif action == "train":
            modelId = request.args['id']
            dataName = request.args['data']
            label = request.args['target']
            features = request.args['train'].split(",")

            model = getModel(modelId)

            datadf = datautil.load(dataName)
            labelData = datautil.getColValues(datadf, label)
            featureData = datautil.getColsValues(datadf, features)

            data = dict()
            data["train"] = featureData
            data["target"] = labelData
            model.train(data)

            return jsonify(result="Success", model=modelId)

        elif action == "predict":
            modelId = request.args['id']
            data = json.loads(request.args['data'])

            model = getModel(modelId)
            return jsonify(result="Success", predict=str(model.predict(data)))

        elif action == "predictViz":
            modelId = request.args['id']
            scale = request.args['scale']

            model = getModel(modelId)
            return jsonify(result="Success",
                           predict=str(model.predictViz(int(scale))))

        else:
            return jsonify(result="Failed",
                           msg="Do not support this action {}".format(action))
    except:
        traceback.print_exc()
        return jsonify(result="Failed", msg="Some Exception")


@app.route('/ml/cluster/<action>', methods=['GET'])
def mlclusterop(action):
    try:
        if action == "create":
            method = request.args['type']
            model = createModel("Cluster", method)
            return jsonify(result="Success", model=model.getId())

        elif action == "train":
            modelId = request.args['id']
            dataName = request.args['data']
            features = request.args['train'].split(",")

            model = getModel(modelId)

            datadf = datautil.load(dataName)
            featureData = datautil.getColsValues(datadf, features)

            data = dict()
            data["train"] = featureData
            model.train(data)

            return jsonify(result="Success", model=modelId)

        elif action == "predict":
            modelId = request.args['id']
            data = json.loads(request.args['data'])

            model = getModel(modelId)
            return jsonify(result="Success", predict=str(model.predict(data)))

        elif action == "predictViz":
            modelId = request.args['id']
            scale = request.args['scale']

            model = getModel(modelId)
            return jsonify(result="Success",
                           predict=str(model.predictViz(int(scale))))

        else:
            return jsonify(result="Failed",
                           msg="Do not support this action {}".format(action))
    except:
        traceback.print_exc()
        return jsonify(result="Failed", msg="Some Exception")


@app.route('/mlmodel/list/<type>', methods=['GET'])
def mlmodel(type):
    return json.dumps(getModelType(type))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run('0.0.0.0', port=20000, debug=True)

