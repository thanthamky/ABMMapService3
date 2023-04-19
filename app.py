import os
from flask import Flask, render_template, request, send_from_directory, jsonify

from map_maker import MapMaker
from agents_locator import *
from array_compressor import *

import numpy as np

app = Flask(__name__)
# ... other code ...

mapmaker = MapMaker('./data/grid_40m_nksw_lc_rast_32647_msk.tif')

def list_files_in_directory_by_pattern(directory_path, string_to_match):
    """
    List all files in the given directory containing a specific string in their names and return their paths as a list.

    Parameters:
    directory_path (str): The path of the directory.
    string_to_match (str): The string to match in the file names.

    Returns:
    list: A list of file paths.
    """
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                  if os.path.isfile(os.path.join(directory_path, f)) and string_to_match in f]
    return sorted(file_paths)

@app.route('/')
def show_map():
    param = request.args.get('sim_id')
    
    print(f'Receive getting map .. sim ID {param}')
    
    map_list = list_files_in_directory_by_pattern('./map_store', param+"_agent")
    
    print(f'... Found {map_list} {len(map_list)} maps in total')
    step = len(map_list)
    
    if step == 0: print(f'Map of sim ID {param} is not found!')
    
    return render_template('map_old.html', param=param, step=step)
    #return {'param': param}

@app.route('/map_store/<path:filename>')
def serve_map_store_file(filename):
    return send_from_directory(os.path.join(app.root_path, "map_store"), filename)

@app.route('/lib/<path:filename>')
def serve_lib_file(filename):
    print(os.path.join(os.getcwd(), "lib"))
    return send_from_directory(os.path.join(app.root_path, "lib"), filename)

@app.route('/data/<path:filename>')
def serve_data_file(filename):
    print(os.path.join(os.getcwd(), "data"))
    return send_from_directory(os.path.join(app.root_path, "data"), filename)

@app.route('/make_map', methods=['POST'])
def make_map():
    
    print("MAPPING START....")
    # RECEIVE ARGUMENTS
    sim_id = request.args.get('simid')
    step = eval(request.args.get('step'))
    map_type = request.args.get('type') #['agent', 'lc', 'income']
    start_i = eval(request.args.get('i'))
    start_j = eval(request.args.get('j'))
    shape = request.args.get('shape')

    #data = request.get_json()['data']

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_content = file.read()
    

    # CONVERT DATA

    if map_type == 'agent':
        data_dtype = np.int32
    elif map_type == 'lc':
        data_dtype = object
    elif map_type == 'income':
        data_dtype = np.float64

    print(f'receive map {map_type} define dtype as {data_dtype}')
    #data_dtype = 'int16' #income 'float64' , #lc 'object', #agent

    #data_shape = (3144, 4726)
    data_shape = eval(shape)


    map_data = decompress_array_nodecode(file_content, dtype=data_dtype, shape=data_shape)
    
    print(f'\n\n shape data: {map_data.shape}\n sim_id {sim_id} \n step {step} \n start i {start_i} \n start j {start_j} \n\n')
    
    saved_path = mapmaker.make_map(map_data, map_type, start_i, start_j, sim_id, step, data_shape[0], data_shape[1], './map_store' )
    
    print(f'saved at {saved_path}')
    return {"status" : f"Saved at {saved_path}"}

app.run(debug=True, port=5050)