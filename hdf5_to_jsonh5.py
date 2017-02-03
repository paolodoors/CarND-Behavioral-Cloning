import argparse
from keras.models import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model')
    args = parser.parse_args()

    model = load_model(args.model)

    model.summary()

    json_save = args.model.replace('hdf5', 'json')
    weights_save = args.model.replace('hdf5', 'h5')

    json_string = model.to_json()
    with open(json_save, 'w') as json_file:
        json_file.write(json_string)
    model.save_weights(weights_save)
