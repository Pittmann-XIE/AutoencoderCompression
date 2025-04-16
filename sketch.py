import os
import tensorflow as tf
import tensorflow_compression as tfc
import argparse
from glob import glob

def load_model(args):
    model = tf.keras.models.load_model(args.model_path, compile=False)

    if hasattr(model, 'compress'):
        print("\nDecompress output attributes:")
        decompress_model = model.decompress()
        print(dir(decompress_model))  # Inspect the attributes
        if hasattr(decompress_model, 'analysis_transform'):
            print("\nAnalysis Transform:")
            decompress_model.analysis_transform.summary()
        else:
            print("No analysis_transform attribute found in the decompressed model.")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str, default='final_model')
    # parser.add_argument('image_path',type=str, default='kodak/kodim20.png')

    args = parser.parse_args()
    model = load_model(args)

# model = tf.keras.models.load_model(args.model_path,compile=False)
# encoder = model.compress()
# decoder = model.decompress()
# # Print the structure of the encoder
# encoder_json = encoder.to_json()
# print("Encoder structure:")
# print(encoder_json)

# # Print the structure of the decoder
# decoder_json = decoder.to_json()
# print("Decoder structure:")
# print(decoder_json)