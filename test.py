# import os
# import tensorflow as tf
# import tensorflow_compression as tfc
# import argparse
# import cv2
# from glob import glob

# ##process the image into suitable dimension
# def load_img(path):
#     string = tf.io.read_file(path)
#     image = tf.image.decode_image(string, channels=3)
#     return image


# def load_model(args):
#     model = tf.keras.models.load_model(args.model_path,compile=False)
#     return model


# def compress(args, model):

#     os.makedirs('outputs/binary', exist_ok=True)

#     if os.path.isdir(args.image_path):
#         pathes = glob(os.path.join(args.image_path, '*'))
#     else:
#         pathes = [args.image_path]

#     for path in pathes:
#         bitpath = "outputs/binary/{}.pth".format(os.path.basename(path).split('.')[0])

#         image = load_img(path)
#         compressed = model.compress(image)
#         packed = tfc.PackedTensors()
#         packed.pack(compressed)
#         with open(bitpath, "wb") as f:
#             f.write(packed.string)
#         num_pixels = tf.reduce_prod(tf.shape(image)[:-1])
#         bpp = len(packed.string) * 8 / num_pixels

#         print('=============================================================')
#         print(os.path.basename(path))

#         print('bitrate : {0:.4}bpp'.format(bpp))
#         print('=============================================================\n')


# def decompress(model, args, dtypes):
#     os.makedirs("./outputs/reconstruction", exist_ok=True)

#     if os.path.isdir(args.binary_path):
#         pathes = glob(os.path.join(args.binary_path, '*'))
#     else:
#         pathes = [args.binary_path]

#     for path in pathes:

#         print('========================================================================')
#         print('image', os.path.basename(path))

#         with open(path, "rb") as f:
#             packed = tfc.PackedTensors(f.read())
#         tensors = packed.unpack(dtypes)
#         x_hat = model.decompress(*tensors)

#         x_hat_resized = tf.image.resize(x_hat, [240, 424], method=tf.image.ResizeMethod.BILINEAR)
#         x_hat_resized = tf.cast(x_hat_resized, tf.uint8)  # Ensure the image is in uint8 format



#         fakepath = "./outputs/reconstruction/{}.png".format(os.path.basename(path).split('.')[0])
#         string = tf.image.encode_png(x_hat_resized)
#         tf.io.write_file(fakepath, string)
#         print('========================================================================\n')


# from ultralytics import YOLO

# def inference(args):
#     os.makedirs("./outputs/annotated", exist_ok=True)

#     # Load the model
#     model = YOLO("yolo11n-pose.pt")  # Load the official model
#     folder_path = args.reconstruction_path
#     # Iterate over all files in the folder
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".png"):  # Check for PNG files
#             image_path = os.path.join(folder_path, filename)
#             # Perform inference - returns a list of Results objects
#             results_list = model(image_path)
            
#             # Process each result (typically just one for single image)
#             for results in results_list:
#                 # Save the annotated image
#                 output_path = os.path.join('outputs/annotated', filename)
#                 results.save(filename=output_path)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('model_path',type=str, default='./final_model')
#     parser.add_argument('image_path',type=str, default='../datasets/dummy/valid')
#     parser.add_argument('binary_path', type=str, default='./outputs/binary')
#     parser.add_argument('reconstruction_path', type=str, default='./outputs/reconstruction')

#     args = parser.parse_args()
    
#     model = load_model(args)

#     # timer starts 
#     compress(args, model)

#     dtypes = [t.dtype for t in model.decompress.input_signature]
#     decompress(model, args, dtypes)
#         # Replace 'path/to/your/folder' with the actual folder path

#     folder_path = args.binary_path

#     # Loop through all files in the specified folder
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.pth'):
#             file_path = os.path.join(folder_path, filename)
#             file_size = os.path.getsize(file_path)
#             print(f"Size of '{filename}': {file_size} bytes")

#     inference(args)
#     # timer ends



# if __name__ == "__main__":
#     main()




import os
import tensorflow as tf
import tensorflow_compression as tfc
import argparse
import cv2
from glob import glob
import time
import hydra 
from ultralytics import YOLO


def load_img(path):
    string = tf.io.read_file(path)
    image = tf.image.decode_image(string, channels=3)
    return image

def load_model(args):
    model = tf.keras.models.load_model(args.model_path, compile=False)
    return model

def compress(args, model):
    os.makedirs('outputs/binary', exist_ok=True)
    
    if os.path.isdir(args.image_path):
        pathes = glob(os.path.join(args.image_path, '*'))
    else:
        pathes = [args.image_path]
    
    num_images = len(pathes)
    total_time = 0
    
    for path in pathes:
        start_time = time.time()
        
        bitpath = "outputs/binary/{}.pth".format(os.path.basename(path).split('.')[0])
        image = load_img(path)
        compressed = model.compress(image)
        packed = tfc.PackedTensors()
        packed.pack(compressed)
        
        with open(bitpath, "wb") as f:
            f.write(packed.string)
            
        num_pixels = tf.reduce_prod(tf.shape(image)[:-1])
        bpp = len(packed.string) * 8 / num_pixels
        iteration_time = time.time() - start_time
        total_time += iteration_time
        
        print('=============================================================')
        print(os.path.basename(path))
        print('bitrate : {0:.4}bpp'.format(bpp))
        print(f"Processing time: {iteration_time:.4f} seconds")
        print('=============================================================\n')
    
    avg_time = total_time / num_images if num_images > 0 else 0
    print(f"\nCOMPRESSION SUMMARY:")
    print(f"Total images processed: {num_images}")
    print(f"Total compression time: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds\n")
    
    return total_time, num_images

def decompress(model, args, dtypes):
    os.makedirs("./outputs/reconstruction", exist_ok=True)
    
    if os.path.isdir(args.binary_path):
        pathes = glob(os.path.join(args.binary_path, '*'))
    else:
        pathes = [args.binary_path]
    
    num_images = len(pathes)
    total_time = 0
    
    for path in pathes:
        start_time = time.time()
        
        print('========================================================================')
        print('image', os.path.basename(path))
        
        with open(path, "rb") as f:
            packed = tfc.PackedTensors(f.read())
        tensors = packed.unpack(dtypes)
        x_hat = model.decompress(*tensors)
        
        x_hat_resized = tf.image.resize(x_hat, [240, 424], method=tf.image.ResizeMethod.BILINEAR)
        x_hat_resized = tf.cast(x_hat_resized, tf.uint8)
        
        fakepath = "./outputs/reconstruction/{}.png".format(os.path.basename(path).split('.')[0])
        string = tf.image.encode_png(x_hat_resized)
        tf.io.write_file(fakepath, string)
        
        iteration_time = time.time() - start_time
        total_time += iteration_time
        
        print(f"Processing time: {iteration_time:.4f} seconds")
        print('========================================================================\n')
    
    avg_time = total_time / num_images if num_images > 0 else 0
    print(f"\nDECOMPRESSION SUMMARY:")
    print(f"Total images processed: {num_images}")
    print(f"Total decompression time: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds\n")
    
    return total_time, num_images

def inference(args):
    os.makedirs("./outputs/annotated", exist_ok=True)
    model = YOLO("yolo11n-pose.pt")
    folder_path = args.reconstruction_path
    
    # Get list of PNG files
    pathes = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    num_images = len(pathes)
    total_time = 0
    
    for filename in pathes:
        start_time = time.time()
        
        image_path = os.path.join(folder_path, filename)
        results_list = model(image_path)
        
        for results in results_list:
            output_path = os.path.join('outputs/annotated', filename)
            results.save(filename=output_path)
        
        iteration_time = time.time() - start_time
        total_time += iteration_time
        print(f"Processed {filename} in {iteration_time:.4f} seconds")
    
    avg_time = total_time / num_images if num_images > 0 else 0
    print(f"\nINFERENCE SUMMARY:")
    print(f"Total images processed: {num_images}")
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds\n")
    
    return total_time, num_images

@hydra.main(version_base=None, config_path="conf", config_name="config") 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./final_model')
    parser.add_argument('--image_path', type=str, default='../datasets/dummy/valid')
    parser.add_argument('--binary_path', type=str, default='./outputs/binary')
    parser.add_argument('--reconstruction_path', type=str, default='./outputs/reconstruction')
    args = parser.parse_args()
    
    model = load_model(args)
    total_start = time.time()
    
    # Compression
    compress_time, compress_count = compress(args, model)
    
    # Decompression
    dtypes = [t.dtype for t in model.decompress.input_signature]
    decompress_time, decompress_count = decompress(model, args, dtypes)
    
    # File sizes
    folder_path = args.binary_path
    total_size = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pth'):
            file_path = os.path.join(folder_path, filename)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            print(f"Size of '{filename}': {file_size} bytes")
    
    # Inference
    inference_time, inference_count = inference(args)
    
    # Total statistics
    total_time = time.time() - total_start
    print("\nFINAL SUMMARY:")
    print(f"Total execution time: {total_time:.4f} seconds")
    print(f"\nCompression: {compress_time:.4f} seconds ({compress_count} images)")
    print(f"  Average: {compress_time/compress_count:.4f} sec/image")
    print(f"\nDecompression: {decompress_time:.4f} seconds ({decompress_count} images)")
    print(f"  Average: {decompress_time/decompress_count:.4f} sec/image")
    print(f"\nInference: {inference_time:.4f} seconds ({inference_count} images)")
    print(f"  Average: {inference_time/inference_count:.4f} sec/image")
    print(f"\nTotal compressed size: {total_size} bytes")
    print(f"Average per image: {total_size/compress_count if compress_count > 0 else 0:.2f} bytes")

if __name__ == "__main__":
    main()