import os
import tensorflow as tf
import tensorflow_compression as tfc
from glob import glob
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO
from types import SimpleNamespace

def load_imgs(paths):
    images = []
    for path in paths:
        string = tf.io.read_file(path)
        image = tf.image.decode_image(string, channels=3)
        images.append(image)
    return tf.stack(images)

def load_img(path):
    string = tf.io.read_file(path)
    image = tf.image.decode_image(string, channels=3)
    return image

def load_model(args):
    model = tf.keras.models.load_model(args.model_path, compile=False)
    return model

def compress(args, model, batch_size):
    os.makedirs(args.binary_path, exist_ok=True)
    
    if os.path.isdir(args.image_path):
        paths = sorted(glob(os.path.join(args.image_path, '*')))
    else:
        paths = [args.image_path]
    
    num_images = len(paths)
    total_time = 0

    for batch_start in range(0, num_images, batch_size):
        batch_paths = paths[batch_start:batch_start + batch_size]
        start_time = time.time()
        
        # Load images in batch
        images = load_imgs(batch_paths)
        
        # Compress the batch
        compressed_tensors = model.compress(images)
        
        # For each image in the batch, save the compressed data
        batch_size_actual = len(batch_paths)
        for i in range(batch_size_actual):
            compressed = [t[i] for t in compressed_tensors]
            packed = tfc.PackedTensors()
            packed.pack(compressed)
            bitpath = os.path.join(args.binary_path, "{}.pth".format(os.path.basename(batch_paths[i]).split('.')[0]))
            with open(bitpath, "wb") as f:
                f.write(packed.string)
        
        iteration_time = time.time() - start_time
        total_time += iteration_time

    avg_time = total_time / num_images if num_images > 0 else 0
    print(f"\nCOMPRESSION SUMMARY:")
    print(f"Total images processed: {num_images}")
    print(f"Total compression time: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds\n")

    return total_time, num_images

# def compress(args, model, batch_size):
#     os.makedirs(args.binary_path, exist_ok=True)
    
#     if os.path.isdir(args.image_path):
#         paths = sorted(glob(os.path.join(args.image_path, '*')))
#     else:
#         paths = [args.image_path]
    
#     num_images = len(paths)
#     total_time = 0

#     # print(f"Processing {num_images} images for compression.")

#     for batch_start in range(0, num_images, batch_size):
#         batch_paths = paths[batch_start:batch_start + batch_size]
#         start_time = time.time()

#         for path in batch_paths:
#             # Load image
#             image = load_img(path)
#             # Compress the image
#             compressed = model.compress(image)
#             # Pack and save compressed data
#             packed = tfc.PackedTensors()
#             packed.pack(compressed)
#             bitpath = os.path.join(args.binary_path, f"{os.path.splitext(os.path.basename(path))[0]}.pth")
#             with open(bitpath, "wb") as f:
#                 f.write(packed.string)

#         iteration_time = time.time() - start_time
#         total_time += iteration_time

#     avg_time = total_time / num_images if num_images > 0 else 0
#     print(f"\nCOMPRESSION SUMMARY:")
#     print(f"Total images processed: {num_images}")
#     print(f"Total compression time: {total_time:.4f} seconds")
#     print(f"Average time per image: {avg_time:.4f} seconds\n")

#     return total_time, num_images

def decompress(model, args, dtypes, batch_size=16):
    os.makedirs(args.reconstruction_path, exist_ok=True)
    
    if os.path.isdir(args.binary_path):
        paths = sorted(glob(os.path.join(args.binary_path, '*')))
    else:
        paths = [args.binary_path]
    
    num_images = len(paths)
    total_time = 0

    print(f"Processing {num_images} images for decompression.")

    for batch_start in range(0, num_images, batch_size):
        batch_paths = paths[batch_start:batch_start + batch_size]
        start_time = time.time()

        for path in batch_paths:
            with open(path, "rb") as f:
                packed = tfc.PackedTensors(f.read())
            tensors = packed.unpack(dtypes)
            # Decompress the image
            x_hat = model.decompress(*tensors)
            # Process and save the image
            x_hat_resized = tf.image.resize(x_hat, [240, 424], method=tf.image.ResizeMethod.BILINEAR)
            x_hat_resized = tf.cast(x_hat_resized, tf.uint8)
            fakepath = os.path.join(args.reconstruction_path, f"{os.path.splitext(os.path.basename(path))[0]}.png")
            string = tf.image.encode_png(x_hat_resized)
            tf.io.write_file(fakepath, string)

        iteration_time = time.time() - start_time
        total_time += iteration_time

    avg_time = total_time / num_images if num_images > 0 else 0
    print(f"\nDECOMPRESSION SUMMARY:")
    print(f"Total images processed: {num_images}")
    print(f"Total decompression time: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds\n")

    return total_time, num_images


def inference(args, batch_size):
    os.makedirs(args.annotated_path, exist_ok=True)
    model = YOLO("yolo11n-pose.pt")
    model.to("cuda")
    folder_path = args.reconstruction_path
    
    # Get list of PNG files
    paths = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    num_images = len(paths)
    total_time = 0

    for batch_start in range(0, num_images, batch_size):
        batch_filenames = paths[batch_start:batch_start + batch_size]
        start_time = time.time()
        
        batch_image_paths = [os.path.join(folder_path, filename) for filename in batch_filenames]
        results_list = model(batch_image_paths)
        
        for filename, results in zip(batch_filenames, results_list):
            output_path = os.path.join(args.annotated_path, filename)
            results.save(output_path)
        
        iteration_time = time.time() - start_time
        total_time += iteration_time

    avg_time = total_time / num_images if num_images > 0 else 0
    print(f"\nINFERENCE SUMMARY:")
    print(f"Total images processed: {num_images}")
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds\n")

    return total_time, num_images

@hydra.main(version_base=None, config_path="conf", config_name="config") 
def main(cfg: DictConfig) -> None:
    # Print the configuration
    print(OmegaConf.to_yaml(cfg))
    # Load the configuration
    model_path = cfg.train_setting.model_path
    image_path = cfg.train_setting.image_path
    binary_path = cfg.train_setting.binary_path
    reconstruction_path = cfg.train_setting.reconstruction_path
    annotated_path = cfg.train_setting.annotated_path
    # Print the loaded configuration
    print(f"Model path: {model_path}")
    print(f"Image path: {image_path}")
    print(f"Binary path: {binary_path}")
    print(f"Reconstruction path: {reconstruction_path}")
    print(f"Annotated path: {annotated_path}")

    args = SimpleNamespace()
    args.model_path = model_path
    args.image_path = image_path
    args.binary_path = binary_path
    args.reconstruction_path = reconstruction_path
    args.annotated_path = annotated_path

    # Check if paths exist
    if not os.path.exists(args.image_path):
        print(f"Error: The image directory {args.image_path} does not exist.")
        return
    if not os.listdir(args.image_path):
        print(f"Error: The image directory {args.image_path} is empty.")
        return
    if not os.path.exists(args.model_path):
        print(f"Error: The model at {args.model_path} does not exist.")
        return

    model = load_model(args)

    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print(f"Using device: {device}")

    total_start = time.time()
    print("Timer starts")

    try:
        # Compression
        compress_time, compress_count = compress(args, model, batch_size=16)
    except Exception as e:
        print(f"Compression failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Decompression
        dtypes = [t.dtype for t in model.decompress.input_signature]
        decompress_time, decompress_count = decompress(model, args, dtypes, batch_size=16)
    except Exception as e:
        print(f"Decompression failed: {e}")
        traceback.print_exc()

    try:
        # Inference
        inference_time, inference_count = inference(args, batch_size=16)
    except Exception as e:
        print(f"Inference failed: {e}")
        traceback.print_exc()
    
    # File sizes
    folder_path = args.binary_path
    total_size = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pth'):
            file_path = os.path.join(folder_path, filename)
            file_size = os.path.getsize(file_path)
            total_size += file_size
      
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