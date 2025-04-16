import tensorflow as tf
from tensorflow_addons.optimizers import CyclicalLearningRate
from modules import AutoencoderModel
from glob import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')  # Use the first GPU
            print(f"Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(e)

    print(OmegaConf.to_yaml(cfg))
    # Configuration from Hydra
    BATCH_SIZE = cfg.train_setting.batch_size
    HEIGHT = cfg.train_setting.height
    WIDTH = cfg.train_setting.width
    AUTOTUNE = tf.data.AUTOTUNE

    # tf.random.set_seed(cfg.train_setting.seed if 'seed' in cfg.train_setting else 42)

    # # 1. PNG Data Loading Functions
    # def load_image(file_path):
    #     img = tf.io.read_file(file_path)
    #     img = tf.image.decode_png(img, channels=3)
    #     img = tf.image.resize(tf.cast(img, dtype=tf.float32)/255., 
    #                         size=(HEIGHT, WIDTH))
    #     return img

    # def prepare_dataset(image_paths, batch_size):
    #     dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    #     dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    #     dataset = dataset.shuffle(buffer_size=len(image_paths))
    #     dataset = dataset.batch(batch_size)
    #     dataset = dataset.prefetch(AUTOTUNE)
    #     return dataset

    # # 2. Load Dataset
    # train_images = glob(os.path.join(cfg.train_setting.train_dir, "*.png"))
    # val_images = glob(os.path.join(cfg.train_setting.val_dir, "*.png"))

    # print(f"Found {len(train_images)} training images")
    # print(f"Found {len(val_images)} validation images")

    # # 3. Load Pre-trained Model
    # try:
    #     model = tf.keras.models.load_model(cfg.train_setting.pretrained_path, compile=False)
    #     print("Loaded full model successfully")
    # except:
    #     model = AutoencoderModel(1)
    #     model.build(input_shape=(None, HEIGHT, WIDTH, 3))
    #     model.load_weights(cfg.train_setting.pretrained_path)
    #     print("Loaded weights successfully")

    # # 4. Fine-Tuning Setup
    # for layer in model.layers[:int(len(model.layers)*cfg.train_setting.freeze_ratio)]:
    #     layer.trainable = False

    # for i, layer in enumerate(model.layers):
    #     print(f"Layer {i}: {layer.name} - Trainable: {layer.trainable}")

    # clr = CyclicalLearningRate(
    #     initial_learning_rate=cfg.train_setting.init_lr,
    #     maximal_learning_rate=cfg.train_setting.max_lr,
    #     scale_fn=lambda x: 1/(2.**(x-1)),
    #     step_size=2 * (len(train_images)//BATCH_SIZE)
    # )

    # # 5. Compile Model
    # model.compile(optimizer=tf.keras.optimizers.Adam(clr))

    # # 6. Callbacks
    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(
    #         cfg.train_setting.checkpoint_path,
    #         save_best_only=True,
    #         save_weights_only=True,
    #         monitor='val_loss'
    #     ),
    #     tf.keras.callbacks.EarlyStopping(
    #         patience=cfg.train_setting.patience,
    #         restore_best_weights=True,
    #         monitor='val_loss',
    #         min_delta=cfg.train_setting.min_delta
    #     ),
    #     tf.keras.callbacks.TensorBoard(
    #         log_dir=cfg.train_setting.log_dir,
    #         histogram_freq=1
    #     ),
    #     tf.keras.callbacks.CSVLogger(cfg.train_setting.history_path)
    # ]

    # # 7. Fine-Tune
    # history = model.fit(
    #     x=prepare_dataset(train_images, BATCH_SIZE),
    #     validation_data=prepare_dataset(val_images, BATCH_SIZE),
    #     epochs=cfg.train_setting.epochs,
    #     callbacks=callbacks,
    #     verbose=1
    # )

    # # 8. Save Final Model
    # model.save(cfg.train_setting.final_model_path)
    # print(f"Fine-tuning complete! Saved model to '{cfg.train_setting.final_model_path}'")

if __name__ == "__main__":
    main()