import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model_unet import build_mobilenetv3_unet

def train_model(X_train, y_train, X_val, y_val, batch_size=8, epochs=30):
    model = build_mobilenetv3_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('models/mobilenetv3_unet.h5', save_best_only=True),
        ReduceLROnPlateau(patience=3)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    return model, history
