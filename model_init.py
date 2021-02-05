"""

 image_retrieval.py  (author: Anson Wong / git: ankonzoid)

 We perform image retrieval using transfer learning on a pre-trained
 VGG image classifier. We plot the k=5 most similar images to our
 query images, as well as the t-SNE visualizations.

"""
import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from src.CV_IO_utils import read_imgs_dir
from src.CV_transform_utils import apply_transformer
from src.CV_transform_utils import resize_img, normalize_img
from src.CV_plot_utils import plot_query_retrieval, plot_tsne, plot_reconstructions
from src.autoencoder import AutoEncoder
from flask import Flask, request
# Run mode: (autoencoder -> simpleAE, convAE) or (transfer learning -> vgg19)
modelName = "convAE"  # try: "simpleAE", "convAE", "vgg19"
trainModel = False
parallel = False  # use multicore processing

# Make paths
dataTrainDir = os.path.join(os.getcwd(), "data", "train")
dataTestDir = os.path.join(os.getcwd(), "data", "test")
outDir = os.path.join(os.getcwd(), "output", modelName)
if not os.path.exists(outDir):
    os.makedirs(outDir)

# Read images
extensions = [".jpg", ".jpeg",".png"]
print("Reading train images from '{}'...".format(dataTrainDir))
imgs_train, species_list = read_imgs_dir(dataTrainDir, extensions, parallel=parallel)
print(species_list)
print("Reading test images from '{}'...".format(dataTestDir))
shape_img = imgs_train[0].shape
print("Image shape = {}".format(shape_img))

# Build models
if modelName in ["simpleAE", "convAE"]:

    # Set up autoencoder
    info = {
        "shape_img": shape_img,
        "autoencoderFile": os.path.join(outDir, "{}_autoecoder.h5".format(modelName)),
        "encoderFile": os.path.join(outDir, "{}_encoder.h5".format(modelName)),
        "decoderFile": os.path.join(outDir, "{}_decoder.h5".format(modelName)),
    }
    model = AutoEncoder(modelName, info)
    model.set_arch()

    if modelName == "simpleAE":
        shape_img_resize = shape_img
        input_shape_model = (model.encoder.input.shape[1],)
        output_shape_model = (model.encoder.output.shape[1],)
        n_epochs = 300
    elif modelName == "convAE":
        shape_img_resize = shape_img
        input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])
        n_epochs = 500
    else:
        raise Exception("Invalid modelName!")

elif modelName in ["vgg19"]:

    # Load pre-trained VGG19 model + higher level layers
    print("Loading VGG19 pre-trained model...")
    model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                        input_shape=shape_img)
    model.summary()

    shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
    input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
    n_epochs = None

else:
    raise Exception("Invalid modelName!")

# Print some model info
print("input_shape_model = {}".format(input_shape_model))
print("output_shape_model = {}".format(output_shape_model))

# Apply transformations to all images
class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

transformer = ImageTransformer(shape_img_resize)
print("Applying image transformer to training images...")
imgs_train_transformed = apply_transformer(imgs_train, transformer, parallel=parallel)
print("Applying image transformer to test images...")

# Convert images to numpy array
X_train = np.array(imgs_train_transformed).reshape((-1,) + input_shape_model)
print(" -> X_train.shape = {}".format(X_train.shape))
# print(" -> X_test.shape = {}".format(X_test.shape))

# Train (if necessary)
if modelName in ["simpleAE", "convAE"]:
    if trainModel:
        model.compile(loss="binary_crossentropy", optimizer="adam")
        model.fit(X_train, n_epochs=n_epochs, batch_size=256)
        model.save_models()
    else:
        model.load_models(loss="binary_crossentropy", optimizer="adam")

# Create embeddings using model
print("Inferencing embeddings using pre-trained model...")
E_train = model.predict(X_train)
E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))


app = Flask(__name__)
UPLOAD_FOLDER = "data/test"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(request.files)
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)

        imgs_test, _ = read_imgs_dir(dataTestDir, extensions, parallel=parallel)
        imgs_test_transformed = apply_transformer(imgs_test, transformer, parallel=parallel)
        X_test = np.array(imgs_test_transformed).reshape((-1,) + input_shape_model)
        E_test = model.predict(X_test)
        E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))

        # Fit kNN model on training images
        print("Fitting k-nearest-neighbour model on training images...")
        knn = NearestNeighbors(n_neighbors=1, metric="cosine")
        knn.fit(E_train_flatten)

        # Perform image retrieval on test images
        print("Performing image retrieval on test images...")
        for i, emb_flatten in enumerate(E_test_flatten):
            _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
            img_query = imgs_test[i] # query image
            imgs_retrieval = [imgs_train[idx] for idx in indices.flatten()]
            break
        #     print(imgs_retrieval) # retrieval images
        #     outFile = os.path.join(outDir, "{}_retrieval_{}.png".format(modelName, i))
        #     plot_query_retrieval(img_query, imgs_retrieval, outFile)

        # # Plot t-SNE visualization
        # print("Visualizing t-SNE on training images...")
        # outFile = os.path.join(outDir, "{}_tsne.png".format(modelName))
        # plot_tsne(E_train_flatten, imgs_train, outFile)
        try:
            specie = species_list[imgs_train.index(imgs_retrieval[0])]
            specie = specie.split(".")[0].split("_")[0].replace("-"," ")
        except:
            specie = "carangoides ferdau"
        os.remove(f"data/test/{file1.filename}")
        return specie

app.run("0.0.0.0", 8000)