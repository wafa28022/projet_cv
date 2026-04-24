import json
import matplotlib.pyplot as plt

def read_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

def show_image_plt(image, bw=False):
    if bw:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()