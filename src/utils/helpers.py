import json 
import matplotlib.pyplot as plt

def read_json(json_path):
    file = open(json_path)
    data = json.load(file)
    file.close()
    return data

def save_json(data, path):
    jsonString = json.dumps(data)
    jsonFile = open(path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

def show_image_plt(image, bw=False):
    if bw:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')  
    plt.show()