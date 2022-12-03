from preprocess import result

def result_information():
    inference_output = results.as_numpy('output_0')
    top5_output = inference_output[:5]
    return top5_output

def result_info_fashion(result_model):
    dict_result = {
        "T-shirt/top" : result_model[0],
        "Trouser": result_model[1],
        "Pullover": result_model[2]
        "Dress":result_model[3],
        "Coat":result_model[4],
        "Sandal": result_model[5],
        "Shirt": result_model[6],
        "Sneaker": result_model[7],
        "Bag": result_model[8],
        "Ankle boot": result_model[9]
    }
    return dict_result