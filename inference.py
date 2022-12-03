def model_inference(image, model_infer):
    output_tensor = model_infer(image)
    output_numpy = output_tensor.detach().numpy()

    return output_numpy