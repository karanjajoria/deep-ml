def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network.

    Args:
        layers: List of dictionaries, each describing a layer.

    Returns:
        Total number of trainable parameters (int).
    """
    # Formula (For Dense) => input*output + bias {For each layer}
    # therefore ([784(input)*128(output)] + 128(bias)) + ([128(input)*10(output) + 10(bias)]) == 101770

    total_params = 0

    for element in layers:
        if element["type"] == "dense":
            if "bias" in element:
                if element["bias"] == True:
                    temp = element["input_size"] * element["output_size"] + element["output_size"]
                    total_params += temp
                elif element["bias"] == False:
                    temp = element["input_size"] * element["output_size"]
                    total_params += temp
            else:
                temp = element["input_size"] * element["output_size"] + element["output_size"]
                total_params += temp

        if element["type"] == "conv2d":
            if element["bias"] == True:
                temp = element["in_channels"] * element["out_channels"] * element["kernel_size"]**2 + element["out_channels"]
                total_params += temp
            elif element["bias"] == False:
                temp = element["in_channels"] * element["out_channels"] * (element["kernel_size"]**2)
                total_params += temp

    pass

    return total_params